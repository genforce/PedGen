"""Lightning wrapper of the pytorch model."""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from smplx import SMPLLayer
from smplx.lbs import vertices2joints
from torch.optim.lr_scheduler import MultiStepLR

from pedgen.model.diffusion_utils import (MLPHead, MotionTransformer,
                                          cosine_beta_schedule, get_dct_matrix)
from pedgen.utils.rot import positional_encoding_2d, rotation_6d_to_matrix


class PedGenModel(LightningModule):
    """Lightning model for pedestrian generation."""

    def __init__(
        self,
        gpus: int,
        batch_size_per_device: int,
        diffuser_conf: Dict,
        noise_steps: int,
        ddim_timesteps: int,
        optimizer_conf: Dict,
        mod_train: float,
        num_sample: int,
        lr_scheduler_conf: Dict,
        use_goal: bool = False,
        use_image: bool = False,
        use_beta: bool = False,
    ) -> None:
        super().__init__()
        self.noise_steps = noise_steps
        self.ddim_timesteps = ddim_timesteps
        self.beta = cosine_beta_schedule(self.noise_steps)
        alpha = 1. - self.beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)
        self.diffuser = MotionTransformer(**diffuser_conf)

        self.criterion = F.mse_loss
        self.criterion_traj = F.l1_loss

        self.optimizer_conf = optimizer_conf
        self.lr_scheduler_conf = lr_scheduler_conf
        self.gpus = gpus
        self.batch_size_per_device = batch_size_per_device
        self.mod_train = mod_train

        self.num_sample = num_sample
        self.use_goal = use_goal
        self.use_beta = use_beta
        self.use_image = use_image

        self.smpl = SMPLLayer(model_path="smpl", gender='neutral')
        for param in self.smpl.parameters():
            param.requires_grad = False

        if self.use_goal:
            self.goal_embed = MLPHead(3, diffuser_conf["latent_dim"])
        if self.use_beta:
            self.beta_embed = MLPHead(10, diffuser_conf["latent_dim"])

        if self.use_image:
            img_ch_in = 40  # hardcoded
            self.img_embed = MLPHead(img_ch_in, diffuser_conf["latent_dim"])
            self.img_cross_attn_norm = nn.LayerNorm(diffuser_conf["latent_dim"])
            self.img_cross_attn = nn.MultiheadAttention(
                diffuser_conf["latent_dim"],
                diffuser_conf["num_heads"],
                dropout=0.2,
                batch_first=True)

        self.cond_embed = nn.Parameter(torch.zeros(diffuser_conf["latent_dim"]))

        self.mask_embed = nn.Parameter(torch.zeros(
            diffuser_conf["input_feats"]))

        self.ddim_timestep_seq = np.asarray(
            list(
                range(0, self.noise_steps,
                      self.noise_steps // self.ddim_timesteps))) + 1
        self.ddim_timestep_prev_seq = np.append(np.array([0]),
                                                self.ddim_timestep_seq[:-1])

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[  # type: ignore
            :, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(
            1 - self.alpha_hat[t])[  # type: ignore
                :, None, None]

        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise

    def forward(self, batch: Dict) -> Dict:
        B = batch['img'].shape[0]

        full_motion = self.get_full_motion(batch)
        cond_embed = self.get_condition(batch)

        # classifier free sampling
        if np.random.random() > self.mod_train:
            cond_embed = None

        # randomly sample timesteps
        ts = torch.randint(0, self.noise_steps, ((B + 1) // 2,))
        if B % 2 == 1:
            ts = torch.cat([ts, self.noise_steps - ts[:-1] - 1], dim=0).long()
        else:
            ts = torch.cat([ts, self.noise_steps - ts - 1], dim=0).long()
        ts = ts.to(self.device)

        # generate Gaussian noise
        noise = torch.randn_like(full_motion)

        # calculate x_t, forward diffusion process
        x_t = self.q_sample(x0=full_motion, t=ts, noise=noise)

        if "motion_mask" in batch:
            x_t[batch["motion_mask"] == 1] = self.mask_embed.unsqueeze(
                0).unsqueeze(0)

        # predict noise
        output = self.diffuser(x_t, ts, cond_embed=cond_embed)

        # calculate loss
        loss_dict = {}
        pred_motion = output
        if "motion_mask" in batch:
            pred_motion[batch["motion_mask"] == 1] = 0
            full_motion[batch["motion_mask"] == 1] = 0

        loss = self.criterion(pred_motion, full_motion)

        loss_dict = {'loss': loss, 'loss_rec': loss.item()}

        local_trans = pred_motion[..., :3]
        gt_local_trans = full_motion[..., :3]

        local_trans_sum = torch.cumsum(local_trans, dim=-2)
        gt_local_trans_sum = torch.cumsum(gt_local_trans, dim=-2)

        loss_traj = self.criterion_traj(local_trans_sum,
                                        gt_local_trans_sum) * 1.0
        loss_dict["loss_traj"] = loss_traj
        loss_dict["loss"] += loss_traj

        # get predicted x_0

        betas = batch["betas"].unsqueeze(1).repeat(1, 60, 1).reshape(-1, 10)
        pred_smpl_output = self.smpl(
            transl=None,
            betas=betas,
            global_orient=None,
            body_pose=rotation_6d_to_matrix(pred_motion[...,
                                                        9:].reshape(-1, 23, 6)),
        )

        pred_joint_locations = vertices2joints(
            self.smpl.J_regressor,  # type: ignore
            pred_smpl_output.vertices)

        gt_smpl_output = self.smpl(
            transl=None,
            betas=betas,
            global_orient=None,
            body_pose=rotation_6d_to_matrix(full_motion[...,
                                                        9:].reshape(-1, 23, 6)),
        )

        gt_joint_locations = vertices2joints(
            self.smpl.J_regressor,  # type: ignore
            gt_smpl_output.vertices)

        loss_geo = self.criterion(pred_joint_locations, gt_joint_locations)

        loss_dict["loss_geo"] = loss_geo.item()
        loss_dict["loss"] += loss_geo
        return loss_dict

    def training_step(self, batch: Dict) -> Dict:
        loss_dict = self(batch)
        for key, val in loss_dict.items():
            self.log("train/" + key,
                     val,
                     prog_bar=True,
                     logger=True,
                     on_step=True,
                     on_epoch=False,
                     batch_size=batch["batch_size"])

        return loss_dict

    def get_condition(self, batch):
        B = batch['img'].shape[0]
        cond_embed = self.cond_embed.unsqueeze(0).repeat(B, 1)

        if self.use_goal:
            goal_pos = batch["global_trans"][:, -1, :] - batch[
                "global_trans"][:, 0, :]
            goal_embed = self.goal_embed(goal_pos)
            cond_embed = cond_embed + goal_embed
        if self.use_beta:
            beta_embed = self.beta_embed(batch["betas"])
            cond_embed = cond_embed + beta_embed

        if self.use_image:
            img = batch['new_img']
            img_feature = img[..., :-2]
            img_pos = img[..., -2:]
            img_pos_embed = positional_encoding_2d(img_pos,
                                                   self.diffuser.latent_dim)
            img_embed = self.img_embed(img_feature)
            img_embed = img_embed + img_pos_embed
            cond_embed = cond_embed.unsqueeze(1)
            cond_embed_res = self.img_cross_attn(
                query=cond_embed,
                key=self.img_cross_attn_norm(img_embed),
                value=self.img_cross_attn_norm(img_embed))
            cond_embed = cond_embed + cond_embed_res[0]
            cond_embed = cond_embed.squeeze(1)

        return cond_embed

    def get_full_motion(self, batch):
        local_trans = batch["global_trans"].clone()

        local_trans[:, 0, :] = 0
        local_trans[:, 1:, :] -= batch["global_trans"][:, :-1, :]

        local_orient = batch["global_orient"]

        full_motion = torch.cat([local_trans, local_orient, batch["body_pose"]],
                                dim=-1)
        return full_motion

    def sample_ddim_progressive(self,
                                full_motion,
                                cond_embed,
                                hand_shake=False):
        sample_num = full_motion.shape[0]

        x = torch.randn_like(full_motion)

        with torch.no_grad():
            for i in reversed(range(0, self.ddim_timesteps)):
                t = (torch.ones(sample_num) *
                     self.ddim_timestep_seq[i]).long().to(self.device)
                prev_t = (torch.ones(sample_num) *
                          self.ddim_timestep_prev_seq[i]).long().to(self.device)

                alpha_hat = self.alpha_hat[t][:, None, None]  # type: ignore
                alpha_hat_prev = self.alpha_hat[prev_t][  # type: ignore
                    :, None, None]

                predicted_x0 = self.diffuser(x, t, cond_embed=cond_embed)
                predicted_x0 = self.inpaint_gt(predicted_x0, full_motion)
                if hand_shake:
                    predicted_x0 = self.hand_shake(predicted_x0)

                predicted_noise = (x - torch.sqrt(
                    (alpha_hat)) * predicted_x0) / torch.sqrt(1 - alpha_hat)

                if i > 0:
                    pred_dir_xt = torch.sqrt(1 -
                                             alpha_hat_prev) * predicted_noise
                    x_prev = torch.sqrt(
                        alpha_hat_prev) * predicted_x0 + pred_dir_xt
                else:
                    x_prev = predicted_x0

                x = x_prev
            return x

    def sample_ddim_progressive_partial(self, xt, x0):
        """
        Generate samples from the model and yield samples from each timestep.

        Args are the same as sample_ddim()
        Returns a generator contains x_{prev_t}, shape as [sample_num, n_pre, 3 * joints_num]
        """
        sample_num = xt.shape[0]
        x = xt

        with torch.no_grad():
            for i in reversed(range(0, 70)):  # hardcoded as add noise t=100
                t = (torch.ones(sample_num) *
                     self.ddim_timestep_seq[i]).long().to(self.device)
                prev_t = (torch.ones(sample_num) *
                          self.ddim_timestep_prev_seq[i]).long().to(self.device)

                alpha_hat = self.alpha_hat[t][:, None, None]  # type: ignore
                alpha_hat_prev = self.alpha_hat[prev_t][  # type: ignore
                    :, None, None]

                predicted_x0 = self.diffuser(x, t, cond_embed=None)
                predicted_x0 = self.inpaint_soft(predicted_x0, x0)

                predicted_noise = (x - torch.sqrt(
                    (alpha_hat)) * predicted_x0) / torch.sqrt(1 - alpha_hat)

                if i > 0:
                    pred_dir_xt = torch.sqrt(1 -
                                             alpha_hat_prev) * predicted_noise
                    x_prev = torch.sqrt(
                        alpha_hat_prev) * predicted_x0 + pred_dir_xt
                else:
                    x_prev = predicted_x0

                x = x_prev

            return x

    def inpaint_soft(self, predicted_x0, x0):
        mask = torch.ones([60]).cuda().float()
        mask[10:20] = torch.linspace(0.80, 0.1, 10).cuda()
        mask[20:30] = 0.1
        mask[30:40] = torch.arange(0.1, 0.8, 10).cuda()
        mask = mask.unsqueeze(0).unsqueeze(-1).repeat(x0.shape[0], 1,
                                                      x0.shape[2])
        predicted_x0 = predicted_x0 * (1. - mask) + x0 * mask

        return predicted_x0

    def inpaint_gt(self, x0, full_motion):
        x0[:, 0, :3] = full_motion[:, 0, :3]
        if self.use_goal:
            gt_goal = torch.sum(full_motion[:, :, :3], dim=1)
            pred_goal = torch.sum(x0[:, :, :3], dim=1)
            goal_scale = gt_goal / (pred_goal + 1e-9)  # B, 3
            x0[:, :, :3] = x0[:, :, :3] * goal_scale.unsqueeze(1)

        return x0

    def hand_shake(self, x0):
        mask = torch.linspace(1.0, 0.0, 10).cuda()
        mask = mask.unsqueeze(0).unsqueeze(-1).repeat(x0.shape[0] - 1, 1,
                                                      x0.shape[2])

        x0_prev = x0[:-1, -10:, :].clone()
        x0_next = x0[1:, :10, :].clone()
        x0[:-1, -10:, :] = x0_prev * mask + (1.0 - mask) * x0_next
        x0[1:, :10, :] = x0_prev * mask + (1.0 - mask) * x0_next

        return x0

    def smooth_motion(self, samples):
        dct, idct = get_dct_matrix(samples.shape[2])
        dct = dct.to(samples.device)
        idct = idct.to(samples.device)
        dct_frames = samples.shape[2] // 6
        dct = dct[:dct_frames, :]
        idct = idct[:, :dct_frames]
        samples = idct @ (dct @ samples)
        return samples

    @torch.no_grad()
    def sample(self,
               full_motion,
               cond_embed,
               num_samples=50,
               hand_shake=False) -> torch.Tensor:
        samples = []
        for _ in range(num_samples):
            samples.append(
                self.sample_ddim_progressive(full_motion,
                                             cond_embed,
                                             hand_shake=hand_shake))
        samples = torch.stack(samples, dim=1)

        return samples

    def eval_step(self, batch: Dict) -> Dict:
        B = batch['img'].shape[0]

        full_motion = self.get_full_motion(batch)
        cond_embed = self.get_condition(batch)

        samples = self.sample(full_motion, cond_embed, self.num_sample)
        samples = self.smooth_motion(samples)
        out_dict = {}
        local_trans = samples[..., :3]

        out_dict["pred_global_orient"] = samples[..., 3:9]

        init_global_trans = batch["global_trans"][:, [0], :].unsqueeze(
            1)  # B, N, 3

        pred_global_trans = torch.cumsum(local_trans, dim=-2)
        pred_global_trans = pred_global_trans + init_global_trans

        out_dict["pred_global_trans"] = pred_global_trans

        out_dict["pred_body_pose"] = samples[..., 9:]

        return out_dict

    def validation_step(self, batch: Dict) -> Dict:
        return self.eval_step(batch)

    def test_step(self, batch: Dict) -> Dict:
        return self.eval_step(batch)

    def predict_step(self, batch: Dict) -> Dict:
        cond_embed = self.get_condition(batch)
        full_motion = self.get_full_motion(batch)

        samples = self.sample(full_motion,
                              cond_embed,
                              self.num_sample,
                              hand_shake=True)

        # re-append noise and denoise, second hand shake

        current_samples = samples[0]

        for i in range(samples.shape[0] - 1):

            x0 = torch.cat(
                [current_samples[:, -30:, :], samples[i + 1, :, 10:40, :]],
                dim=1)

            noise = torch.randn_like(x0)

            # calculate x_t, forward diffusion process
            t = torch.Tensor([700]).cuda().long().repeat(self.num_sample)

            xt = self.q_sample(x0=x0, t=t, noise=noise)

            x0_pred = self.sample_ddim_progressive_partial(xt, x0)

            current_samples = torch.cat([
                current_samples[:, :-30, :], x0_pred, samples[i + 1, :, 41:, :]
            ],
                                        dim=1)

        samples = current_samples.unsqueeze(0)
        samples = self.smooth_motion(samples)
        out_dict = {}
        local_trans = samples[..., :3]

        out_dict["pred_global_orient"] = samples[..., 3:9]

        init_global_trans = batch["global_trans"][[0], [0], :].unsqueeze(
            1)  # 1, N, 3 NOTE: pred only support bs==1!
        pred_global_trans = torch.cumsum(local_trans, dim=-2)
        pred_global_trans = pred_global_trans + init_global_trans

        out_dict["pred_global_trans"] = pred_global_trans

        out_dict["pred_body_pose"] = samples[..., 9:]

        return out_dict

    def configure_optimizers(self):
        lr = self.optimizer_conf["basic_lr_per_img"] * \
            self.batch_size_per_device * self.gpus

        # Create a list of parameter groups with different learning rates
        param_groups = []
        param_group_1 = {'params': [], 'lr': lr * 0.1}
        param_group_2 = {'params': [], 'lr': lr}
        for name, param in self.named_parameters():
            if "backbone" in name:
                param_group_1['params'].append(param)
            else:
                param_group_2['params'].append(param)
        param_groups.append(param_group_1)
        param_groups.append(param_group_2)

        optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=1e-7)

        scheduler = MultiStepLR(optimizer,
                                milestones=self.lr_scheduler_conf["milestones"],
                                gamma=self.lr_scheduler_conf["gamma"])
        return [[optimizer], [scheduler]]
