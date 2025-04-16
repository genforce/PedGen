"""Visualization Callback for PedGen."""
import os
import pickle
from typing import Dict

import cv2
import imageio
import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from lightning_utilities.core.rank_zero import rank_zero_only
from smplx import SMPLLayer
from smplx.lbs import vertices2joints

from pedgen.utils.colors import IMG_MEAN, IMG_STD, get_colors
from pedgen.utils.eval import (AverageMeter, compute_joint_metrics,
                               compute_pose_metrics, compute_traj_metrics)
from pedgen.utils.renderer import Renderer
from pedgen.utils.rot import rotation_6d_to_matrix


class PedGenVisCallback(Callback):
    """Visualization Callback for PedGen."""

    def __init__(self, vis_num: Dict, store_result: Dict, vis_image: bool,
                 vis_video: bool) -> None:
        """Init."""
        self.vis_num = vis_num
        self.vis_image = vis_image
        self.vis_video = vis_video
        self.store_result = store_result
        self.smpl = SMPLLayer(model_path="smpl", gender='neutral')
        if self.vis_image or self.vis_video:
            self.smpl = SMPLLayer(model_path="smpl", gender='neutral')
            texture_file = np.load("smpl/texture.npz")
            self.faces_cpu = texture_file['smpl_faces'].astype('uint32')
            self.colors = get_colors()

        super(PedGenVisCallback).__init__()

    def on_epoch_start(self, trainer: Trainer, mode):
        self.mode = mode
        self.vis_count = 0
        vis_root = trainer.strategy.broadcast(trainer.default_root_dir, 0)
        self.vis_root = os.path.join(vis_root, f"vis_{mode}",
                                     f"epoch_{trainer.current_epoch}")

        if mode in ["val", "test"]:
            self.stats_names = [
                'APD', 'ADE_m', 'FDE_m', 'ADE_a', 'FDE_a', 'APD_traj',
                'ADE_traj_m', 'FDE_traj_m', 'ADE_traj_a', 'FDE_traj_a',
                "AOE_init_m", "ADE_init_m", "AOE_init_a", "ADE_init_a",
                "APD_init", "mpjpe"
            ]
            self.stats_meter = {x: AverageMeter() for x in self.stats_names}

        os.makedirs(self.vis_root, exist_ok=True)
        os.makedirs(os.path.join(self.vis_root, "smpl"), exist_ok=True)
        if self.store_result[self.mode]:
            self.result_list = []

    @rank_zero_only
    def on_eval_epoch_end(self, pl_module: LightningModule, mode):
        if self.store_result[self.mode]:
            with open(os.path.join(self.vis_root, "result.pkl"), "wb") as f:
                pickle.dump(self.result_list, f)

        if mode in ["val", "test"]:
            for stats in self.stats_meter:
                pl_module.log(f"{self.mode}/{stats}",
                              self.stats_meter[stats].avg,
                              rank_zero_only=True)

    def on_validation_epoch_start(self, trainer: Trainer,
                                  pl_module: LightningModule) -> None:
        self.on_epoch_start(trainer, "val")

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule) -> None:
        self.on_eval_epoch_end(pl_module, "val")

    def on_test_epoch_start(self, trainer: Trainer,
                            pl_module: LightningModule) -> None:
        self.on_epoch_start(trainer, "test")

    def on_test_epoch_end(self, trainer: Trainer,
                          pl_module: LightningModule) -> None:
        self.on_eval_epoch_end(pl_module, "test")

    def on_predict_epoch_start(self, trainer: Trainer,
                               pl_module: LightningModule) -> None:
        self.on_epoch_start(trainer, "pred")

    def on_predict_epoch_end(self, trainer: Trainer,
                             pl_module: LightningModule) -> None:
        self.on_eval_epoch_end(pl_module, "pred")

    def eval_main(self, trainer: Trainer, pl_module: LightningModule,
                  outputs: Dict, batch: Dict):
        # compute metrics
        B, N, T, _ = outputs["pred_global_trans"].shape

        for i in range(B):
            body_pose = rotation_6d_to_matrix(
                outputs["pred_body_pose"][i].reshape(-1, 23, 6))

            pred_transl = outputs["pred_global_trans"][i]
            pred_rot = rotation_6d_to_matrix(outputs["pred_global_orient"][i])

            pred_smpl_output = self.smpl(
                transl=pred_transl.reshape(-1, 3),
                betas=batch["betas"][i].unsqueeze(0).unsqueeze(0).repeat(
                    N, T, 1).reshape(-1, 10),
                global_orient=pred_rot.reshape(-1, 3, 3),
                body_pose=body_pose,
            )

            pred_joint_locations = vertices2joints(
                self.smpl.J_regressor,  # type: ignore
                pred_smpl_output.vertices)

            pred_joint_locations = pred_joint_locations.reshape(N, T, -1)

            if self.store_result[self.mode]:
                for j in range(outputs["pred_global_trans"][i].shape[0]):
                    result_dict = {}
                    result_dict["image"] = batch["meta"][i]["img_id"]
                    if "map_info" in batch["meta"][i]:
                        result_dict["map_info"] = batch["meta"][i]["map_info"]
                    if "sensor_pose" in batch["meta"][i]:
                        result_dict["sensor_pose"] = batch["meta"][i][
                            "sensor_pose"]
                    result_dict["pred_id"] = j
                    result_dict["global_trans"] = outputs["pred_global_trans"][
                        i][j].cpu().numpy()
                    result_dict["global_orient"] = rotation_6d_to_matrix(
                        outputs["pred_global_orient"][i][j]).cpu().numpy()
                    result_dict["betas"] = batch["betas"][i].cpu().numpy()
                    result_dict["body_pose"] = rotation_6d_to_matrix(
                        outputs["pred_body_pose"][i][j].reshape(
                            -1, 23, 6)).cpu().numpy()
                    self.result_list.append(result_dict)

            if self.mode == "val" or self.mode == "test":

                body_pose = rotation_6d_to_matrix(batch["body_pose"][i].reshape(
                    -1, 23, 6))

                gt_transl = batch["global_trans"][i]
                gt_rot = rotation_6d_to_matrix(batch["global_orient"][i])

                gt_smpl_output = self.smpl(
                    transl=gt_transl.reshape(-1, 3),
                    betas=batch["betas"][i].unsqueeze(0).repeat(T, 1).reshape(
                        -1, 10),
                    global_orient=gt_rot.reshape(-1, 3, 3),
                    body_pose=body_pose,
                )

                gt_joint_locations = vertices2joints(
                    self.smpl.J_regressor,  # type: ignore
                    gt_smpl_output.vertices)

                gt_joint_locations = gt_joint_locations.reshape(1, T, -1)
                gt_transl = gt_transl.unsqueeze(0)
                gt_rot = gt_rot.unsqueeze(0)

                # dealing with nans
                contains_nan = torch.isnan(pred_joint_locations).any(dim=2).any(
                    dim=1)
                pred_joint_locations = pred_joint_locations[~contains_nan]
                pred_rot = pred_rot[~contains_nan]
                pred_transl = pred_transl[~contains_nan]

                # dealing with motion masks
                if 'motion_mask' in batch:
                    pred_joint_locations = pred_joint_locations[:, ~batch[
                        'motion_mask'][i]]
                    pred_rot = pred_rot[:, ~batch['motion_mask'][i]]
                    pred_transl = pred_transl[:, ~batch['motion_mask'][i]]
                    gt_joint_locations = gt_joint_locations[:, ~batch[
                        'motion_mask'][i]]
                    gt_rot = gt_rot[:, ~batch['motion_mask'][i]]
                    gt_transl = gt_transl[:, ~batch['motion_mask'][i]]

                if batch["meta"][i]["source"] != "waymo":
                    apd, aade, made, afde, mfde, apd_init, aade_init, made_init = compute_pose_metrics(
                        pred_joint_locations, gt_joint_locations)

                    apds = pl_module.all_gather(apd)
                    aades = pl_module.all_gather(aade)
                    mades = pl_module.all_gather(made)
                    afdes = pl_module.all_gather(afde)
                    mfdes = pl_module.all_gather(mfde)
                    apd_inits = pl_module.all_gather(apd_init)
                    aade_inits = pl_module.all_gather(aade_init)
                    made_inits = pl_module.all_gather(made_init)

                    self.stats_meter["ADE_a"].update(aades)
                    self.stats_meter["ADE_m"].update(mades)
                    self.stats_meter["FDE_a"].update(afdes)
                    self.stats_meter["FDE_m"].update(mfdes)
                    self.stats_meter["APD_init"].update(apd_inits)
                    self.stats_meter["ADE_init_a"].update(aade_inits)
                    self.stats_meter["ADE_init_m"].update(made_inits)

                    apd_traj, aade_traj, made_traj, afde_traj, mfde_traj, aaoe_init, maoe_init = compute_traj_metrics(
                        pred_transl, gt_transl, pred_rot, gt_rot)
                    apd_trajs = pl_module.all_gather(apd_traj)
                    aade_trajs = pl_module.all_gather(aade_traj)
                    made_trajs = pl_module.all_gather(made_traj)
                    afde_trajs = pl_module.all_gather(afde_traj)
                    mfde_trajs = pl_module.all_gather(mfde_traj)
                    aaoe_inits = pl_module.all_gather(aaoe_init)
                    maoe_inits = pl_module.all_gather(maoe_init)

                    self.stats_meter["APD_traj"].update(apd_trajs)
                    self.stats_meter["ADE_traj_a"].update(aade_trajs)
                    self.stats_meter["ADE_traj_m"].update(made_trajs)
                    self.stats_meter["FDE_traj_a"].update(afde_trajs)
                    self.stats_meter["FDE_traj_m"].update(mfde_trajs)
                    self.stats_meter["AOE_init_a"].update(aaoe_inits)
                    self.stats_meter["AOE_init_m"].update(maoe_inits)
                else:
                    aade, made, afde, mfde = compute_joint_metrics(
                        pred_joint_locations, {
                            "frame_ids": batch["frame_ids"][i],
                            "keypoints": batch["keypoints"][i]
                        })
                    aades = pl_module.all_gather(aade)
                    mades = pl_module.all_gather(made)
                    afdes = pl_module.all_gather(afde)
                    mfdes = pl_module.all_gather(mfde)

                    self.stats_meter["ADE_a"].update(aades)
                    self.stats_meter["ADE_m"].update(mades)
                    self.stats_meter["FDE_a"].update(afdes)
                    self.stats_meter["FDE_m"].update(mfdes)

            # visualization
            if self.vis_count < self.vis_num[self.mode]:
                intrisics = batch["intrinsics"][0].cpu().numpy()
                render = Renderer(
                    focal_length=[intrisics[0, 0], intrisics[1, 1]],
                    camera_center=[intrisics[0, 2], intrisics[1, 2]],
                    img_res=[1280, 720],
                    faces=self.faces_cpu,
                    metallicFactor=0.0,
                    roughnessFactor=0.7)

                img = batch["img"][i, :3, :, :].cpu().permute(
                    1, 2, 0).numpy()  # B,H,W,C
                # unnormalize
                mean = np.array(IMG_MEAN)[None, None, :]
                std = np.array(IMG_STD)[None, None, :]
                img = img * std + mean

                img = cv2.resize(
                    img,
                    (1280, 720),
                    interpolation=cv2.INTER_LINEAR,
                )
                img = img.astype(np.uint8)

                img_id = batch["meta"][i]["img_id"]

                pred_vertices = pred_smpl_output.vertices
                pred_vertices = pred_vertices.reshape(N, T, -1, 3)
                for j in range(pred_vertices.shape[0]):
                    # only vis the first sample
                    pred_vertice = pred_vertices[j]
                    if self.mode == "val":
                        gt_vertices = gt_smpl_output.vertices
                    else:
                        gt_vertices = pred_smpl_output.vertices
                    if self.vis_image or self.vis_video:
                        self.vis_smpl_impl(render, img, img_id, gt_vertices,
                                           pred_vertice, j)
                    self.vis_count += 1
                    if self.vis_count >= self.vis_num[self.mode]:
                        break

    def vis_smpl_impl(self, render, img, img_id, gt_vertices, pred_vertices,
                      id):
        """Main function for visualization."""
        # vis gt
        if self.mode == "val":
            img_smpl, valid_mask = render.visualize_all(
                gt_vertices[[0, 10, 20, 30, 40, 50, 59]].cpu().numpy(),
                self.colors[[7, 6, 5, 4, 3, 2, 1]],
            )

            if self.vis_image:
                img_vis = img.copy()
                img_vis = img_smpl[:, :, :3] * valid_mask + (
                    1 - valid_mask) * img_vis / 255.
                img_vis = img_vis * 255
                img_vis = img_vis.astype(np.uint8)
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(self.vis_root, "smpl", img_id + "_gt.jpg"),
                    img_vis)
            if self.vis_video:
                writer = imageio.get_writer(
                    os.path.join(self.vis_root, "smpl", img_id + "_gt.mp4"),
                    fps=30,
                    mode='I',
                    format='FFMPEG',  # type: ignore
                    macro_block_size=1)

                for t in range(gt_vertices.shape[0]):
                    img_gt = img.copy()
                    img_smpl = img_gt.copy()
                    img_smpl, valid_mask = render.visualize_all(
                        gt_vertices[[t]].cpu().numpy(),
                        self.colors[[0]],
                    )

                    img_gt = img_smpl[:, :, :3] * valid_mask + (
                        1 - valid_mask) * img_gt / 255.
                    img_gt = img_gt * 255
                    img_gt = img_gt.astype(np.uint8)
                    writer.append_data(img_gt)
                writer.close()

        # vis prediction
        if self.vis_image:
            img_vis = img.copy()
            img_smpl, valid_mask = render.visualize_all(
                pred_vertices[[0, 10, 20, 30, 40, 50, 59]].cpu().numpy(),
                self.colors[[7, 6, 5, 4, 3, 2, 1]],
            )

            img_vis = img_smpl[:, :, :3] * valid_mask + (
                1 - valid_mask) * img_vis / 255.
            img_vis = img_vis * 255
            img_vis = img_vis.astype(np.uint8)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(self.vis_root, "smpl",
                             img_id + "_" + str(id) + "_pred.jpg"), img_vis)
        if self.vis_video:
            writer = imageio.get_writer(
                os.path.join(self.vis_root, "smpl",
                             img_id + "_" + str(id) + "_pred.mp4"),
                fps=30,
                mode='I',
                format='FFMPEG',  # type: ignore
                macro_block_size=1)

            for t in range(pred_vertices.shape[0]):
                img_pred = img.copy()
                img_smpl = img_pred.copy()
                img_smpl, valid_mask = render.visualize_all(
                    pred_vertices[[t]].cpu().numpy(),
                    self.colors[[1]],
                )

                img_pred = img_smpl[:, :, :3] * valid_mask + (
                    1 - valid_mask) * img_pred / 255.
                img_pred = img_pred * 255
                img_pred = img_pred.astype(np.uint8)
                writer.append_data(img_pred)
            writer.close()

    def on_validation_batch_end(self, trainer: Trainer,
                                pl_module: LightningModule, outputs: Dict,
                                batch: Dict, batch_idx: int) -> None:
        self.smpl.to(batch["img"].device)
        self.eval_main(trainer, pl_module, outputs, batch)

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs: Dict, batch: Dict, batch_idx: int) -> None:
        self.smpl.to(batch["img"].device)
        self.eval_main(trainer, pl_module, outputs, batch)

    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                             outputs: Dict, batch: Dict,
                             batch_idx: int) -> None:
        self.smpl.to(batch["img"].device)
        self.eval_main(trainer, pl_module, outputs, batch)
