"""Our custom pedestrian dataset."""
import copy
import os
import pickle
from functools import reduce
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pedgen.utils.colors import IMG_MEAN, IMG_STD
from pedgen.utils.rot import (axis_angle_to_matrix, create_2d_grid,
                              create_occupancy_grid, depth_to_3d,
                              matrix_to_rotation_6d)


class CityWalkersDataset(Dataset):
    """Lightning dataset for pedestrian generation."""

    def __init__(
        self,
        label_file: str,
        mode: str,
        data_root: str,
        img_root: str,
        img_dim: List,
        min_timestamp: int,
        use_partial: bool,
        num_timestamp: int,
        depth_root: str,
        semantic_root: str,
        sample_interval: int,
        sample_start_idx: int,
        grid_size: List,
        grid_points: List,
        use_image: bool,
        use_data_augmentation: bool,
        train_percent: float,
    ) -> None:
        with open(os.path.join(data_root, label_file), "rb") as f:
            labels = pickle.load(f)
        self.label_list = []
        self.img_mean = np.array(IMG_MEAN)
        self.img_std = np.array(IMG_STD)
        self.img_w = img_dim[1]
        self.img_h = img_dim[0]
        self.mode = mode
        self.num_timestamp = num_timestamp
        self.min_timestamp = min_timestamp  # hardcoded
        self.use_partial = use_partial
        self.grid_size = grid_size
        self.use_image = use_image
        self.grid_points = grid_points
        self.use_data_augmentation = use_data_augmentation

        for idx, val in enumerate(labels):
            if np.isnan(val["global_trans"]).any() or np.isnan(
                    val["local_trans"]).any():
                continue
            image_path = os.path.join(data_root, img_root, val["image"])
            if self.mode != "pred":
                i = sample_start_idx
                max_i = val["global_trans"].shape[
                    0] - self.min_timestamp + 1 if self.use_partial and self.mode == "train" else val[
                        "global_trans"].shape[0] - self.num_timestamp + 1
                while i < max_i:
                    img_id = int(image_path.split("/")[-1].split(".")[0]) + i
                    new_val = copy.deepcopy(val)
                    new_val["start_t"] = i
                    new_image_path = image_path[:-10] + str(img_id).zfill(
                        6) + ".jpg"
                    new_val["image"] = new_image_path
                    if not os.path.exists(new_val["image"]):
                        break
                    depth_path = new_image_path.replace(img_root, depth_root)
                    depth_path = depth_path.replace("jpg", "png")
                    new_val["depth"] = depth_path

                    semantic_path = new_image_path.replace(
                        img_root, semantic_root)
                    semantic_path = semantic_path.replace("jpg", "png")
                    new_val["semantic"] = semantic_path

                    voxel_path = new_image_path.replace(img_root, "voxel")
                    voxel_path = voxel_path.replace("jpg", "npy")
                    new_val["voxel"] = voxel_path
                    new_val["index"] = idx

                    img_id = new_val["image"].split("/")
                    img_id = img_id[-2] + "_" + img_id[-1][:-4] + "_" + str(idx)
                    img_id = img_id[-2] + "_" + \
                    img_id[-1][:-4] + "_" + str(idx)
                    self.label_list.append(new_val)
                    i += sample_interval
            else:
                val["start_t"] = 0
                val["image"] = image_path
                if not os.path.exists(val["image"]):
                    continue
                depth_path = image_path.replace(img_root, depth_root)
                depth_path = depth_path.replace("jpg", "png")
                val["depth"] = depth_path
                semantic_path = image_path.replace(img_root, semantic_root)
                semantic_path = semantic_path.replace("jpg", "png")
                val["semantic"] = semantic_path
                voxel_path = image_path.replace(img_root, "voxel")
                voxel_path = voxel_path.replace("jpg", "npy")
                val["voxel"] = voxel_path
                # ground_path = image_path.replace(img_root, "ground")
                # ground_path = ground_path.replace("jpg", "npy")
                # val["ground"] = ground_path
                self.label_list.append(val)

        if self.mode == "train" and train_percent < 1.0:
            self.label_list = self.label_list[:int(
                len(self.label_list) * train_percent)]

    def __len__(self) -> int:
        return len(self.label_list)

    def __getitem__(self, index: int) -> Dict:
        label = self.label_list[index]
        tt = lambda x: torch.from_numpy(x).float()
        data_dict = {}
        img_id = label["image"].split("/")
        img_id = img_id[-2] + "_" + img_id[-1][:-4] + "_" + str(label["index"])
        meta = {"source": "pedmotion", "img_id": img_id}
        data_dict["meta"] = meta

        if self.mode == "train":
            img = torch.zeros([3, 720, 1280], dtype=torch.float32)
        else:
            rgb = cv2.imread(label["image"])
            rgb = np.array(rgb, dtype=np.float32)
            rgb_vis = copy.deepcopy(rgb)
            mean = np.float64(self.img_mean.reshape(1, -1))
            stdinv = 1 / np.float64(self.img_std.reshape(1, -1))
            cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB, rgb)  # type: ignore
            cv2.subtract(rgb, mean, rgb)  # type: ignore
            cv2.multiply(rgb, stdinv, rgb)  # type: ignore
            img = tt(rgb).permute(2, 0, 1)  # 3, H ,W

        f = (1280**2 + 720**2)**0.5
        cx = 0.5 * 1280
        cy = 0.5 * 720
        intrinsics_old = np.eye(3)
        intrinsics_old[0, 0] = f
        intrinsics_old[1, 1] = f
        intrinsics_old[0, 2] = cx
        intrinsics_old[1, 2] = cy

        start_t = label["start_t"]

        global_orient_source = axis_angle_to_matrix(
            tt(label['global_orient'][start_t]))
        global_orient_target = axis_angle_to_matrix(
            tt(label['local_orient'][start_t]))

        transl_source = tt(label['global_trans'][[start_t]])
        transl_target = tt(label['local_trans'][[start_t]])

        source_to_target_rotation = global_orient_target @ global_orient_source.T

        global_orient = axis_angle_to_matrix(
            tt(label['global_orient'][start_t:, :3]))
        global_orient = source_to_target_rotation @ global_orient

        source_to_target_translation = transl_target.T - \
            source_to_target_rotation @ transl_source.T
        transl = tt(label['global_trans'][start_t:])

        transl = source_to_target_rotation @ transl.T + source_to_target_translation
        transl = transl.T

        data_dict["img"] = img
        data_dict["intrinsics"] = tt(intrinsics_old)
        data_dict["global_trans"] = transl[:self.num_timestamp]
        data_dict["motion_mask"] = torch.zeros((self.num_timestamp,),
                                               dtype=torch.bool)

        data_dict["global_orient"] = matrix_to_rotation_6d(
            global_orient[:self.num_timestamp])
        data_dict["betas"] = torch.mean(tt(label["betas"][start_t:start_t +
                                                          self.num_timestamp]),
                                        dim=0)  # use the average beta

        data_dict["body_pose"] = matrix_to_rotation_6d(
            axis_angle_to_matrix(
                tt(label["body_pose"][start_t:start_t +
                                      self.num_timestamp]))).reshape(
                                          -1, 23 * 6)

        if data_dict["global_trans"].shape[0] < self.num_timestamp:
            motion_length = data_dict["global_trans"].shape[0]
            data_dict["motion_mask"][motion_length:] = True
            data_dict["global_trans"] = torch.cat([
                data_dict["global_trans"],
                torch.zeros(self.num_timestamp - motion_length, 3)
            ],
                                                  dim=0)
            data_dict["global_orient"] = torch.cat([
                data_dict["global_orient"],
                torch.zeros(self.num_timestamp - motion_length, 6)
            ],
                                                   dim=0)
            data_dict["body_pose"] = torch.cat([
                data_dict["body_pose"],
                torch.zeros(self.num_timestamp - motion_length, 23 * 6)
            ],
                                               dim=0)

        if not self.use_image:
            return data_dict

        if not self.mode == "pred":
            # convert x, y, z in global to x, y in pixel and relative d
            try:
                depth_3d = np.load(label["voxel"])
                depth_3d = tt(depth_3d)
            except:
                depth = cv2.imread(label["depth"], -1)
                depth = np.array(depth, dtype=np.float32)
                depth = depth / 256.  # original is unit16 format, divide 256 to convert to metric depth
                depth_3d = depth_to_3d(depth, intrinsics_old)
                depth_3d = tt(depth_3d)  # h,w,c

                semantic_raw = cv2.imread(label["semantic"], -1)
                semantic_raw = np.array(semantic_raw, dtype=np.float32)
                semantic_raw = semantic_raw / 18.  # normalize
                semantic_raw = tt(semantic_raw).unsqueeze(-1)
                depth_3d = torch.cat([depth_3d, semantic_raw], dim=-1)  # h,w,c

                voxel_dir = label["voxel"].rsplit("/", 1)[0]
                os.makedirs(voxel_dir, exist_ok=True)
                np.save(label["voxel"], depth_3d.cpu().numpy())

            init_trans = transl[0]

            dx = int(init_trans[0] * (1280**2 + 720**2)**0.5 / init_trans[2] +
                     1280 / 2)
            dy = int(init_trans[1] * (1280**2 + 720**2)**0.5 / init_trans[2] +
                     720 / 2)

            dx = max(min(dx, 1279), 0)
            dy = max(min(dy, 719), 0)

            metric_depth = depth_3d[dy, dx, 2]

            depth_3d[..., :3] = depth_3d[..., :3] * init_trans[2] / metric_depth
            depth_3d[..., :3] = depth_3d[..., :3] - init_trans.unsqueeze(
                0).unsqueeze(0)

            # add data augmentation

            if self.mode == "train" and self.use_data_augmentation:
                transl = transl - transl[[0]]
                rot = np.random.uniform(-np.pi * 180 / 180, np.pi * 180 / 180)
                rot_mat = torch.tensor(
                    [[np.cos(rot), 0, np.sin(rot)], [0, 1, 0],
                     [-np.sin(rot), 0, np.cos(rot)]],
                    dtype=global_orient.dtype)
                global_orient = rot_mat @ global_orient
                transl = transl @ rot_mat.T
                depth_3d[..., :3] = depth_3d[..., :3] @ rot_mat.T

            mask = torch.ones([720, 1280], dtype=torch.bool)
            bbox = label["bbox_2d"][start_t]
            mask[int(bbox[2]):int(bbox[3]), int(bbox[0]):int(bbox[1])] = 0

            mask = reduce(torch.logical_and, [
                mask, depth_3d[..., 0] >= self.grid_size[0] + 1e-5,
                depth_3d[..., 0] < self.grid_size[1] - 1e-5, depth_3d[..., 1]
                >= self.grid_size[2] + 1e-5, depth_3d[..., 1]
                < self.grid_size[3] - 1e-5, depth_3d[..., 2]
                >= self.grid_size[4] + 1e-5, depth_3d[..., 2]
                < self.grid_size[5] - 1e-5
            ])
            depth_3d = depth_3d.reshape(720 * 1280, -1)
            mask = mask.flatten()
            depth_3d = depth_3d[mask, :]

            occupancy_grid = create_occupancy_grid(
                depth_3d,
                self.grid_size,
                self.grid_points,
            )
            grid_2d = tt(
                create_2d_grid(num_points=self.grid_points,
                               grid_size=self.grid_size))

            occupancy_grid = occupancy_grid.permute(0, 2, 1)
            occupancy_grid = torch.cat([occupancy_grid, grid_2d], dim=-1)
            occupancy_grid = occupancy_grid.reshape(
                occupancy_grid.shape[0] * occupancy_grid.shape[1], -1)

            data_dict["new_img"] = occupancy_grid

        else:
            pred_goal = torch.Tensor([[0., 0., -2.], [2., 0., 0.]]).repeat(5, 1)
            num_pred_steps = pred_goal.shape[0]
            init_trans = transl[0]
            current_init_pos = transl[0]

            data_dict["img"] = data_dict["img"].unsqueeze(0).repeat(
                num_pred_steps, 1, 1, 1)
            data_dict["global_trans"] = data_dict["global_trans"].unsqueeze(
                0).repeat(num_pred_steps, 1, 1)
            data_dict["global_orient"] = data_dict["global_orient"].unsqueeze(
                0).repeat(num_pred_steps, 1, 1)
            data_dict["betas"] = data_dict["betas"].unsqueeze(0).repeat(
                num_pred_steps, 1)
            data_dict["body_pose"] = data_dict["body_pose"].unsqueeze(0).repeat(
                num_pred_steps, 1, 1)
            data_dict["meta"] = [data_dict["meta"]]
            data_dict["batch_size"] = 1

            data_dict["new_img"] = []
            for i in range(pred_goal.shape[0]):

                depth = cv2.imread(label["depth"], -1)
                depth = np.array(depth, dtype=np.float32)
                depth = depth / 256.  # original is unit16 format, divide 256 to convert to metric depth
                depth_3d = depth_to_3d(depth, intrinsics_old)
                depth_3d = tt(depth_3d)  # h,w,c

                semantic_raw = cv2.imread(label["semantic"], -1)
                semantic_raw = np.array(semantic_raw, dtype=np.float32)
                semantic_raw = semantic_raw / 18.  # normalize
                semantic_raw = tt(semantic_raw).unsqueeze(-1)
                depth_3d = torch.cat([depth_3d, semantic_raw], dim=-1)  # h,w,c

                dx = int(current_init_pos[0] *
                         (1280**2 + 720**2)**0.5 / current_init_pos[2] +
                         1280 / 2)
                dy = int(current_init_pos[1] *
                         (1280**2 + 720**2)**0.5 / current_init_pos[2] +
                         720 / 2)

                dx = max(min(dx, 1279), 0)
                dy = max(min(dy, 719), 0)

                metric_depth = depth[dy, dx]

                depth_3d[
                    ..., :3] = depth_3d[..., :3] * init_trans[2] / metric_depth
                depth_3d[..., :3] = depth_3d[
                    ..., :3] - current_init_pos.unsqueeze(0).unsqueeze(0)

                mask = torch.ones([720, 1280], dtype=torch.bool)
                bbox = label["bbox_2d"][start_t]
                mask[int(bbox[2]):int(bbox[3]), int(bbox[0]):int(bbox[1])] = 0

                mask = reduce(torch.logical_and, [
                    mask, depth_3d[..., 0] >= self.grid_size[0] + 1e-5,
                    depth_3d[..., 0] < self.grid_size[1] - 1e-5,
                    depth_3d[..., 1] >= self.grid_size[2] + 1e-5,
                    depth_3d[..., 1] < self.grid_size[3] - 1e-5,
                    depth_3d[..., 2] >= self.grid_size[4] + 1e-5,
                    depth_3d[..., 2] < self.grid_size[5] - 1e-5
                ])
                depth_3d = depth_3d.reshape(720 * 1280, -1)
                mask = mask.flatten()
                depth_3d = depth_3d[mask, :]

                occupancy_grid = create_occupancy_grid(
                    depth_3d,
                    self.grid_size,
                    self.grid_points,
                )

                grid_2d = tt(
                    create_2d_grid(num_points=self.grid_points,
                                   grid_size=self.grid_size))

                occupancy_grid = occupancy_grid.permute(0, 2, 1)
                occupancy_grid = torch.cat([occupancy_grid, grid_2d], dim=-1)

                occupancy_grid = occupancy_grid.reshape(
                    occupancy_grid.shape[0] * occupancy_grid.shape[1], -1)

                data_dict["new_img"].append(occupancy_grid)

                data_dict["global_trans"][
                    i, -1] = data_dict["global_trans"][i, 0] + pred_goal[i]
                current_init_pos += pred_goal[i]
            data_dict["new_img"] = torch.stack(data_dict["new_img"], dim=0)

        return data_dict


def collate_fn_pedmotion(data):
    """Custom collate function."""
    img_batch = []
    new_img_batch = []
    intrinsics_batch = []
    global_trans_batch = []
    betas_batch = []
    global_orient_batch = []
    body_pose_batch = []
    meta_batch = []
    motion_mask_batch = []

    for data_dict in data:
        img_batch.append(data_dict["img"])
        intrinsics_batch.append(data_dict["intrinsics"])
        global_trans_batch.append(data_dict["global_trans"])
        betas_batch.append(data_dict["betas"])
        global_orient_batch.append(data_dict["global_orient"])
        body_pose_batch.append(data_dict["body_pose"])
        meta_batch.append(data_dict["meta"])
        if "motion_mask" in data_dict:
            motion_mask_batch.append(data_dict["motion_mask"])
        if "new_img" in data_dict:
            new_img_batch.append(data_dict["new_img"])

    ret_dict = {
        "img": torch.stack(img_batch),
        "intrinsics": torch.stack(intrinsics_batch),
        "meta": meta_batch,
        "global_trans": torch.stack(global_trans_batch),
        "betas": torch.stack(betas_batch),
        "global_orient": torch.stack(global_orient_batch),
        "body_pose": torch.stack(body_pose_batch),
        "batch_size": len(img_batch),
    }

    if len(motion_mask_batch) > 0:
        ret_dict["motion_mask"] = torch.stack(motion_mask_batch)

    if len(new_img_batch) > 0:
        ret_dict["new_img"] = torch.stack(new_img_batch)

    return ret_dict


def collate_fn_pedmotion_pred(data):
    """Custom collate function."""
    return data[0]
