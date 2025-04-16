"""Our custom pedestrian dataset."""
import copy
import os
import pickle
from functools import reduce
from glob import glob
from typing import Dict, List

import cv2
import numpy as np
import torch
from smplx import SMPL
from torch.utils.data import Dataset

from pedgen.dataset.sloper4d_utils import SLOPER4D_Dataset
from pedgen.utils.colors import IMG_MEAN, IMG_STD
from pedgen.utils.rot import (axis_angle_to_matrix, create_2d_grid,
                              create_occupancy_grid, depth_to_3d,
                              matrix_to_rotation_6d)


class SLOPER4D(Dataset):
    """Lightning dataset for pedestrian generation."""

    def __init__(
        self,
        mode: str,
        data_root: str,
        img_dim: List,
        num_timestamp: int,
        grid_size: List,
        grid_points: List,
        use_image: bool,
        use_data_augmentation: bool,
        sample_interval: int,
        **kwargs,
    ) -> None:
        self.label_list = []
        self.img_mean = np.array(IMG_MEAN)
        self.img_std = np.array(IMG_STD)
        self.img_w = img_dim[1]
        self.img_h = img_dim[0]
        self.mode = mode
        self.num_timestamp = num_timestamp
        self.grid_size = grid_size
        self.use_image = use_image
        self.grid_points = grid_points
        self.use_data_augmentation = use_data_augmentation

        data_root = os.path.join(data_root, "SLOPER4D")

        self.smpl = SMPL(model_path="smpl")

        for seq in sorted(os.listdir(data_root)):
            os.makedirs(os.path.join(data_root, seq, "voxel"), exist_ok=True)
            pkl_file = glob(os.path.join(data_root, seq, "*_labels.pkl"))[0]
            seq_dataset = SLOPER4D_Dataset(pkl_file)
            bboxes = seq_dataset.bbox
            valid = np.array([len(box) == 4 for box in bboxes])
            image_list = [
                os.path.join(data_root, seq, "images", s)
                for s in seq_dataset.file_basename
            ]

            start_t = 0

            while start_t <= len(image_list) - self.num_timestamp:
                label_dict = {}
                label_dict["image"] = image_list[start_t]
                depth_path = image_list[start_t].replace("images", "depth")
                depth_path = depth_path.replace("jpg", "png")
                label_dict["depth"] = depth_path

                semantic_path = image_list[start_t].replace(
                    "images", "semantics")
                semantic_path = semantic_path.replace("jpg", "png")
                label_dict["semantic"] = semantic_path

                if not os.path.exists(
                        label_dict["depth"]) or not os.path.exists(
                            label_dict["semantic"]):
                    break

                voxel_path = image_list[start_t].replace("images", "voxel")
                voxel_path = voxel_path.replace("jpg", "npy")
                label_dict["voxel"] = voxel_path

                label_dict["intrinsics"] = seq_dataset.cam["intrinsics"]
                label_dict["bbox_2d"] = seq_dataset.bbox[start_t]
                label_dict["cam_pose"] = seq_dataset.cam_pose[start_t]
                label_dict["global_trans"] = seq_dataset.global_trans[
                    start_t:start_t + self.num_timestamp]
                label_dict["global_orient"] = seq_dataset.smpl_pose[
                    start_t:start_t + self.num_timestamp, :3]
                label_dict["body_pose"] = seq_dataset.smpl_pose[
                    start_t:start_t + self.num_timestamp,
                    3:].reshape(-1, 23, 3)
                label_dict["betas"] = np.array(seq_dataset.betas)

                self.label_list.append(label_dict)

                start_t += sample_interval
                while start_t <= len(
                        image_list) - self.num_timestamp and not valid[start_t]:
                    start_t += sample_interval  # 20fps video
        print(f"total length of sloper 4d: {len(self.label_list)}")

    def __len__(self) -> int:
        return len(self.label_list)

    def __getitem__(self, index: int) -> Dict:
        label = self.label_list[index]
        tt = lambda x: torch.from_numpy(x).float()
        data_dict = {}
        img_id = label["image"].split("/")
        img_id = img_id[-3][3:6] + "_" + img_id[-1][:-4]
        meta = {"source": "sloper4d", "img_id": img_id}

        data_dict["meta"] = meta

        if self.mode == "train":
            img = torch.zeros([3, 720, 1280], dtype=torch.float32)
        else:
            rgb = cv2.imread(label["image"])
            rgb = cv2.resize(rgb, [1280, 720], interpolation=cv2.INTER_LINEAR)
            rgb = np.array(rgb, dtype=np.float32)
            rgb_vis = copy.deepcopy(rgb)
            mean = np.float64(self.img_mean.reshape(1, -1))
            stdinv = 1 / np.float64(self.img_std.reshape(1, -1))
            cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB, rgb)  # type: ignore
            cv2.subtract(rgb, mean, rgb)  # type: ignore
            cv2.multiply(rgb, stdinv, rgb)  # type: ignore
            img = tt(rgb).permute(2, 0, 1)  # 3, H ,W

        fx = label["intrinsics"][0] * 1280 / 1920
        fy = label["intrinsics"][1] * 720 / 1080
        cx = label["intrinsics"][2] * 1280 / 1920
        cy = label["intrinsics"][3] * 720 / 1080
        intrinsics_old = np.eye(3)
        intrinsics_old[0, 0] = fx
        intrinsics_old[1, 1] = fy
        intrinsics_old[0, 2] = cx
        intrinsics_old[1, 2] = cy

        data_dict["intrinsics"] = tt(intrinsics_old)

        smpl_temp = self.smpl(betas=tt(label["betas"]).unsqueeze(0))
        smpl_offset = smpl_temp.joints[0, 0]

        smpl_offset_transform = np.eye(4)
        smpl_offset_transform[:3, 3] = smpl_offset.detach().cpu().numpy()

        init_trans = tt(
            np.linalg.inv(smpl_offset_transform) @ label["cam_pose"]
            @ smpl_offset_transform)

        global_orient = axis_angle_to_matrix(tt(label['global_orient']))

        transl = tt(label["global_trans"])

        global_orient = torch.einsum("ij,bjk->bik", init_trans[:3, :3],
                                     global_orient)
        transl = torch.einsum("ij,bj->bi", init_trans[:3, :3],
                              transl) + init_trans[:3, 3].unsqueeze(0)

        data_dict["img"] = img
        data_dict["global_trans"] = transl
        data_dict["global_orient"] = matrix_to_rotation_6d(global_orient)

        data_dict["betas"] = tt(label["betas"])

        data_dict["body_pose"] = matrix_to_rotation_6d(
            axis_angle_to_matrix(tt(label["body_pose"]))).reshape(-1, 23 * 6)

        if not self.use_image:
            return data_dict

        # convert x, y, z in global to x, y in pixel and relative d
        # convert x, y, z in global to x, y in pixel and relative d
        try:
            depth_3d = np.load(label["voxel"])
            depth_3d = tt(depth_3d)
        except:
            depth = cv2.imread(label["depth"], -1)
            depth = np.array(depth, dtype=np.float32)
            depth = depth / 256.  # original is unit16 format, divide 256 to convert to metric depth
            depth = cv2.resize(depth, [1280, 720],
                               interpolation=cv2.INTER_LINEAR)
            depth_3d = depth_to_3d(depth, intrinsics_old)
            depth_3d = tt(depth_3d)  # h,w,c

            semantic_raw = cv2.imread(label["semantic"], -1)
            semantic_raw = np.array(semantic_raw, dtype=np.float32)
            semantic_raw = semantic_raw / 18.  # normalize
            semantic_raw = cv2.resize(semantic_raw, [1280, 720],
                                      interpolation=cv2.INTER_NEAREST)
            semantic_raw = tt(semantic_raw).unsqueeze(-1)
            depth_3d = torch.cat([depth_3d, semantic_raw], dim=-1)  # h,w,c

            voxel_dir = label["voxel"].rsplit("/", 1)[0]
            os.makedirs(voxel_dir, exist_ok=True)
            np.save(label["voxel"], depth_3d.cpu().numpy())

        init_trans = transl[0]

        dx = int(init_trans[0] * fx / init_trans[2] + cx)
        dy = int(init_trans[1] * fy / init_trans[2] + cy)

        dx = max(min(dx, 1279), 0)
        dy = max(min(dy, 719), 0)

        metric_depth = depth_3d[dy, dx, 2]

        depth_3d[..., :3] = depth_3d[..., :3] * init_trans[2] / metric_depth
        depth_3d[
            ..., :3] = depth_3d[..., :3] - init_trans.unsqueeze(0).unsqueeze(0)

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
        bbox = label["bbox_2d"]
        mask[int(bbox[1] * 1280 / 1920):int(bbox[3] * 1280 / 1920),
             int(bbox[0] * 720 / 1080):int(bbox[2] * 720 / 1080)] = 0

        mask = reduce(torch.logical_and, [
            mask, depth_3d[..., 0] >= self.grid_size[0] + 1e-5, depth_3d[..., 0]
            < self.grid_size[1] - 1e-5, depth_3d[..., 1] >= self.grid_size[2] +
            1e-5, depth_3d[..., 1] < self.grid_size[3] - 1e-5, depth_3d[..., 2]
            >= self.grid_size[4] + 1e-5, depth_3d[...,
                                                  2] < self.grid_size[5] - 1e-5
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

        return data_dict
