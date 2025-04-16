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
from pedgen.utils.rot import create_2d_grid, create_occupancy_grid, depth_to_3d


# for inference only
class CarlaDataset(Dataset):
    """Lightning dataset for pedestrian generation."""

    def __init__(
        self,
        mode: str,
        data_root: str,
        num_timestamp: int,
        img_dim: List,
        grid_size: List,
        grid_points: int,
        **kwargs,
    ) -> None:
        with open(os.path.join(data_root, "label.pkl"), "rb") as f:
            labels = pickle.load(f)
        self.label_list = []
        self.img_mean = np.array(IMG_MEAN)
        self.img_std = np.array(IMG_STD)
        self.mode = mode
        self.img_w = img_dim[1]
        self.img_h = img_dim[0]
        self.num_timestamp = num_timestamp
        self.grid_size = grid_size
        self.grid_points = grid_points

        assert self.mode in ["test", "pred"]

        for val in labels:
            image_path = os.path.join(data_root, "image", val["image"])
            depth_path = os.path.join(data_root, "depth", val["image"])
            semantic_path = os.path.join(data_root, "semantic", val["image"])

            new_val = {}
            new_val["image"] = image_path
            new_val["depth"] = depth_path
            new_val["semantic"] = semantic_path
            new_val["map_info"] = val["map_info"]
            new_val["global_trans"] = val["global_trans"]
            new_val["global_trans_goal"] = val["global_trans_goal"]
            new_val["betas"] = val["betas"]
            new_val["sensor_pose"] = val["sensor_pose"]
            self.label_list.append(new_val)

    def __len__(self) -> int:
        return len(self.label_list)

    def __getitem__(self, index: int) -> Dict:
        label = self.label_list[index]

        def tt(x):
            return torch.from_numpy(x).float()

        data_dict = {}
        img_id = label["image"].split("/")
        img_id = img_id[-1][:-4]
        meta = {
            "source": "carla",
            "sensor_pose": label["sensor_pose"],
            "map_info": label["map_info"],
            "img_id": img_id,
        }
        data_dict["meta"] = meta

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

        data_dict["intrinsics"] = tt(intrinsics_old)

        # label["global_trans"][2] = label["global_trans"][2] - 0.2
        # label["global_trans_goal"][2] = label["global_trans_goal"][2] - 0.2

        transl = tt(label["global_trans"]).unsqueeze(0).repeat(
            self.num_timestamp, 1)
        if "global_trans_goal" in label:
            transl[-1] = tt(label["global_trans_goal"])

        data_dict["global_trans"] = transl

        data_dict["betas"] = tt(label["betas"])

        data_dict["global_orient"] = tt(np.zeros(6,)).unsqueeze(0).repeat(
            self.num_timestamp, 1)

        data_dict["body_pose"] = tt(np.zeros(23 * 6,)).unsqueeze(0).repeat(
            self.num_timestamp, 1)

        data_dict["img"] = img
        if not self.mode == "pred":

            depth_raw = cv2.imread(label["depth"])
            cv2.cvtColor(depth_raw, cv2.COLOR_BGR2RGB, depth_raw)
            depth = 1000 * (depth_raw[..., 0] + 256 * depth_raw[..., 1] + 256 *
                            256 * depth_raw[..., 2]) / (256 * 256 * 256 - 1)
            depth = np.array(depth, dtype=np.float32)
            depth_3d = depth_to_3d(depth, intrinsics_old)
            depth_3d = tt(depth_3d)  # h,w,c

            semantic_raw = cv2.imread(label["semantic"])
            semantic_raw = semantic_raw[..., 2]
            semantic_raw = np.array(semantic_raw, dtype=np.float32)
            semantic_raw = semantic_raw - 1
            semantic_raw[semantic_raw == 19] = 3  # static to building
            semantic_raw[semantic_raw == 20] = 3  # dynamic to building
            semantic_raw[semantic_raw > 20] = 0
            semantic_raw[semantic_raw < 0] = 0
            semantic_raw = semantic_raw / 18.  # normalize
            semantic_raw = tt(semantic_raw).unsqueeze(-1)
            depth_3d = torch.cat([depth_3d, semantic_raw], dim=-1)  # h,w,c

            init_trans = data_dict["global_trans"][0]
            depth_3d[..., :3] = depth_3d[..., :3] - init_trans.unsqueeze(
                0).unsqueeze(0)

            mask = torch.ones([720, 1280], dtype=torch.bool)
            mask = reduce(torch.logical_and, [
                mask, depth_3d[..., 0] >= -self.grid_size[0], depth_3d[..., 0]
                < self.grid_size[1], depth_3d[..., 1] >= -self.grid_size[2],
                depth_3d[..., 1] < self.grid_size[3], depth_3d[..., 2]
                >= -self.grid_size[4], depth_3d[..., 2] < self.grid_size[5]
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

                depth_raw = cv2.imread(label["depth"])
                cv2.cvtColor(depth_raw, cv2.COLOR_BGR2RGB, depth_raw)
                depth = 1000 * (depth_raw[..., 0] + 256 * depth_raw[..., 1] +
                                256 * 256 * depth_raw[..., 2]) / (
                                    256 * 256 * 256 - 1)
                depth = np.array(depth, dtype=np.float32)
                depth_3d = depth_to_3d(depth, intrinsics_old)
                depth_3d = tt(depth_3d)  # h,w,c

                semantic_raw = cv2.imread(label["semantic"])
                semantic_raw = semantic_raw[..., 2]
                semantic_raw = np.array(semantic_raw, dtype=np.float32)
                semantic_raw = semantic_raw - 1
                semantic_raw[semantic_raw == 19] = 3  # static to building
                semantic_raw[semantic_raw == 20] = 3  # dynamic to building
                semantic_raw[semantic_raw > 20] = 0
                semantic_raw[semantic_raw < 0] = 0
                semantic_raw = semantic_raw / 18.  # normalize
                semantic_raw = tt(semantic_raw).unsqueeze(-1)
                depth_3d = torch.cat([depth_3d, semantic_raw], dim=-1)  # h,w,c

                init_trans = data_dict["global_trans"][0]
                depth_3d[..., :3] = depth_3d[
                    ..., :3] - current_init_pos.unsqueeze(0).unsqueeze(0)

                mask = torch.ones([720, 1280], dtype=torch.bool)
                mask = reduce(torch.logical_and, [
                    mask, depth_3d[..., 0] >= -self.grid_size[0], depth_3d[...,
                                                                           0]
                    < self.grid_size[1], depth_3d[..., 1] >= -self.grid_size[2],
                    depth_3d[..., 1] < self.grid_size[3], depth_3d[..., 2]
                    >= -self.grid_size[4], depth_3d[..., 2] < self.grid_size[5]
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

    def create_3d_grid(self, grid_size=1.0, num_points=3):
        x = np.linspace(-grid_size, grid_size, num_points)
        y = np.linspace(-grid_size, grid_size, num_points)
        z = np.linspace(-grid_size, grid_size, num_points)

        # Create a mesh grid (3D)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Stack the coordinates for easy iteration or display
        grid_points = np.stack([X, Y, Z], axis=-1)
        grid_points = grid_points.reshape(-1, 3)

        return grid_points
