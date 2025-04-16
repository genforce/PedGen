import copy
import os
from collections import defaultdict
from functools import reduce
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from smplx import SMPLLayer
from smplx.lbs import vertices2joints
from torch.utils.data import Dataset
from tqdm import tqdm

from pedgen.utils.rot import create_2d_grid, create_occupancy_grid, depth_to_3d

HKP_NAMES = {
    1: 'nose',
    5: 'lshoulder',
    6: 'lelbow',
    7: 'lwrist',
    8: 'lhip',
    9: 'lknee',
    10: 'lankle',
    13: 'rshoulder',
    14: 'relbow',
    15: 'rwrist',
    16: 'rhip',
    17: 'rknee',
    18: 'rankle',
    19: 'head',
    20: 'head'
}

# based on smpl joints
HKP_INDEX = {
    'nose': 15,  # no nose in smpl
    'lshoulder': 17,
    'lelbow': 19,
    'lwrist': 21,
    'lhip': 2,  # no hip in smpl
    'lknee': 5,
    'lankle': 8,
    'rshoulder': 16,
    'relbow': 18,
    'rwrist': 20,
    'rhip': 1,  # no hip in smpl
    'rknee': 4,
    'rankle': 7,
    'head': 15  # no head in smpl
}

from pedgen.utils.colors import IMG_MEAN, IMG_STD


class WaymoDataset(Dataset):
    """Lightning dataset for pedestrian generation with WaymoDataset."""

    def __init__(self,
                 data_root: str,
                 depth_root: str,
                 semantic_root: str,
                 num_timestamp: int,
                 grid_size: List = [-4, 4, -2, 2, -4, 4],
                 grid_points: List = [40, 40, 40],
                 *args,
                 **kwargs) -> None:
        self.data_root = data_root
        self.depth_root = depth_root
        self.semantic_root = semantic_root
        self.grid_size = grid_size
        self.grid_points = grid_points
        self.num_timestamp = num_timestamp
        self.img_mean = np.array(IMG_MEAN)
        self.img_std = np.array(IMG_STD)

        smpl = SMPLLayer(model_path="smpl", gender='neutral')
        smpl_output = smpl()
        joints = vertices2joints(
            smpl.J_regressor,  # type: ignore
            smpl_output.vertices)
        # ankle = joints[0, 7] (translate to chest)
        ankle = joints[0, 7] - joints[0, 0]  # (translate to pelvis)
        self.ankle_to_root_trans = ankle

        self.label_list = []
        print("initializing waymo dataset...")
        for segment in tqdm(os.listdir(f"{data_root}/lidar_hkp")):
            camera_image = pd.read_parquet(
                f"{data_root}/camera_image/{segment}")
            camera_calibration = pd.read_parquet(
                f"{data_root}/camera_calibration/{segment}")
            # use front camera
            camera_calibration = camera_calibration[
                camera_calibration['key.camera_name'] == 1]
            camera_intrinsic = np.array([
                [
                    camera_calibration[
                        '[CameraCalibrationComponent].intrinsic.f_u'].values[0],
                    0, camera_calibration[
                        '[CameraCalibrationComponent].intrinsic.c_u'].values[0]
                ],
                [
                    0, camera_calibration[
                        '[CameraCalibrationComponent].intrinsic.f_v'].values[0],
                    camera_calibration[
                        '[CameraCalibrationComponent].intrinsic.c_v'].values[0]
                ], [0, 0, 1]
            ])
            camera_extrinsic = camera_calibration[
                '[CameraCalibrationComponent].extrinsic.transform'].values[
                    0].reshape(4, 4)
            camera_box = pd.read_parquet(f"{data_root}/camera_box/{segment}")
            camera_to_lidar_box_association = pd.read_parquet(
                f"{data_root}/camera_to_lidar_box_association/{segment}")
            camera_to_lidar_box_association = camera_to_lidar_box_association[
                camera_to_lidar_box_association['key.camera_name'] == 1]
            if len(camera_to_lidar_box_association) == 0:
                continue

            lidar_hkp = pd.read_parquet(f"{data_root}/lidar_hkp/{segment}")
            vehicle_pose = pd.read_parquet(
                f"{data_root}/vehicle_pose/{segment}")
            segment = segment[:-8]
            images = sorted(os.listdir(f"{data_root}/visualize/{segment}"))
            for id in lidar_hkp['key.laser_object_id'].unique():
                person = lidar_hkp[lidar_hkp['key.laser_object_id'] == id]
                camera_object_id = camera_to_lidar_box_association[
                    camera_to_lidar_box_association['key.laser_object_id'] ==
                    id]['key.camera_object_id'].values
                if len(camera_object_id) == 0:
                    continue
                camera_object_id = camera_object_id[0]
                bboxes = camera_box[camera_box['key.camera_object_id'] ==
                                    camera_object_id]
                timestamps = sorted(person['key.frame_timestamp_micros'])
                label = defaultdict(list)
                for t in timestamps:
                    if "timestamps" in label and t - label["timestamps"][
                            0] > 2e6:
                        vehicle_poses = vehicle_pose[
                            vehicle_pose['key.frame_timestamp_micros'].isin(
                                label["timestamps"]
                            )]['[VehiclePoseComponent].world_from_vehicle.transform'].values
                        vehicle_poses = [
                            pose.reshape(4, 4) for pose in vehicle_poses
                        ]
                        label["vehicle_poses"] = vehicle_poses
                        label["camera_intrinsic"] = [camera_intrinsic]
                        label["camera_extrinsic"] = [camera_extrinsic]
                        bbox = bboxes[bboxes['key.frame_timestamp_micros'] ==
                                      label["timestamps"][0]]
                        cx = bbox['[CameraBoxComponent].box.center.x'].values
                        cy = bbox['[CameraBoxComponent].box.center.y'].values
                        sx = bbox['[CameraBoxComponent].box.size.x'].values
                        sy = bbox['[CameraBoxComponent].box.size.y'].values
                        label["bbox"] = [np.concatenate([cx, cy, sx, sy])]
                        valid_bbox = len(bbox) > 0
                        joints = reduce(np.intersect1d,
                                        [k[:, 3] for k in label["keypoints"]])
                        for i in range(len(label["keypoints"])):
                            label["keypoints"][i] = np.stack([
                                k for k in label["keypoints"][i]
                                if k[3] in joints
                            ])
                        valid_keypoints = (10 in label["keypoints"][0][:, 3]
                                          ) or (18 in label["keypoints"][0][:,
                                                                            3])
                        if valid_bbox and valid_keypoints:  # check if at least one foot joint exists
                            self.label_list.append(label)
                        label = defaultdict(list)
                    label["img"].append(
                        f"{data_root}/visualize/{segment}/{t}.png")
                    label["timestamps"].append(t)
                    frame_id = images.index(
                        label["img"][-1].split('/')[-1]) - images.index(
                            label["img"][0].split('/')[-1])
                    label["frame_ids"].append(frame_id)
                    # if not os.path.exists(f"{data_root}/visualize/{segment}/{t}.png"):
                    #     image = Image.open(BytesIO(
                    #         camera_image[(camera_image['key.frame_timestamp_micros'] == t).values &
                    #                      (camera_image['key.camera_name'] == 1).values]['[CameraImageComponent].image'][0]))
                    #     os.makedirs(
                    #         f"{data_root}/visualize/{segment}", exist_ok=True)
                    #     image.save(f"{data_root}/visualize/{segment}/{t}.png")
                    j = person[person['key.frame_timestamp_micros'] == t][
                        '[LiDARHumanKeypointsComponent].lidar_keypoints[*].type'].values[
                            0][..., np.newaxis]
                    x = person[person['key.frame_timestamp_micros'] == t][
                        '[LiDARHumanKeypointsComponent].lidar_keypoints[*].keypoint_3d.location_m.x'].values[
                            0][..., np.newaxis]
                    y = person[person['key.frame_timestamp_micros'] == t][
                        '[LiDARHumanKeypointsComponent].lidar_keypoints[*].keypoint_3d.location_m.y'].values[
                            0][..., np.newaxis]
                    z = person[person['key.frame_timestamp_micros'] == t][
                        '[LiDARHumanKeypointsComponent].lidar_keypoints[*].keypoint_3d.location_m.z'].values[
                            0][..., np.newaxis]
                    label["keypoints"].append(np.hstack([x, y, z,
                                                         j]))  # type: ignore
        print(f"selected {len(self.label_list)} samples")

    def __len__(self) -> int:
        return len(self.label_list)

    def __getitem__(self, index: int) -> dict:

        def tt(x):
            return torch.from_numpy(x).float()

        label = self.label_list[index]
        data_dict = {}
        rgb = cv2.imread(label["img"][0])
        rgb = cv2.resize(rgb, [1280, 720], interpolation=cv2.INTER_LINEAR)
        rgb = np.array(rgb, dtype=np.float32)
        mean = np.float64(self.img_mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.img_std.reshape(1, -1))
        cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB, rgb)  # type: ignore
        cv2.subtract(rgb, mean, rgb)  # type: ignore
        cv2.multiply(rgb, stdinv, rgb)  # type: ignore
        data_dict["img"] = tt(rgb).permute(2, 0, 1)

        image = cv2.imread(label["img"][0])
        image = cv2.resize(image, [1280, 720], interpolation=cv2.INTER_LINEAR)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
        img_h, img_w, _ = image.shape

        timestamps = label["timestamps"]
        data_dict["timestamps"] = tt(np.stack(timestamps))
        data_dict["frame_ids"] = tt(np.stack(label["frame_ids"]))

        keypoints = label["keypoints"]
        keypoints = np.stack(keypoints)
        vehicle_poses = np.stack(label["vehicle_poses"])
        world_to_cam = []
        for pose in vehicle_poses:
            # world_to_cam.append(np.linalg.inv(
            #     pose @ label["camera_extrinsic"][0]))
            world_to_cam.append(
                np.linalg.inv(vehicle_poses[0] @ label["camera_extrinsic"][0]))
        world_to_cam = np.stack(world_to_cam)

        world_keypoints = np.einsum(
            "nij,nkj->nki", vehicle_poses,
            np.concatenate(
                [keypoints[..., :3],
                 np.ones((*keypoints.shape[:-1], 1))],
                axis=-1))
        keypoints[..., :3] = np.einsum("nij,nkj->nki", world_to_cam,
                                       world_keypoints)[..., :3]
        keypoints = np.stack([
            -keypoints[..., 1], -keypoints[..., 2], keypoints[..., 0],
            keypoints[..., 3]
        ],
                             axis=-1)
        data_dict["keypoints"] = tt(keypoints)

        transl = keypoints[0, :, :3].mean(axis=-2)
        ankles = []
        if 10 in keypoints[0, :, 3]:  # lankle
            ankles.append(keypoints[0, list(keypoints[0, :, 3]).index(10), 1])
        if 18 in keypoints[0, :, 3]:  # rankle
            ankles.append(keypoints[0, list(keypoints[0, :, 3]).index(18), 1])
        transl[1] = np.mean(ankles) + self.ankle_to_root_trans[1]

        intrinsics = copy.deepcopy(label["camera_intrinsic"][0])
        intrinsics[0] = intrinsics[0] * 1280 / 1920
        intrinsics[1] = intrinsics[1] * 720 / 1280

        import matplotlib.pyplot as plt
        bbox = label["bbox"][0]  # cx, cy, sx, sy
        bbox[0] *= 1280 / 1920
        bbox[1] *= 720 / 1280
        bbox[2] *= 1280 / 1920
        bbox[3] *= 720 / 1280
        image = cv2.rectangle(
            image, (int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2)),
            (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)),
            color=[255, 0, 0],
            thickness=1)
        projected_transl = intrinsics @ transl
        projected_transl = projected_transl / projected_transl[..., 2:]
        image = cv2.circle(image,
                           (int(projected_transl[0]), int(projected_transl[1])),
                           radius=5,
                           color=[0, 255, 0],
                           thickness=-1)
        for i in range(keypoints.shape[0]):
            projected_keypoints = np.einsum("ij,nj->ni", intrinsics,
                                            keypoints[i, ..., :3])
            projected_keypoints = projected_keypoints / \
                projected_keypoints[..., 2:]
            u = projected_keypoints[:, 0]
            v = projected_keypoints[:, 1]
            for ju, jv in zip(u, v):
                image = cv2.circle(image, (int(ju), int(jv)),
                                   radius=5,
                                   color=[255, 0, 0],
                                   thickness=-1)
            break

        # for debugging
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f"debug/frame_{index}.png", image)
        # import ipdb; ipdb.set_trace()

        label["voxel"] = label["img"][0].replace("visualize", "voxel")
        label["semantic"] = label["img"][0].replace("visualize", "semantic")
        label["depth"] = label["img"][0].replace("visualize", "depth")
        if True:  # not os.path.exists(label["voxel"].replace("png", "npy")):
            depth = cv2.imread(label["depth"], -1)
            depth = cv2.resize(depth, [1280, 720],
                               interpolation=cv2.INTER_LINEAR)
            # cv2.imwrite(f"debug/depth_{index}.png", depth)
            depth = np.array(depth, dtype=np.float32)
            depth = depth / 256.  # original is unit16 format, divide 256 to convert to metric depth
            depth_3d = depth_to_3d(depth, intrinsics)
            depth_3d = tt(depth_3d)  # h,w,c

            semantic_raw = cv2.imread(label["semantic"], -1)
            semantic_raw = cv2.resize(semantic_raw, [1280, 720],
                                      interpolation=cv2.INTER_NEAREST)
            semantic_raw = np.array(semantic_raw, dtype=np.float32)
            semantic_raw = semantic_raw / 18.  # normalize
            semantic_raw = tt(semantic_raw).unsqueeze(-1)
            depth_3d = torch.cat([depth_3d, semantic_raw], dim=-1)  # h,w,c
            init_trans = tt(transl)

            # dx = int(init_trans[0] *
            #          (img_w ** 2 + img_h ** 2) ** 0.5 / init_trans[2] + img_w / 2)
            # dy = int(init_trans[1] *
            #          (img_w ** 2 + img_h ** 2) ** 0.5 / init_trans[2] + img_h / 2)
            dx = int(init_trans[0] * intrinsics[0][0] / init_trans[2] +
                     intrinsics[0][2])
            dy = int(init_trans[1] * intrinsics[1][1] / init_trans[2] +
                     intrinsics[1][2])

            dx = max(min(dx, img_w - 1), 0)
            dy = max(min(dy, img_h - 1), 0)

            metric_depth = depth[dy, dx]

            depth_3d[..., :3] = depth_3d[..., :3] * init_trans[2] / metric_depth
            depth_3d[..., :3] = depth_3d[..., :3] - init_trans.unsqueeze(
                0).unsqueeze(0)

            # voxel_dir = label["voxel"].rsplit("/", 1)[0]
            # os.makedirs(voxel_dir, exist_ok=True)
            # np.save(label["voxel"][:-4], depth_3d.cpu().numpy())
        else:
            depth_3d = np.load(label["voxel"].replace("png", "npy"))
            depth_3d = tt(depth_3d)

        mask = torch.ones([img_h, img_w], dtype=torch.bool)
        bbox = label["bbox"][0]  # cx, cy, sx, sy
        bbox[0] *= 1280 / 1920
        bbox[1] *= 720 / 1280
        bbox[2] *= 1280 / 1920
        bbox[3] *= 720 / 1280
        mask[int(bbox[1] - bbox[3] / 2):int(bbox[1] + bbox[3] / 2),
             int(bbox[0] - bbox[2] / 2):int(bbox[0] + bbox[2] / 2)] = 0

        grid_mask = reduce(torch.logical_and, [
            depth_3d[..., 0] >= self.grid_size[0] + 1e-5, depth_3d[..., 0]
            < self.grid_size[1] - 1e-5, depth_3d[..., 1] >= self.grid_size[2] +
            1e-5, depth_3d[..., 1] < self.grid_size[3] - 1e-5, depth_3d[..., 2]
            >= self.grid_size[4] + 1e-5, depth_3d[...,
                                                  2] < self.grid_size[5] - 1e-5
        ])

        depth_3d = depth_3d.reshape(img_h * img_w, -1)

        mask = torch.logical_and(mask, grid_mask)
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

        occupancy_grid = torch.cat([occupancy_grid, grid_2d], dim=-1)
        occupancy_grid = occupancy_grid.reshape(
            occupancy_grid.shape[0] * occupancy_grid.shape[1], -1)

        data_dict["new_img"] = occupancy_grid

        data_dict["vehicle_poses"] = tt(vehicle_poses)
        data_dict["camera_intrinsic"] = tt(label["camera_intrinsic"][0])
        data_dict["camera_extrinsic"] = tt(label["camera_extrinsic"][0])

        data_dict["global_trans"] = tt(transl).unsqueeze(0).repeat(
            self.num_timestamp, 1)
        data_dict["global_orient"] = tt(np.zeros(6,)).unsqueeze(0).repeat(
            self.num_timestamp, 1)
        data_dict["betas"] = tt(np.zeros(10,))
        data_dict["body_pose"] = tt(np.zeros(23 * 6,)).unsqueeze(0).repeat(
            self.num_timestamp, 1)

        meta = {
            "source":
                "waymo",
            "img_id":
                f"{label['img'][0].split('/')[-2]}-{label['img'][0].split('/')[-1][:-4]}"
        }
        data_dict["meta"] = meta

        return data_dict


def collate_fn_waymo(data):
    img_batch = []
    new_img_batch = []
    timestamps_batch = []
    frame_ids_batch = []
    keypoints_batch = []
    vehicle_poses_batch = []
    camera_intrinsic_batch = []
    camera_extrinsic_batch = []
    global_trans_batch = []
    global_orient_batch = []
    betas_batch = []
    body_pose_batch = []
    meta_batch = []
    for data_dict in data:
        img_batch.append(data_dict["img"])
        new_img_batch.append(data_dict["new_img"])
        timestamps_batch.append(data_dict["timestamps"])
        frame_ids_batch.append(data_dict["frame_ids"])
        keypoints_batch.append(data_dict["keypoints"])
        vehicle_poses_batch.append(data_dict["vehicle_poses"])
        camera_intrinsic_batch.append(data_dict["camera_intrinsic"])
        camera_extrinsic_batch.append(data_dict["camera_extrinsic"])
        global_trans_batch.append(data_dict["global_trans"])
        global_orient_batch.append(data_dict["global_orient"])
        betas_batch.append(data_dict["betas"])
        body_pose_batch.append(data_dict["body_pose"])
        meta_batch.append(data_dict["meta"])
    ret_dict = {
        "img": torch.stack(img_batch),
        "new_img": torch.stack(new_img_batch),
        "timestamps":
            timestamps_batch,  # timestamps have unequal length across bach
        "frame_ids": frame_ids_batch,
        "keypoints":
            keypoints_batch,  # keypoints have unequal length across batch
        # vehicle poses have unequal length across batch
        "vehicle_poses": vehicle_poses_batch,
        "intrinsics": torch.stack(camera_intrinsic_batch),
        "camera_extrinsic": torch.stack(camera_extrinsic_batch),
        "global_trans": torch.stack(global_trans_batch),
        "global_orient": torch.stack(global_orient_batch),
        "betas": torch.stack(betas_batch),
        "body_pose": torch.stack(body_pose_batch),
        "meta": meta_batch
    }

    return ret_dict
