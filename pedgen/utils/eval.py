import torch

from pedgen.dataset.waymo_dataset import HKP_INDEX, HKP_NAMES
from pedgen.utils.rot import matrix_to_axis_angle


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, vals):
        if torch.numel(vals) > 1:
            for val in vals:
                self.sum += val
                self.count += 1
        else:
            self.sum += vals
            self.count += 1
        self.avg = self.sum / self.count


def compute_pose_metrics(pred, gt):
    """
    calculate all metrics

    Args:
        pred: candidate prediction, shape as [50, t_pred, 3 * joints_num]
        gt: ground truth, shape as [1, t_pred, 3 * joints_num]

    Returns:
        diversity, ade, fde 
    """
    pred_init = pred[:, :1, :]
    gt_init = gt[:, :1, :]

    if pred.shape[0] > 0:
        dist_diverse = torch.pdist(pred.reshape(pred.shape[0], -1))
        apd = dist_diverse.mean()
        dist_diverse_init = torch.pdist(
            pred_init.reshape(pred_init.shape[0], -1))
        apd_init = dist_diverse_init.mean()
    else:
        apd, apd_init = 0., 0.

    dist = torch.linalg.norm(pred - gt, dim=2)  # [50, t_pred]
    dist_init = torch.linalg.norm(pred_init - gt_init, dim=2)  # [50, t_pred]
    # we can reuse 'dist' to optimize metrics calculation

    made, _ = dist.mean(dim=1).min(dim=0)
    mfde, _ = dist[:, -1].min(dim=0)
    made = made.mean()
    mfde = mfde.mean()

    aade = dist.mean(dim=1).mean(dim=0)
    afde = dist[:, -1].mean(dim=0)
    aade = aade.mean()
    afde = afde.mean()

    aade_init = dist_init.mean(dim=1).mean(dim=0)
    aade_init = aade_init.mean()

    made_init, _ = dist_init.mean(dim=1).min(dim=0)
    made_init = made_init.mean()

    return apd, aade, made, afde, mfde, apd_init, aade_init, made_init


def compute_traj_metrics(pred_trans, gt_trans, pred_rot, gt_rot):
    pred_rot_init = pred_rot[:, :1, :]
    gt_rot_init = gt_rot[:, :1, :]

    if pred_trans.shape[0] > 0:
        dist_diverse = torch.pdist(pred_trans.reshape(pred_trans.shape[0], -1))
        apd_traj = dist_diverse.mean()
    else:
        apd_traj = 0.

    dist = torch.linalg.norm(pred_trans - gt_trans, dim=2)  # [50, t_pred]

    made_traj, _ = dist.mean(dim=1).min(dim=0)
    mfde_traj, _ = dist[:, -1].min(dim=0)
    made_traj = made_traj.mean()
    mfde_traj = mfde_traj.mean()

    aade_traj = dist.mean(dim=1).mean(dim=0)
    afde_traj = dist[:, -1].mean(dim=0)
    aade_traj = aade_traj.mean()
    afde_traj = afde_traj.mean()

    delta_rot = pred_rot_init @ gt_rot_init.transpose(-2, -1)
    delta_rot = matrix_to_axis_angle(delta_rot)
    rot_dist = torch.linalg.norm(delta_rot, dim=2)
    maoe_init, _ = rot_dist.min(dim=0)
    aaoe_init = rot_dist.mean(dim=0)
    maoe_init = maoe_init.mean()
    aaoe_init = aaoe_init.mean()

    return apd_traj, aade_traj, made_traj, afde_traj, mfde_traj, aaoe_init, maoe_init


def compute_ground_metrics(pred, ground_map, init_transl, threshold=0.05):
    """
    calculate ground metrics

    Args:
        pred: candidate prediction, shape as [50, t_pred, 3 * joints_num]
        ground_map: ground height, shape as [x, z, 1]
        init_transl: translation of first frame, shape as [1, 3]
    """
    N, T, _ = pred.shape
    pred = pred.reshape(N, T, -1, 3)
    init_transl = init_transl.unsqueeze(0).unsqueeze(0)
    pred -= init_transl
    foot = torch.cat([pred[..., 7:9, :], pred[..., 10:12, :]], dim=-2)
    # foot_indices = torch.argmax(foot[..., 1], dim=-1, keepdim=True)
    foot_transl = torch.abs(torch.diff(foot, dim=1))
    foot_indices = torch.argmin(torch.linalg.norm(foot_transl, dim=-1),
                                dim=-1,
                                keepdim=True)
    foot_indices = foot_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    foot = torch.gather(foot, 2, foot_indices).squeeze(-2)

    grid_size = [-4, 4, -2, 2, -4, 4]  # hard-coded
    grid_points = [40, 40, 40]  # hard-coded
    voxel_size_x = (grid_size[1] - grid_size[0]) / grid_points[0]
    voxel_size_y = (grid_size[3] - grid_size[2]) / grid_points[1]
    voxel_size_z = (grid_size[5] - grid_size[4]) / grid_points[2]
    voxel_size = torch.Tensor([voxel_size_x, voxel_size_y,
                               voxel_size_z]).to(foot.device)
    grid_lower_bound = torch.Tensor([grid_size[0], grid_size[2],
                                     grid_size[4]]).to(foot.device)
    indices = (foot - grid_lower_bound.unsqueeze(0).unsqueeze(0)
              ) // voxel_size.unsqueeze(0).unsqueeze(0)
    # clip motion that leaves grid
    indices = indices.clip(0, grid_points[0] - 1)
    indices = indices.long()

    ground_map = ground_map.reshape(grid_points[0], grid_points[2], 1)
    dist = torch.abs(foot[..., 1:2] -
                     ground_map[indices[..., 0], indices[..., 2]]).squeeze(-1)

    # import matplotlib.pyplot as plt
    # plt.imshow(ground_map.cpu().numpy())
    # plt.savefig("ground_map")
    # ground_map[indices[0, :, 0], indices[0, :, 2]] = foot[0, :, 1:2]
    # plt.imshow(ground_map.cpu().numpy())
    # plt.savefig("ground_map_motion.png")

    mfe, _ = dist.mean(dim=1).min(dim=0)
    mfpr, _ = (dist > threshold).float().mean(dim=1).min(
        dim=0)  # floating/penetration rate
    mfe = mfe.mean()
    mfpr = mfpr.mean()

    afe = dist.mean(dim=1).mean(dim=0)
    afpr = (dist > threshold).float().mean(dim=1).mean(dim=0)
    afe = afe.mean()
    afpr = afpr.mean()

    return afe, mfe, afpr, mfpr


def compute_joint_metrics(pred, label):
    """
    calculate ground metrics

    Args:
        pred: candidate prediction, shape as [50, t_pred, 3 * joints_num]
        label: ground truth keypoints dict
            (frame_ids): shape as [num_gt_frames]
            (keypoints): shape as [num_gt_frames, keypoints_num, 4]

    Returns:
        aade, made, afde, mfde
    """
    N, T, _ = pred.shape
    pred = pred.reshape(N, T, -1, 3)
    frame_ids = label["frame_ids"] * 3  # for waymo
    # frame_ids = label["frame_ids"] * 2 # for sloper
    keypoints = label["keypoints"]
    filtered_pred = []

    for i, f in enumerate(frame_ids):
        indices = [
            HKP_INDEX[HKP_NAMES[int(j)]]
            for j in keypoints[i, :, 3]
            if int(j) in HKP_NAMES
        ]
        filtered_pred.append(pred[:, min(int(f), pred.shape[1] - 1),
                                  indices].unsqueeze(1))
    filtered_pred = torch.cat(filtered_pred, dim=1)

    dist = torch.linalg.norm(filtered_pred - keypoints[..., :3], dim=-1)
    aade = dist.mean(dim=1).mean(dim=0).mean()
    made, _ = dist.mean(dim=1).min(dim=0)
    made = made.mean()

    afde = dist[:, -1].mean(dim=0)
    mfde, _ = dist[:, -1].min(dim=0)
    mfde = mfde.mean()

    return aade, made, afde, mfde
