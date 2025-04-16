# Some functions are borrowed from https://github.com/akanazawa/human_dynamics/blob/master/src/evaluation/eval_util.py
# Adhere to their licence to use these functions

import torch


def compute_jpe(S1, S2):
    return torch.sqrt(((S1 - S2)**2).sum(dim=-1)).mean(dim=-1).numpy()


# The functions below are borrowed from SLAHMR official implementation.
# Reference: https://github.com/vye16/slahmr/blob/main/slahmr/eval/tools.py
def global_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_glob, R_glob, t_glob = align_pcl(gt_joints.reshape(-1, 3),
                                       pred_joints.reshape(-1, 3))
    pred_glob = (s_glob * torch.einsum("ij,tnj->tni", R_glob, pred_joints) +
                 t_glob[None, None])
    return pred_glob


def first_align_joints(gt_joints, pred_joints):
    """
    align the first two frames
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    # (1, 1), (1, 3, 3), (1, 3)
    s_first, R_first, t_first = align_pcl(gt_joints[:2].reshape(1, -1, 3),
                                          pred_joints[:2].reshape(1, -1, 3))
    pred_first = (s_first * torch.einsum("tij,tnj->tni", R_first, pred_joints) +
                  t_first[:, None])
    return pred_first


def local_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_loc, R_loc, t_loc = align_pcl(gt_joints, pred_joints)
    pred_loc = (
        s_loc[:, None] * torch.einsum("tij,tnj->tni", R_loc, pred_joints) +
        t_loc[:, None])
    return pred_loc


def align_pcl(Y, X, weight=None, fixed_scale=False):
    """align similarity transform to align X with Y using umeyama method
    X' = s * R * X + t is aligned with Y
    :param Y (*, N, 3) first trajectory
    :param X (*, N, 3) second trajectory
    :param weight (*, N, 1) optional weight of valid correspondences
    :returns s (*, 1), R (*, 3, 3), t (*, 3)
    """
    *dims, N, _ = Y.shape
    N = torch.ones(*dims, 1, 1) * N

    if weight is not None:
        Y = Y * weight
        X = X * weight
        N = weight.sum(dim=-2, keepdim=True)  # (*, 1, 1)

    # subtract mean
    my = Y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]  # (*, N, 3)
    x0 = X - mx[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = torch.eye(3).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fixed_scale:
        s = torch.ones(*dims, 1, device=Y.device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2),
                        keepdim=True) / N  # (*, 1, 1)
        s = (torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(
            dim=-1, keepdim=True) / var[..., 0])  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    return s, R, t


def compute_foot_distance(target_output,
                          pred_output,
                          masks,
                          gt_points,
                          thr=1e-2):
    foot_idxs = [3216, 3387, 6617, 6787]

    # Compute contact label
    foot_loc = target_output.vertices[masks][:, foot_idxs]
    foot_disp = (foot_loc[1:] - foot_loc[:-1]).norm(2, dim=-1)
    contact = foot_disp[:] < thr

    pred_feet_loc = pred_output.vertices[:, foot_idxs]  # B, 4, 3
    pred_feet_loc = pred_feet_loc[:-1][contact]  # M, 3

    distance = (pred_feet_loc.unsqueeze(1).cuda() -
                gt_points.unsqueeze(0).cuda()).norm(2, dim=-1)
    distance = torch.min(distance, dim=1)[0]

    return distance.cpu().numpy()


def compute_foot_sliding(target_output, pred_output, masks, thr=1e-2):
    """compute foot sliding error
    The foot ground contact label is computed by the threshold of 1 cm/frame
    Args:
        target_output (SMPL ModelOutput).
        pred_output (SMPL ModelOutput).
        masks (N).
    Returns:
        error (N frames in contact).
    """

    # Foot vertices idxs
    foot_idxs = [3216, 3387, 6617, 6787]

    # Compute contact label
    foot_loc = target_output.vertices[masks][:, foot_idxs]
    foot_disp = (foot_loc[1:] - foot_loc[:-1]).norm(2, dim=-1)
    contact = foot_disp[:] < thr

    pred_feet_loc = pred_output.vertices[:, foot_idxs]
    pred_disp = (pred_feet_loc[1:] - pred_feet_loc[:-1]).norm(2, dim=-1)

    error = pred_disp[contact]

    return error.cpu().numpy()


def compute_jitter(pred_output, fps=30):
    """compute jitter of the motion
    Args:
        pred_output (SMPL ModelOutput).
        fps (float).
    Returns:
        jitter (N-3).
    """

    pred3d = pred_output.joints[:, :24]

    pred_jitter = torch.norm(
        (pred3d[3:] - 3 * pred3d[2:-1] + 3 * pred3d[1:-2] - pred3d[:-3]) *
        (fps**3),
        dim=2,
    ).mean(dim=-1)

    return pred_jitter.cpu().numpy() / 10.0


def compute_rte(target_trans, pred_trans):
    # Compute the global alignment
    _, rot, trans = align_pcl(target_trans[None, :],
                              pred_trans[None, :],
                              fixed_scale=True)
    pred_trans_hat = (torch.einsum("tij,tnj->tni", rot, pred_trans[None, :]) +
                      trans[None, :])[0]

    # Compute the entire displacement of ground truth trajectory
    disps, disp = [], 0
    for p1, p2 in zip(target_trans, target_trans[1:]):
        delta = (p2 - p1).norm(2, dim=-1)
        disp += delta
        disps.append(disp)

    # Compute absolute root-translation-error (RTE)
    rte = torch.norm(target_trans - pred_trans_hat, 2, dim=-1)

    # Normalize it to the displacement
    return (rte / disp).numpy()
