from functools import reduce

import cv2
import imageio
import numpy as np
import torch
import yaml
from PIL import Image
from smplx import SMPLLayer
from smplx.lbs import vertices2joints
from transformers import (SegformerFeatureExtractor,
                          SegformerForSemanticSegmentation)

from pedgen.model.pedgen_model import PedGenModel
from pedgen.utils.colors import IMG_MEAN, IMG_STD, get_colors
from pedgen.utils.renderer import Renderer
from pedgen.utils.rot import (create_2d_grid, create_occupancy_grid,
                              depth_to_3d, rotation_6d_to_matrix)

# feel free to play with the following context factors
INIT_POS = [0.5, 0.5, 5.0]  # init position
IMAGE_PATH = "scripts/demo_input.png"  # image scene context
GOALS = [[2., 0., 2.], [2., 0., 2.], [-2., 0., 2.]]  # goal context
BETAS = np.zeros(10,)  # human body shape context

# use pedgen_no_context to disable the context factors
CKPT_PATH = "ckpts/pedgen_with_context.ckpt"
CFG_PATH = "cfgs/pedgen_with_context.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"

# preprocess the image
repo = "isl-org/ZoeDepth"
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK",
                              pretrained=True).to("cuda").eval()

image = Image.open(IMAGE_PATH).convert("RGB")
depth = model_zoe_nk.infer_pil(image)
depth = depth.astype(np.float32)

image_processor = SegformerFeatureExtractor.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(device)

image = Image.open(IMAGE_PATH).convert("RGB")
inputs = image_processor(images=image, return_tensors="pt").to(device)
pred = model(**inputs)
logits = pred.logits  # shape (batch_size, num_labels, height/4, width/4)
logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],  # (height, width)
    mode="bilinear",
    align_corners=False)
segmentation = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
segmentation = segmentation.astype(np.float32)

# load the model
with open(CFG_PATH, 'r') as f:
    config = yaml.safe_load(f)

model_conf = config["model"]
num_timestamp = config["data"]["num_timestamp"]
grid_size = config["data"]["grid_size"]
grid_points = config["data"]["grid_points"]

model = PedGenModel.load_from_checkpoint(CKPT_PATH,
                                         **model_conf,
                                         map_location="cpu")
model = model.to(device)
tt = lambda x: torch.from_numpy(x).float()

# preprocess the inputs
batch = {}
rgb = cv2.imread(IMAGE_PATH)
rgb = np.array(rgb, dtype=np.float32)
mean = np.float64(IMG_MEAN).reshape(1, -1)
stdinv = 1 / np.float64(IMG_STD).reshape(1, -1)
cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB, rgb)  # type: ignore
cv2.subtract(rgb, mean, rgb)  # type: ignore
cv2.multiply(rgb, stdinv, rgb)  # type: ignore
img = tt(rgb).permute(2, 0, 1)  # 3, H, W

_, H, W = img.shape
f = (W**2 + H**2)**0.5
cx = 0.5 * W
cy = 0.5 * H
intrinsics = np.eye(3)
intrinsics[0, 0] = f
intrinsics[1, 1] = f
intrinsics[0, 2] = cx
intrinsics[1, 2] = cy

batch["intrinsics"] = tt(intrinsics)

pred_goal = torch.Tensor(GOALS)
num_pred_steps = pred_goal.shape[0]
init_trans = torch.Tensor(INIT_POS)
current_init_pos = init_trans.clone()

batch["img"] = img.unsqueeze(0).repeat(num_pred_steps, 1, 1, 1)
batch["global_trans"] = init_trans.unsqueeze(0).unsqueeze(0).repeat(
    num_pred_steps, num_timestamp, 1)
batch["global_orient"] = tt(np.zeros(6,)).unsqueeze(0).unsqueeze(0).repeat(
    num_pred_steps, num_timestamp, 1)
batch["body_pose"] = tt(np.zeros(23 * 6,)).unsqueeze(0).unsqueeze(0).repeat(
    num_pred_steps, num_timestamp, 1)
batch["betas"] = tt(BETAS).unsqueeze(0).repeat(num_pred_steps, 1)
batch["batch_size"] = 1

batch["new_img"] = []
for i in range(pred_goal.shape[0]):
    depth_3d = depth_to_3d(depth, intrinsics)
    depth_3d = tt(depth_3d)  # H, W, C

    semantic_raw = segmentation / 18.  # normalize
    semantic_raw = tt(semantic_raw).unsqueeze(-1)
    depth_3d = torch.cat([depth_3d, semantic_raw], dim=-1)  # H, W, C

    depth_3d[..., :3] = depth_3d[..., :3] - current_init_pos.unsqueeze(
        0).unsqueeze(0)

    mask = torch.ones([H, W], dtype=torch.bool)

    mask = reduce(torch.logical_and, [
        mask, depth_3d[..., 0] >= grid_size[0] + 1e-5, depth_3d[..., 0]
        < grid_size[1] - 1e-5, depth_3d[..., 1] >= grid_size[2] + 1e-5,
        depth_3d[..., 1] < grid_size[3] - 1e-5, depth_3d[..., 2]
        >= grid_size[4] + 1e-5, depth_3d[..., 2] < grid_size[5] - 1e-5
    ])
    depth_3d = depth_3d.reshape(720 * 1280, -1)
    mask = mask.flatten()
    depth_3d = depth_3d[mask, :]

    occupancy_grid = create_occupancy_grid(
        depth_3d,
        grid_size,
        grid_points,
    )

    grid_2d = tt(create_2d_grid(num_points=grid_points, grid_size=grid_size))

    occupancy_grid = occupancy_grid.permute(0, 2, 1)
    occupancy_grid = torch.cat([occupancy_grid, grid_2d], dim=-1)

    occupancy_grid = occupancy_grid.reshape(
        occupancy_grid.shape[0] * occupancy_grid.shape[1], -1)

    batch["new_img"].append(occupancy_grid)

    batch["global_trans"][i, -1] = batch["global_trans"][i, 0] + pred_goal[i]
    current_init_pos += pred_goal[i]

batch["new_img"] = torch.stack(batch["new_img"], dim=0)
batch = {
    k: batch[k].to(device) if isinstance(batch[k], torch.Tensor) else batch[k]
    for k in batch
}

# run model inference
pred = model.predict_step(batch)
pred = {k: pred[k].cpu() for k in pred}

smpl = SMPLLayer(model_path="smpl", gender='neutral')
colors = get_colors()


# visualize the outputs
def vis_smpl_impl(render, img, pred_vertices, id):
    img_vis = img.copy()
    img_smpl, valid_mask = render.visualize_all(
        pred_vertices[[0, 10, 20, 30, 40, 50, 59]].cpu().numpy(),
        colors[[7, 6, 5, 4, 3, 2, 1]],
    )

    img_vis = img_smpl[:, :, :3] * valid_mask + (1 -
                                                 valid_mask) * img_vis / 255.
    img_vis = img_vis * 255
    img_vis = img_vis.astype(np.uint8)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite("scripts/demo_ouput.png", img_vis)

    writer = imageio.get_writer(
        "scripts/demo_output.mp4",
        fps=30,
        mode='I',
        format='FFMPEG',  # type: ignore
        macro_block_size=1)

    for t in range(pred_vertices.shape[0]):
        img_pred = img.copy()
        img_smpl = img_pred.copy()
        img_smpl, valid_mask = render.visualize_all(
            pred_vertices[[t]].cpu().numpy(),
            colors[[1]],
        )

        img_pred = img_smpl[:, :, :3] * valid_mask + (
            1 - valid_mask) * img_pred / 255.
        img_pred = img_pred * 255
        img_pred = img_pred.astype(np.uint8)
        writer.append_data(img_pred)
    writer.close()


B, N, T, _ = pred["pred_global_trans"].shape

for i in range(B):
    body_pose = rotation_6d_to_matrix(pred["pred_body_pose"][i].reshape(
        -1, 23, 6))

    pred_transl = pred["pred_global_trans"][i]
    pred_rot = rotation_6d_to_matrix(pred["pred_global_orient"][i])

    pred_smpl_output = smpl(
        transl=pred_transl.reshape(-1, 3),
        betas=batch["betas"][i].unsqueeze(0).unsqueeze(0).repeat(
            N, T, 1).reshape(-1, 10).cpu(),
        global_orient=pred_rot.reshape(-1, 3, 3),
        body_pose=body_pose,
    )

    pred_joint_locations = vertices2joints(
        smpl.J_regressor,  # type: ignore
        pred_smpl_output.vertices)

    pred_joint_locations = pred_joint_locations.reshape(N, T, -1)

    # visualization
    print(f"Visualizing...")

    intrisics = batch["intrinsics"].cpu().numpy()
    render = Renderer(focal_length=[intrisics[0, 0], intrisics[1, 1]],
                      camera_center=[intrisics[0, 2], intrisics[1, 2]],
                      img_res=[W, H],
                      faces=smpl.faces,
                      metallicFactor=0.0,
                      roughnessFactor=0.7)

    img = batch["img"][i, :3, :, :].cpu().permute(1, 2, 0).numpy()  # B, H, W, C
    # unnormalize
    mean = np.array(IMG_MEAN)[None, None, :]
    std = np.array(IMG_STD)[None, None, :]
    img = img * std + mean

    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.uint8)

    pred_vertices = pred_smpl_output.vertices
    pred_vertices = pred_vertices.reshape(N, T, -1, 3)
    for j in range(pred_vertices.shape[0]):
        # only vis the first sample
        vis_smpl_impl(render, img, pred_vertices[j], j)
        break

    del render
