import os

import numpy as np
import pyrender
import torch
import trimesh

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (torch.tensor([0, 0, 0, 1],
                           device=R.device).reshape(*(1,) * len(dims), 1,
                                                    4).expand(*dims, 1, 4))
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """

    def __init__(self,
                 focal_length=[5000, 5000],
                 img_res=[1280, 720],
                 camera_center=None,
                 faces=None,
                 metallicFactor=0.0,
                 roughnessFactor=0.5):
        self.img_res = img_res
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res[0],
                                                   viewport_height=img_res[1],
                                                   point_size=1.0)
        self.focal_length = focal_length
        if camera_center is None:
            self.camera_center = [img_res[0] // 2, img_res[1] // 2]
        else:
            self.camera_center = camera_center
        self.faces = faces
        self.metallicFactor = metallicFactor
        self.roughnessFactor = roughnessFactor

    def __del__(self):
        del self.renderer
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.img_res[0],
            viewport_height=self.img_res[1],
            point_size=1.0)

    def visualize_all(self, vertices, color):

        baseColorFactors = np.hstack(
            [color[:, [2, 1, 0]],
             np.ones((color.shape[0], 1))])

        fl = self.focal_length
        verts = vertices.copy()
        color = self.__call__(verts,
                              focal_length=fl,
                              baseColorFactors=baseColorFactors)

        valid_mask = color[:, :, 3:4]

        return color[:, :, :3], valid_mask

    def __call__(self,
                 vertices,
                 focal_length=[5000, 5000],
                 baseColorFactors=[(1.0, 1.0, 0.9, 1.0)]):
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))

        for i_, verts in enumerate(vertices):
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=self.metallicFactor,
                roughnessFactor=self.roughnessFactor,
                alphaMode='OPAQUE',
                baseColorFactor=baseColorFactors[i_])

            mesh = trimesh.Trimesh(verts.copy(),
                                   self.faces.copy())  # type: ignore
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)  # type: ignore
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length[0],
                                           fy=self.focal_length[1],
                                           cx=self.camera_center[0],
                                           cy=self.camera_center[1],
                                           zfar=1000)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_lighting(scene, camera_node)

        color, rend_depth = self.renderer.render(
            scene, flags=pyrender.RenderFlags.RGBA)
        self.__del__()
        color = color.astype(np.float32) / 255.0

        return color

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color,
                                                intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)
