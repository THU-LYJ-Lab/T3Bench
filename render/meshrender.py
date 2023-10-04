import os
# switch to "osmesa" or "egl" before loading pyrender
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image
from tqdm import tqdm
import imageio

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--name', type=str)
args = parser.parse_args()

obj_path = args.path

# load mesh
try:
    mesh = trimesh.load(obj_path)
except:
    icosphere = trimesh.creation.icosphere(subdivisions=2)
    for idx, v in enumerate(icosphere.vertices):
        for cam_idx in range(5):
            color = Image.fromarray(np.zeros((512, 512, 3)).astype(np.uint8))
            imageio.imwrite(f'{args.name}/{idx:03d}_{cam_idx}.png', color)
    exit(0)

# normalize mesh to unit cube [-1, 1]^3
mesh.vertices -= mesh.vertices.mean(axis=0)
mesh.vertices /= np.abs(mesh.vertices).max()

mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
mesh.primitives[0].material.baseColorFactor = [1., 1., 1., 1.]

Radius = 2.2

# compose scene
scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[0, 0, 0])
cameras = [
    pyrender.PerspectiveCamera(yfov=np.pi / 3.0),
    pyrender.PerspectiveCamera(yfov=np.pi / 4.0),
    pyrender.PerspectiveCamera(yfov=np.pi / 5.0),
    pyrender.PerspectiveCamera(yfov=np.pi / 6.0),
    pyrender.PerspectiveCamera(yfov=np.pi / 7.5)
]

scene.add(mesh, pose=np.eye(4))

scores = {}

# use icosphere as poses
icosphere = trimesh.creation.icosphere(subdivisions=2, radius=Radius)
for idx, v in tqdm(enumerate(icosphere.vertices)):
    # location [v]; vector [-v]
    f = v / np.linalg.norm(v)
    u = np.array([0, 0, 1]) if np.abs(np.dot(f, [0, 0, 1])) < 0.99 else np.array([0, 1, 0])
    r = np.cross(u, -v)
    r = r / np.linalg.norm(r)
    u_ = np.cross(-v, r)
    u_ = u_ / np.linalg.norm(u_)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([-r, u_, f], axis=1)
    pose[:3, 3] = v

    for cam_idx, cam in enumerate(cameras):
        new_scene = deepcopy(scene)
        new_scene.add(cam, pose=pose)

        # render scene
        r = pyrender.OffscreenRenderer(512, 512)
        color, _ = r.render(new_scene, flags=pyrender.constants.RenderFlags.FLAT)

        # convert color to PIL image
        color = Image.fromarray(color.astype(np.uint8))
        imageio.imwrite(f'{args.name}/{idx:03d}_{cam_idx}.png', color)

