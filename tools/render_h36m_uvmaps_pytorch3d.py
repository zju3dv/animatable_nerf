from pytorch3d.renderer.cameras import PerspectiveCameras, OrthographicCameras
from pytorch3d.renderer.mesh.rasterizer import (
    MeshRasterizer,
    RasterizationSettings,
)
import pytorch3d.structures as struct
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import torch


def set_pytorch3d_intrinsic_matrix(K, H, W):
    fx = -K[0, 0] * 2.0 / W
    fy = -K[1, 1] * 2.0 / H
    px = -(K[0, 2] - W / 2.0) * 2.0 / W
    py = -(K[1, 2] - H / 2.0) * 2.0 / H
    K = [
        [fx, 0, px, 0],
        [0, fy, py, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
    K = np.array(K)
    return K


def load_obj(path):
    model = {}
    pts = []
    tex = []
    faces = []

    with open(path) as file:
        while True:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                pts.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                tex.append((float(strs[1]), float(strs[2])))

    uv_faces = []
    with open(path) as file:
        while True:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "f":
                face = (int(strs[1].split("/")[0]) - 1,
                        int(strs[2].split("/")[0]) - 1,
                        int(strs[4].split("/")[0]) - 1)
                texcoord = (int(strs[1].split("/")[1]) - 1,
                            int(strs[2].split("/")[1]) - 1,
                            int(strs[4].split("/")[1]) - 1)
                faces.append(face)
                uv_faces.append(texcoord)
                # for i in range(3):
                #     uv[face[i]] = tex[texcoord[i]]

        uv = []
        for uv_face in uv_faces:
            uv_ = []
            for vertex in uv_face:
                uv_.append(tex[vertex])
            uv.append(uv_)

        model['pts'] = np.array(pts)
        model['faces'] = np.array(faces)
        model['uv'] = np.array(uv)

    return model


obj_path = 'data/h36m/smplx/smpl_uv.obj'
model = load_obj(obj_path)
model['pts'] = model['pts'] * 1000
height, width = (1002, 1000)

data_root = 'data/h36m/S11/Posing'
annots_path = os.path.join(data_root, 'annots.npy')
annots = np.load(annots_path, allow_pickle=True).item()
cameras = annots['cams']
ims = annots['ims']

# caculate the transformation from world to image space
Ks = np.array(cameras['K'])
Rs = np.array(cameras['R'])
Ts = np.array(cameras['T']) / 1000
Ds = np.array(cameras['D'])

for i in range(0, len(ims), 5):
    img_path = ims[i]['ims'][0]
    vertices_path = os.path.join(data_root, 'new_vertices/{}.npy'.format(i))
    vertices = np.load(vertices_path)

    K = Ks[0]
    R = Rs[0]
    T = Ts[0]

    pytorch3d_K = set_pytorch3d_intrinsic_matrix(K, height, width)
    cameras = PerspectiveCameras(device='cuda',
                                 K=pytorch3d_K[None].astype(np.float32),
                                 R=R.T[None].astype(np.float32),
                                 T=T.T.astype(np.float32))
    raster_settings = RasterizationSettings(image_size=(height, width),
                                            blur_radius=0.0,
                                            faces_per_pixel=1,
                                            bin_size=None)
    rasterizer = MeshRasterizer(cameras=cameras,
                                raster_settings=raster_settings)

    vertex = torch.FloatTensor(vertices).cuda()[None]
    triangle = torch.LongTensor(model['faces']).cuda()[None]
    ppose = struct.Meshes(verts=vertex, faces=triangle)
    fragments = rasterizer(ppose)

    face_idx_map = fragments.pix_to_face[0, ..., 0].detach().cpu().numpy()
    bary_coords_map = fragments.bary_coords[0, :, :, 0].detach().cpu().numpy()

    mask = face_idx_map >= 0
    pixel_face_idx = face_idx_map[mask]
    pixel_bary_coord = bary_coords_map[mask]

    pixel_vertex_idx = triangle[0][pixel_face_idx].detach().cpu().numpy()
    pixel_face_uv = model['uv'][pixel_face_idx]
    pixel_uv = np.sum(pixel_face_uv * pixel_bary_coord[..., None], axis=1)

    uv_map = np.zeros((height, width, 3))
    uv_map[mask, :2] = pixel_uv
    uv_map[mask, 2] = 1

    for j in range(len(ims[i]['ims'])):
        frame_ind = int(os.path.basename(ims[i]['ims'][j])[:-4])
        cam_ind = j
        uv_path = os.path.join(data_root, 'uv/frame{:04d}_view{:04d}.png'.format(frame_ind, cam_ind))
        os.system("mkdir -p '{}'".format(os.path.dirname(uv_path)))
        cv2.imwrite(uv_path, uv_map * 255)
