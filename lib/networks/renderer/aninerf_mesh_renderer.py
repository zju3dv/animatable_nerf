import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import numpy as np
import mcubes
import trimesh
from lib.utils.blend_utils import *


class Renderer:
    def __init__(self, net):
        self.net = net

    def batchify_rays(self, wpts, alpha_decoder, net, chunk, batch):
        """Render rays in smaller minibatches to avoid OOM.
        """
        n_point = wpts.shape[0]
        all_ret = []
        for i in range(0, n_point, chunk):
            ret = alpha_decoder(wpts[i:i + chunk])
            all_ret.append(ret.detach().cpu().numpy())
        all_ret = np.concatenate(all_ret, 0)
        return all_ret

    def render(self, batch):
        pts = batch['pts']
        sh = pts.shape

        inside = batch['inside'][0].bool()
        pts = pts[0][inside]

        alpha_decoder = lambda x: self.net.get_alpha(x, batch)

        alpha = self.batchify_rays(pts, alpha_decoder, self.net, 2048 * 64, batch)

        cube = np.zeros(sh[1:-1])
        inside = inside.detach().cpu().numpy()
        cube[inside == 1] = alpha

        cube = np.pad(cube, 10, mode='constant')
        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)
        vertices = (vertices - 10) * cfg.voxel_size[0]
        vertices = vertices + batch['wbounds'][0, 0].detach().cpu().numpy()

        # mesh = trimesh.Trimesh(vertices, triangles)
        # labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        # triangles = triangles[labels == 0]
        # import open3d as o3d
        # mesh_o3d = o3d.geometry.TriangleMesh()
        # mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        # mesh_o3d.triangles = o3d.utility.Vector3iVector(triangles)
        # mesh_o3d.remove_unreferenced_vertices()
        # vertices = np.array(mesh_o3d.vertices)
        # triangles = np.array(mesh_o3d.triangles)

        ret = {
            'posed_vertex': vertices,
            'triangle': triangles,
            # 'rgb': rgb,
        }

        return ret
