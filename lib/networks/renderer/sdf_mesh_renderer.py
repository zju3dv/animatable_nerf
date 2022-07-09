import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import numpy as np
import mcubes
import trimesh
from lib.utils.blend_utils import *
from lib.utils import sample_utils


class Renderer:
    def __init__(self, net):
        self.net = net

    def batchify_rays(self, wpts, sdf_decoder, net, chunk, batch):
        """Render rays in smaller minibatches to avoid OOM.
        """
        n_point = wpts.shape[0]
        all_ret = []
        for i in range(0, n_point, chunk):
            # resd = net.calculate_residual_deformation(wpts[i:i + chunk][None],
            #                                           batch['latent_index'])
            # wpts[i:i + chunk] = wpts[i:i + chunk] + resd[0]
            ret = sdf_decoder(wpts[i:i + chunk])[:, :1]
            all_ret.append(ret.detach().cpu().numpy())
        all_ret = np.concatenate(all_ret, 0)
        return all_ret

    def batchify_blend_weights(self, pts, bw_input, chunk=1024 * 32):
        all_ret = []
        for i in range(0, pts.shape[1], chunk):
            ret = self.net.calculate_bigpose_smpl_bw(pts[:, i:i + chunk],
                                                     bw_input)
            all_ret.append(ret)
        all_ret = torch.cat(all_ret, 2)
        return all_ret

    def batchify_normal_sdf(self, pts, batch, chunk=1024 * 32):
        all_normal = []
        all_sdf = []
        for i in range(0, pts.shape[1], chunk):
            normal, sdf = self.net.gradient_of_deformed_sdf(
                pts[:, i:i + chunk], batch)
            all_normal.append(normal.detach().cpu().numpy())
            all_sdf.append(sdf.detach().cpu().numpy())
        all_normal = np.concatenate(all_normal, axis=1)
        all_sdf = np.concatenate(all_sdf, axis=1)
        return all_normal, all_sdf

    def render(self, batch):
        pts = batch['pts']
        sh = pts.shape

        pts = pts.view(sh[0], -1, 3)

        tbw, tnorm = sample_utils.sample_blend_closest_points(pts, batch['tvertices'], batch['weights'])
        tnorm = tnorm[..., 0]
        norm_th = 0.1
        inside = tnorm < norm_th

        # # sampling points for blend weight training
        # if 'tbw' in batch:
        #     bw = pts_sample_blend_weights(pts, batch['tbw'], batch['tbounds'])
        #     tnorm = bw[:, 24]
        #     inside = tnorm < cfg.norm_th
        # else:
        #     inside = torch.ones(pts.shape[:2]).bool()

        pts = pts[inside]

        sdf_decoder = lambda x: self.net.tpose_human.sdf_network(x, batch)
        # sdf_decoder = lambda x: self.net.get_sdf(x, batch)

        sdf = self.batchify_rays(pts, sdf_decoder, self.net, 2048 * 64, batch)

        inside = inside.detach().cpu().numpy()
        full_sdf = 10 * np.ones(inside.shape)
        full_sdf[inside] = sdf[:, 0]
        sdf = -full_sdf

        cube = sdf.reshape(*sh[1:-1])
        cube = np.pad(cube, 10, mode='constant', constant_values=-10)
        vertices, triangles = mcubes.marching_cubes(cube, 0)
        mesh = trimesh.Trimesh(vertices, triangles)
        vertices = (vertices - 10) * cfg.voxel_size[0]
        vertices = vertices + batch['tbounds'][0, 0].detach().cpu().numpy()

        labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        triangles = triangles[labels == 0]
        import open3d as o3d
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(triangles)
        mesh_o3d.remove_unreferenced_vertices()
        vertices = np.array(mesh_o3d.vertices)
        triangles = np.array(mesh_o3d.triangles)

        # transform vertices to the world space
        pts = torch.from_numpy(vertices).to(pts)[None]
        tbw, _ = sample_utils.sample_blend_closest_points(pts, batch['tvertices'], batch['weights'])
        tbw = tbw.permute(0, 2, 1)

        if 'gradient_of_deformed_sdf' in dir(self.net):
            normal, sdf = self.batchify_normal_sdf(pts, batch)
            resd = -normal * sdf
            resd = torch.from_numpy(resd).to(pts)
            deformed_pts = pts + resd
        else:
            deformed_pts = pts

        tpose_pts = pose_points_to_tpose_points(deformed_pts, tbw,
                                                batch['big_A'])
        pose_pts = tpose_points_to_pose_points(tpose_pts, tbw, batch['A'])
        pose_pts = pose_points_to_world_points(pose_pts, batch['R'],
                                               batch['Th'])
        # posed_vertices = deformed_pts[0].detach().cpu().numpy()
        posed_vertices = pose_pts[0].detach().cpu().numpy()

        ret = {
            # 'vertex': vertices,
            'posed_vertex': posed_vertices,
            'triangle': triangles,
            # 'rgb': rgb,
        }

        return ret
