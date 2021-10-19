import trimesh
import trimesh.proximity
import trimesh.sample
import numpy as np
import math
import os
from PIL import Image
from lib.config import cfg
import open3d as o3d
from termcolor import colored


class Evaluator:
    def __init__(self) -> None:
        self.p2ss = []
        self.chamfers = []
        self.mesh_eval = MeshEvaluator()

    def evaluate(self, output, batch):
        human = cfg.test_dataset.human
        vertices = output['posed_vertex']
        if 'rp' in human:
            new_vertices = np.zeros_like(vertices)
            new_vertices[:, 0] = vertices[:, 0]
            new_vertices[:, 1] = vertices[:, 2]
            new_vertices[:, 2] = -vertices[:, 1]
            vertices = new_vertices
        src_mesh = trimesh.Trimesh(vertices, output['triangle'], process=False)

        data_root = cfg.test_dataset.data_root
        frame_index = batch['frame_index'].item()
        tgt_mesh_path = os.path.join(data_root,
                                     'object/{:06d}.obj'.format(frame_index))
        tgt_mesh = o3d.io.read_triangle_mesh(tgt_mesh_path)
        tgt_mesh = trimesh.Trimesh(tgt_mesh.vertices,
                                   tgt_mesh.triangles,
                                   process=False)

        self.mesh_eval.set_src_mesh(src_mesh)
        self.mesh_eval.set_tgt_mesh(tgt_mesh)
        chamfer = self.mesh_eval.get_chamfer_dist()
        p2s = self.mesh_eval.get_surface_dist()
        self.chamfers.append(chamfer)
        self.p2ss.append(p2s)

    def summarize(self):
        result_dir = cfg.result_dir
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))
        result_path = os.path.join(cfg.result_dir, 'mesh_metrics.npy')
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        metrics = {'p2s': self.p2ss, 'chamfer': self.chamfers}
        np.save(result_path, metrics)

        print('p2s: {}'.format(np.mean(self.p2ss)))
        print('chamfer: {}'.format(np.mean(self.chamfers)))

        self.p2ss = []
        self.chamfers = []


class MeshEvaluator:
    """
    From https://github.com/facebookresearch/pifuhd/blob/master/lib/evaluator.py
    """
    _normal_render = None

    def __init__(self, scale_factor=1.0, offset=0):
        self.scale_factor = scale_factor
        self.offset = offset
        pass

    def set_mesh(self, src_path, tgt_path):
        self.src_mesh = trimesh.load(src_path)
        self.tgt_mesh = trimesh.load(tgt_path)

    def apply_registration(self):
        transform, _ = trimesh.registration.mesh_other(self.src_mesh,
                                                       self.tgt_mesh)
        self.src_mesh.apply_transform(transform)

    def set_src_mesh(self, mesh):
        self.src_mesh = mesh

    def set_tgt_mesh(self, mesh):
        self.tgt_mesh = mesh

    def get_chamfer_dist(self, num_samples=1000):
        # breakpoint()
        # Chamfer
        src_surf_pts, _ = trimesh.sample.sample_surface(
            self.src_mesh, num_samples)
        # self.src_mesh.show()
        tgt_surf_pts, _ = trimesh.sample.sample_surface(
            self.tgt_mesh, num_samples)

        _, src_tgt_dist, _ = trimesh.proximity.closest_point(
            self.tgt_mesh, src_surf_pts)
        _, tgt_src_dist, _ = trimesh.proximity.closest_point(
            self.src_mesh, tgt_surf_pts)

        src_tgt_dist[np.isnan(src_tgt_dist)] = 0
        tgt_src_dist[np.isnan(tgt_src_dist)] = 0

        src_tgt_dist = src_tgt_dist.mean()
        tgt_src_dist = tgt_src_dist.mean()

        chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2

        return chamfer_dist

    def get_surface_dist(self, num_samples=10000):
        # P2S
        src_surf_pts, _ = trimesh.sample.sample_surface(
            self.src_mesh, num_samples)

        _, src_tgt_dist, _ = trimesh.proximity.closest_point(
            self.tgt_mesh, src_surf_pts)

        src_tgt_dist[np.isnan(src_tgt_dist)] = 0

        src_tgt_dist = src_tgt_dist.mean()

        return src_tgt_dist
