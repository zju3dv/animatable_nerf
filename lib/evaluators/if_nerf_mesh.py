import trimesh
import trimesh.proximity
import trimesh.sample
import numpy as np
import math
import os
from PIL import Image


class Evaluator:
    def __init__(self) -> None:
        self.p2ss = []
        self.chamfers = []
        self.mesh_eval = MeshEvaluator()

    def evaluate(self, output, batch):
        src_mesh = output["mesh"]
        self.mesh_eval.set_src_mesh(src_mesh)
        self.mesh_eval.set_tgt_mesh(batch["obj"][0])
        chamfer = self.mesh_eval.get_chamfer_dist()
        p2s = self.mesh_eval.get_surface_dist()
        self.chamfers.append(chamfer)
        self.p2ss.append(p2s)
        print("Chamfer: {}, P2S: {}".format(chamfer, p2s))

    def summarize(self):
        self.p2ss = np.array(self.p2ss)
        self.chamfers = np.array(self.chamfers)
        np.save("p2s.npy", self.p2ss)
        np.save("chamfer.npy", self.chamfers)
        pass


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

    def set_tgt_mesh(self, tgt_path: str):
        self.tgt_mesh = trimesh.load(tgt_path)

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
