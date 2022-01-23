import torch.nn as nn
from lib.networks.pointnet2.pointnet2_msg import get_model
from .pcpr_parameters import PCPRParameters
from .pcprender import PCPRender
import torch
from lib.config import cfg
from lib.utils.blend_utils import *


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.pcpr_parameters = PCPRParameters(feature_dim=18)
        self.pointnet = get_model(0, 18)
        self.render = PCPRender(feature_dim=18,
                                tar_width=int(cfg.W * cfg.ratio),
                                tar_height=int(cfg.H * cfg.ratio),
                                use_mask=True,
                                use_dir_in_world=True,
                                use_depth=False)

        tpose_dir = 'data/smpl_faces.npy'
        tpose_faces = np.load(tpose_dir, allow_pickle=True)
        self.tface = torch.from_numpy(tpose_faces).cuda().float()

    def forward(self, batch):
        self.tvertex = batch['tpose'][0]
        self.bw = pts_sample_blend_weights(self.tvertex, batch['tbw'],
                                           batch['tbounds'])
        self.bw = self.bw[:, :24]
        ppose = pose_points_to_tpose_points(self.tvertex[None], self.bw,
                                            batch['big_A'])
        pvertex_i = tpose_points_to_pose_points(ppose, self.bw, batch['A'])
        pts = pvertex_i
        vertex = pose_points_to_world_points(pvertex_i, batch['R'],
                                             batch['Th'])
        in_points = vertex
        _, default_features = self.pcpr_parameters()
        point_features = self.pointnet(pts)
        point_features = torch.cat(
            [point_features[i] for i in range(len(point_features))], dim=1)

        K = batch['K']
        T = batch['RT']
        near_far_max_splatting_size = torch.tensor(
            [[1.0, 8.5, 1.5] for _ in range(len(in_points))]).to(pts)
        num_points = torch.tensor(
            [in_points.size(1) for _ in range(len(in_points))]).to(pts)
        torch.cuda.synchronize()
        import time
        now = time.time()
        res, depth, features, dir_in_world, rgb = self.render(
            point_features, default_features, in_points, K, T,
            near_far_max_splatting_size, num_points, batch)
        torch.cuda.synchronize()
        time = time.time() - now
        return res, time
