import torch
import torch.nn as nn
import sys
from lib.csrc.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import lib.csrc.pointnet2.pytorch_utils as pt_utils
from .query_group import QueryAndGroup
import torch.nn.functional as F


def get_model(input_channels=0, out_dim=18):
    return Pointnet2MSG(input_channels=input_channels, out_dim=out_dim)


NPOINTS = [4096, 1024, 256, 64]
RADIUS = [[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16]]
NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
MLPS = [[[16, 16], [32, 32]], [[32, 32], [32, 32]], [[64, 64], [64, 64]],
        [[64, 64], [64, 64]]]
FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]
CLS_FC = [128]
DP_RATIO = 0.5


class Pointnet2MSG(nn.Module):
    def __init__(self,
                 input_channels=6,
                 out_dim=32,
                 npoints=[4096, 1024, 256, 64],
                 radius=[[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]):
        super().__init__()

        NPOINTS = npoints
        RADIUS = radius

        FP_MLPS[0] = [out_dim, out_dim]

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(npoint=NPOINTS[k],
                                    radii=RADIUS[k],
                                    nsamples=NSAMPLE[k],
                                    mlps=mlps,
                                    use_xyz=True,
                                    bn=True))
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[
                k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] +
                                 FP_MLPS[k]))
        '''
        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=True))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        cls_layers.insert(1, nn.Dropout(0.5))
        self.cls_layer = nn.Sequential(*cls_layers)
        '''

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(1, 2).contiguous()
                    if pc.size(-1) > 3 else None)

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i],
                                                   l_features[i - 1],
                                                   l_features[i])

        #pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, 1)
        return l_features[0]  # only output features
