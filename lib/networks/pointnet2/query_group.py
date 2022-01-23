import torch.nn as nn
from lib.csrc.pointnet2 import pointnet2_utils
import torch


class QueryAndGroup(nn.Module):
    def __init__(self, radius, nsample, use_xyz):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        # create an empty term at 0th slot to handle empty neighbors
        xyz_at_0 = torch.ones_like(xyz[:, :1, :]) * 100
        xyz = torch.cat([xyz_at_0, xyz], dim=1)
        feature_at_0 = torch.zeros_like(features[:, :, :1])
        features = torch.cat([feature_at_0, features], dim=2)

        idx = pointnet2_utils.ball_query(self.radius, self.nsample, xyz,
                                         new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = pointnet2_utils.grouping_operation(
            xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = pointnet2_utils.grouping_operation(
                features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        new_features = new_features.permute(0, 2, 3, 1)
        new_features[idx == 0] = 0
        new_features = new_features.permute(0, 3, 1, 2)

        return new_features
