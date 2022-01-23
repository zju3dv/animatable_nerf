import torch
from .unet_model import UNet
from .pcpr_layer import PCPRModel
import pcpr
import numpy as np
from .instant_norm import instant_norm_2d


class PCPRender(torch.nn.Module):
    def __init__(self,
                 feature_dim,
                 tar_width,
                 tar_height,
                 use_mask=False,
                 use_dir_in_world=True,
                 use_depth=False):
        super(PCPRender, self).__init__()
        self.feature_dim = feature_dim
        self.tar_width = tar_width
        self.tar_height = tar_height
        self.use_mask = use_mask
        self.use_depth = use_depth
        self.use_rgb_map = False

        self.use_dir_in_world = use_dir_in_world

        add_dir_input = 0
        if use_dir_in_world:
            add_dir_input = 3

        add_rgb_input = 0

        if self.use_rgb_map:
            add_rgb_input = 3

        add_depth_input = 0
        if use_depth:
            add_depth_input = 1
            self.depth_norm = instant_norm_2d

        self.pcpr_layer = PCPRModel(tar_width, tar_height)
        self.unet = UNet(
            feature_dim + add_dir_input + add_rgb_input +
            add_depth_input,  # input channel: feature[feature_dim] + depth[1] + viewin directions[3] + %%%points color[3]%%%(no used for now)
            3,
            [32, 64, 180, 450, 450, 180, 64, 32, 26],
            self.use_mask,
            no_modified=False)  # output channel: 3 RGB
        self.unet = self.unet.cuda()

        # generate meshgrid
        xh, yw = torch.meshgrid(
            [torch.arange(0, tar_height),
             torch.arange(0, tar_width)])
        self.coord_meshgrid = torch.stack([yw, xh, torch.ones_like(xh)],
                                          dim=0).float()
        self.coord_meshgrid = self.coord_meshgrid.view(1, 3, -1)
        self.coord_meshgrid = self.coord_meshgrid.cuda()

    def forward(self, point_features, default_features, point_clouds,
                cam_intrinsic, cam_extrinsic, near_far_max_splatting_size,
                num_points, batch):

        batch_num = cam_intrinsic.size(0)

        # out_feature (batch, feature_dim, tar_height, tar_width )
        # out_depth (batch, 1, tar_height, tar_width )
        tar_height, tar_width = batch['img'].shape[2:]
        out_feature, out_depth = self.pcpr_layer(
            point_features, default_features, point_clouds, cam_intrinsic,
            cam_extrinsic, near_far_max_splatting_size, num_points, tar_height, tar_width)

        #assert not (out_depth!=0).any(), print(point_clouds)

        if self.use_depth:
            out_depth_feature = self.depth_norm(out_depth)

        # generate viewin directions
        Kinv = torch.inverse(cam_intrinsic)
        xh, yw = torch.meshgrid(
            [torch.arange(0, tar_height),
             torch.arange(0, tar_width)])
        coord_meshgrid = torch.stack([yw, xh, torch.ones_like(xh)],
                                          dim=0).float()
        coord_meshgrid = coord_meshgrid.view(1, 3, -1)
        coord_meshgrid = coord_meshgrid.to(out_feature)

        coord_meshgrids = coord_meshgrid.repeat(batch_num, 1, 1)
        dir_in_camera = torch.bmm(Kinv, coord_meshgrids)
        dir_in_camera = torch.cat([
            dir_in_camera,
            torch.ones(batch_num, 1, dir_in_camera.size(2)).cuda()
        ],
                                  dim=1)
        dir_in_world = torch.bmm(cam_extrinsic, dir_in_camera)
        dir_in_world = dir_in_world / dir_in_world[:, 3:4, :].repeat(1, 4, 1)
        dir_in_world = dir_in_world[:, 0:3, :]
        dir_in_world = torch.nn.functional.normalize(dir_in_world, dim=1)
        dir_in_world = dir_in_world.reshape(batch_num, 3, tar_height,
                                            tar_width)

        #set direction to zeros for depth==0
        depth_mask = out_depth.repeat(1, 3, 1, 1)
        dir_in_world[depth_mask == 0] = 0

        # fuse all features

        fused_features = out_feature

        if self.use_dir_in_world:
            fused_features = torch.cat([out_feature, dir_in_world], dim=1)

        if self.use_depth:
            fused_features = torch.cat([fused_features, out_depth_feature],
                                       dim=1)

        # rendering
        #assert not (out_depth_feature!=out_depth_feature).any(), print(out_depth_feature)
        x = self.unet(fused_features)
        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('TKAgg')
        # plt.subplot(1,2,1)
        # plt.imshow(fused_features.squeeze().permute(1,2,0).detach().cpu()[:,:,:3])
        # plt.subplot(1,2,2)
        # plt.imshow(batch['img'].squeeze().permute(1,2,0).detach().cpu())
        # plt.show()
        return x, out_depth.detach(), out_feature.detach(
        ), dir_in_world.detach(), None
