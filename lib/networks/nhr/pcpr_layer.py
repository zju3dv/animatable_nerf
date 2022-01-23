import math
import torch
import torch.nn as nn
import pcpr
'''
point_clouds: (num_points_1 + num_points_1 + ... + num_points_[batch_size], 3) [float]-GPU
point_features (k, num_points_1 + num_points_1 + ... + num_points_[batch_size]) [float]-GPU
default_features (k, 1)             [float]-GPU
cam_intrinsic: (batch,3,3)          [float]-GPU
cam_extrinsic: (batch,4,4)          [float]-GPU
near_far_max_splatting_size: (batch,3)    [float]-CPU
num_points: (batch)              [int]-CPU
tar_image_size: (2)              [int]-CPU [tar_width,tar_heigh]
'''


class PCPRFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, point_features, default_features, point_clouds,
                cam_intrinsic, cam_extrinsic, near_far_max_splatting_size,
                num_points, tar_image_size):

        batch_size = cam_intrinsic.size(0)
        dim_features = point_features.size(0)

        if cam_extrinsic.size(0) != batch_size or near_far_max_splatting_size.size(0)!=batch_size or\
            num_points.size(0)!= batch_size:
            raise Exception('[PCPR] batch_sizes are not consistant.')

        _cam_extrinsic = torch.cat([
            cam_extrinsic[:, 0:3, 2], cam_extrinsic[:, 0:3, 0],
            cam_extrinsic[:, 0:3, 1], cam_extrinsic[:, 0:3, 3]
        ],
                                   dim=1)

        tar_width, tar_heigh = int(tar_image_size[0].item()), int(
            tar_image_size[1].item())

        out_depth = torch.zeros(batch_size, 1, tar_heigh, tar_width).cuda()
        out_index = torch.zeros(batch_size,
                                tar_heigh,
                                tar_width,
                                dtype=torch.int32).cuda()
        out_feature = torch.zeros(batch_size, dim_features, tar_heigh,
                                  tar_width).cuda()

        _num_points = num_points.int().tolist()

        beg = 0

        point_clouds = point_clouds.view(-1, 3)
        for i in range(batch_size):

            #print('Start Kernel.',flush = True)
            out_depth[i][0], out_index[i] = pcpr.forward(
                point_clouds[beg:beg + _num_points[i], :], cam_intrinsic[i],
                _cam_extrinsic[i], out_depth[i][0], out_index[i],
                *(near_far_max_splatting_size[i].tolist()))
            #print('End Kernel.',flush = True)

            features = point_features[:, beg:beg + _num_points[i]].detach()

            features = torch.cat([features, default_features.detach()], dim=1)

            out_index[i] = out_index[i] - 1
            out_index[i][out_index[i] < 0] = _num_points[i]
            tmp_index = out_index[i].long()

            out_feature[i] = features[:, tmp_index].detach()

            beg = beg + _num_points[i]

        out_index = out_index.int()

        ctx.save_for_backward(out_index, out_feature, point_features,
                              default_features, num_points.int())

        return out_feature, out_depth

    @staticmethod
    def backward(ctx, grad_feature, grad_depth=None):
        out_index, out_feature, point_features, default_features, num_points = ctx.saved_tensors

        grad_feature = grad_feature.contiguous()
        #out_feature.backward(grad_feature)

        d_point_features = torch.ones_like(point_features).float()
        d_default_features = torch.ones_like(default_features)

        total_sum = torch.sum(num_points)

        flag = False
        if point_features.type() == 'torch.HalfTensor' or point_features.type(
        ) == 'torch.cuda.HalfTensor':
            flag = True
            #grad_feature = grad_feature.float()
        #print(grad_feature.type(),d_default_features.type(), d_point_features.type())

        d_point_features, d_default_features = pcpr.backward(
            grad_feature, out_index, num_points.cuda(), d_point_features,
            d_default_features, total_sum)

        #d_point_features = point_features.grad
        #d_default_features = default_features.grad
        if flag:
            d_point_features = d_point_features.half()
            #d_default_features = d_default_features.half()

        return d_point_features, d_default_features, None, None, None, None, None, None


class PCPRModel(nn.Module):
    def __init__(self, tar_width, tar_heigh):
        super(PCPRModel, self).__init__()
        self.tar_image_size = torch.Tensor([tar_width, tar_heigh]).int()

    def forward(self, point_features, default_features, point_clouds,
                cam_intrinsic, cam_extrinsic, near_far_max_splatting_size,
                num_points, tar_height, tar_width):
        tar_image_size = torch.Tensor([tar_width, tar_height]).int()

        return PCPRFunction.apply(point_features, default_features,
                                  point_clouds, cam_intrinsic, cam_extrinsic,
                                  near_far_max_splatting_size, num_points,
                                  tar_image_size)
