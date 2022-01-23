import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple

import torch
import sys
from .vgg import vgg16
from mmcv.runner import load_checkpoint

default_layer_name = {
            'conv1_2_relu': 1,
            'conv2_2_relu': 1,
            'conv3_2_relu': 1,
            'conv4_2_relu': 1,
            'conv5_2_relu': 1,
        }

LossDict = {
    'MSELoss': 'l2_loss',
    'SSIMLoss': 'ssim_loss'
}

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self,
                 pretrained = './data/celeba/vgg16.pytorch.pth',
                 layer_name = default_layer_name,
                 ):
        super(VGGPerceptualLoss, self).__init__()

        if pretrained is not None:
            self.vgg_layers = vgg16().features
            print(f'initial vgg with {pretrained}') 
            self.init_weights(pretrained=pretrained)
        else:
            self.vgg_layers = vgg16(pretrained=True).features()
        self.layer_name = layer_name
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def init_weights(self, pretrained):
        if pretrained is not None:
            ck = load_checkpoint(self.vgg_layers, pretrained, map_location='cpu')
        else:
            raise NotImplementedError 

    def exp_running_avg(self, x, x_avg, rho=0.99):
        w_update = 1.0 - rho
        x_new = x_avg + w_update*(x-x_avg)
        return x_new

    def normalize_img(self, img):
        img = img.mean(dim=1, keepdim=True)
        img = (img - 114.451)/255
        return img

    def get_feat(self, x, is_gram=False, layer_name=None):
        def gram(f, is_gram):
            if is_gram:
                bs, c, h, w = f.shape
                f = f.reshape(bs, c, -1)
                f_T = f.permute(0, 2, 1)
                G = torch.bmm(f, f_T) / (c*h*w)
                # print('style')
            else:
                # print('not style')
                G = f
            return G
        net_list = {}
        norm_x = self.normalize_img(x)
        if layer_name is None:
            layer_name = default_layer_name
        # print(layer_name)
        for name, module in self.vgg_layers._modules.items():
            norm_x = module(norm_x)
            if name in layer_name:
                if name == 'input':
                    net_list[name] = gram(x, is_gram)
                else:
                    net_list[name] = gram(norm_x, is_gram)
        return net_list

    def forward(self, pred_img, gt_img, mask=None, is_gram=False, layer_name=None):
        layer_name = layer_name if layer_name is not None else self.layer_name
        assert len(layer_name) == len(default_layer_name)
        new_layer_name = {}
        for idx, x in enumerate(default_layer_name):
            new_layer_name[x] = layer_name[idx]
        gt_img_feat_list = self.get_feat(gt_img, is_gram, layer_name=new_layer_name)
        pred_img_feat_list = self.get_feat(pred_img, is_gram, layer_name=new_layer_name)

        loss_list = []

        if is_gram:
            assert mask is not None,'mask must be none when gram is True'
            mask = None
        for idx, key in enumerate(gt_img_feat_list.keys()):
            gt_img_feat = gt_img_feat_list[key]
            pred_img_feat = pred_img_feat_list[key]
            if mask is not None:
                resize_mask = F.interpolate(mask, gt_img_feat.shape[-2:], mode='bilinear')
                # loss = (((gt_img_feat-pred_img_feat)**2)*resize_mask).mean()
                loss = (F.mse_loss(pred_img_feat, gt_img_feat, reduction='none') * resize_mask).mean()
            else:
                loss = (F.mse_loss(pred_img_feat, gt_img_feat, reduction='none')).mean()
            loss_list.append(loss*new_layer_name[key])
        return loss_list

if __name__ == '__main__':
    perceptual_loss = VGGPerceptualLoss(
                          pretrained = sys.argv[1])
    loss_func = perceptual_loss.cuda()
    pred_img = torch.randn(1,3,32,32).cuda()
    gt_img = torch.randn(1,3,32,32).cuda()
    loss = loss_func(pred_img, gt_img)
    print(loss)
