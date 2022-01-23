import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import nt_renderer
from lib.train import make_optimizer
from lib.losses.nhr_perceptual_loss import Perceptual_loss


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = nt_renderer.Renderer(self.net)

        # self.loss = torch.nn.L1Loss(reduction='mean')
        self.loss = Perceptual_loss()

    def forward(self, batch):
        ret = self.renderer.render(batch)

        scalar_stats = {}
        loss = 0

        msk = batch['msk'][:, None].float()
        rgb = ret['rgb'] * msk
        img = batch['img'] * msk
        x = torch.cat((rgb, ret['mask']), dim=1)
        target = torch.cat((img, msk), dim=1)
        loss1, loss2, loss3 = self.loss(x, target)
        loss = loss + loss1 + loss2

        # rgb = ret['rgb']
        # img = batch['img']
        # loss = self.loss(rgb, img)

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
