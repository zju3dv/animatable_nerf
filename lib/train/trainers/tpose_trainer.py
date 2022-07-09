import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import make_renderer
from lib.networks.renderer import tpose_renderer
from lib.train import make_optimizer
from lib.utils.if_nerf import if_nerf_net_utils
from . import crit


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = make_renderer(cfg, self.net)

        self.bw_crit = torch.nn.functional.smooth_l1_loss
        self.img2mse = lambda x, y: torch.mean((x - y)**2)

    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = {}
        loss = 0

        if 'resd' in ret:
            offset_loss = torch.norm(ret['resd'], dim=2).mean()
            scalar_stats.update({'offset_loss': offset_loss})
            loss += 0.01 * offset_loss

        if 'gradients' in ret:
            gradients = ret['gradients']
            grad_loss = (torch.norm(gradients, dim=2) - 1.0)**2
            grad_loss = grad_loss.mean()
            scalar_stats.update({'grad_loss': grad_loss})
            loss += 0.01 * grad_loss

        if 'observed_gradients' in ret:
            ogradients = ret['observed_gradients']
            ograd_loss = (torch.norm(ogradients, dim=2) - 1.0)**2
            ograd_loss = ograd_loss.mean()
            scalar_stats.update({'ograd_loss': ograd_loss})
            loss += 0.01 * ograd_loss

        if 'pred_pbw' in ret:
            bw_loss = (ret['pred_pbw'] - ret['smpl_tbw']).pow(2).mean()
            scalar_stats.update({'tbw_loss': bw_loss})
            loss += bw_loss

        if 'pbw' in ret:
            bw_loss = self.bw_crit(ret['pbw'], ret['tbw'])
            scalar_stats.update({'bw_loss': bw_loss})
            loss += bw_loss

        if 'msk_sdf' in ret:
            mask_loss = crit.sdf_mask_crit(ret, batch)
            scalar_stats.update({'mask_loss': mask_loss})
            loss += mask_loss

        mask = batch['mask_at_box']
        img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
        scalar_stats.update({'img_loss': img_loss})
        loss += img_loss

        if 'rgb0' in ret:
            img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
            scalar_stats.update({'img_loss0': img_loss0})
            loss += img_loss0

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
