import torch
import torch.nn.functional as F
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from lib.utils.blend_utils import *
from . import tpose_renderer


class Renderer(tpose_renderer.Renderer):
    def __init__(self, net):
        super(Renderer, self).__init__(net)

    def prepare_inside_pts(self, pts, batch):
        if 'Ks' not in batch:
            __import__('ipdb').set_trace()
            return raw

        sh = pts.shape
        pts = pts.view(sh[0], -1, sh[3])

        insides = []
        for nv in range(batch['Ks'].size(1)):
            # project pts to image space
            R = batch['RT'][:, nv, :3, :3]
            T = batch['RT'][:, nv, :3, 3]
            pts_ = torch.matmul(pts, R.transpose(2, 1)) + T[:, None]
            pts_ = torch.matmul(pts_, batch['Ks'][:, nv].transpose(2, 1))
            pts2d = pts_[..., :2] / pts_[..., 2:]

            # ensure that pts2d is inside the image
            pts2d = pts2d.round().long()
            H, W = batch['H'].item(), batch['W'].item()
            pts2d[..., 0] = torch.clamp(pts2d[..., 0], 0, W - 1)
            pts2d[..., 1] = torch.clamp(pts2d[..., 1], 0, H - 1)

            # remove the points outside the mask
            pts2d = pts2d[0]
            msk = batch['msks'][0, nv]
            inside = msk[pts2d[:, 1], pts2d[:, 0]][None].bool()
            insides.append(inside)

        inside = insides[0]
        for i in range(1, len(insides)):
            inside = inside * insides[i]

        return inside

    def get_density_color(self, wpts, viewdir, z_vals, inside, raw_decoder):
        """
        wpts: n_batch, n_pixel, n_sample, 3
        viewdir: n_batch, n_pixel, 3
        z_vals: n_batch, n_pixel, n_sample
        """
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch * n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch * n_pixel * n_sample, -1)

        # calculate dists for the opacity computation
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=2)
        dists = dists.view(n_batch * n_pixel * n_sample)

        full_raw = torch.zeros([n_batch * n_pixel * n_sample, 4]).to(wpts)
        if inside.sum() == 0:
            ret = {'raw': full_raw}
            return ret

        inside = inside.view(n_batch * n_pixel * n_sample)
        wpts = wpts[inside]
        viewdir = viewdir[inside]
        dists = dists[inside]
        ret = raw_decoder(wpts, viewdir, dists)

        full_raw[inside] = ret['raw']
        ret = {'raw': full_raw}

        return ret

    def get_pixel_value(self, ray_o, ray_d, near, far, batch):
        n_batch = ray_o.shape[0]

        # sampling points for nerf training
        wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, near, far)
        inside = self.prepare_inside_pts(wpts, batch)

        # viewing direction, ray_d has been normalized in the dataset
        viewdir = ray_d

        raw_decoder = lambda wpts_val, viewdir_val, dists_val: self.net(
            wpts_val, viewdir_val, dists_val, batch)

        # compute the color and density
        ret = self.get_density_color(wpts, viewdir, z_vals, inside,
                                     raw_decoder)

        # reshape to [num_rays, num_samples along ray, 4]
        n_batch, n_pixel, n_sample = z_vals.shape
        raw = ret['raw'].reshape(-1, n_sample, 4)
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, cfg.white_bkgd)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)

        ret = {
            'rgb_map': rgb_map.detach().cpu(),
            'acc_map': acc_map.detach().cpu(),
            'depth_map': depth_map.detach().cpu()
        }

        return ret

    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        sh = ray_o.shape
        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 2048
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk, batch)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret
