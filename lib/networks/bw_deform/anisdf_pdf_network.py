import torch.nn as nn
#import spconv
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.utils.blend_utils import *
from .. import embedder
from lib.utils import net_utils
import os
from lib.utils import sample_utils


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.tpose_human = TPoseHuman()

        self.resd_latent = nn.Embedding(cfg.num_latent_code, 128)

        self.actvn = nn.ReLU()

        input_ch = 135
        D = 8
        W = 256
        self.skips = [4]
        self.resd_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.resd_fc = nn.Conv1d(W, 3, 1)
        self.resd_fc.bias.data.fill_(0)

        if cfg.get('init_sdf', False):
            init_path = os.path.join('data/trained_model', cfg.task,
                                     cfg.init_sdf)
            net_utils.load_network(self,
                                   init_path,
                                   only=['tpose_human.sdf_network'])

    def get_point_feature(self, pts, ind, latents):
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        latent = latents(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def calculate_residual_deformation(self, tpose, batch):
        pts = embedder.xyz_embedder(tpose)
        pts = pts.transpose(1, 2)
        latent = batch['poses']
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)

        net = features
        for i, l in enumerate(self.resd_linears):
            net = self.actvn(self.resd_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        resd = self.resd_fc(net)
        resd = resd.transpose(1, 2)
        resd = 0.05 * torch.tanh(resd)
        return resd

    def pose_points_to_tpose_points(self, pose_pts, pose_dirs, batch):
        """
        pose_pts: n_batch, n_point, 3
        """
        # initial blend weights of points at i
        pbw, _ = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
        pbw = pbw.permute(0, 2, 1)

        # transform points from i to i_0
        init_tpose = pose_points_to_tpose_points(pose_pts, pbw,
                                                 batch['A'])
        init_bigpose = tpose_points_to_pose_points(init_tpose, pbw,
                                                   batch['big_A'])
        resd = self.calculate_residual_deformation(init_bigpose, batch)
        tpose = init_bigpose + resd

        if cfg.tpose_viewdir and pose_dirs is not None:
            init_tdirs = pose_dirs_to_tpose_dirs(pose_dirs, pbw,
                                                 batch['A'])
            tpose_dirs = tpose_dirs_to_pose_dirs(init_tdirs, pbw,
                                                 batch['big_A'])
        else:
            tpose_dirs = None

        return tpose, tpose_dirs, init_bigpose, resd

    def calculate_bigpose_smpl_bw(self, bigpose, input_bw):
        smpl_bw = pts_sample_blend_weights(bigpose, input_bw['tbw'],
                                           input_bw['tbounds'])
        return smpl_bw

    def calculate_wpts_sdf(self, wpts, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])

        # transform points from the pose space to the tpose space
        tpose, tpose_dirs, init_bigpose, resd = self.pose_points_to_tpose_points(
            pose_pts, None, batch)
        tpose = tpose[0]
        sdf = self.tpose_human.sdf_network(tpose, batch)[:, :1]

        return sdf

    def wpts_gradient(self, wpts, batch):
        wpts.requires_grad_(True)
        with torch.enable_grad():
            sdf = self.calculate_wpts_sdf(wpts, batch)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf,
                                        inputs=wpts,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients

    def gradient_of_deformed_sdf(self, x, batch):
        x.requires_grad_(True)
        with torch.enable_grad():
            resd = self.calculate_residual_deformation(x, batch)
            tpose = x + resd
            tpose = tpose[0]
            y = self.tpose_human.sdf_network(tpose, batch)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients, y[None]

    def forward(self, wpts, viewdir, dists, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])
        viewdir = viewdir[None]
        pose_dirs = world_dirs_to_pose_dirs(viewdir, batch['R'])

        with torch.no_grad():
            pbw, pnorm = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
            pnorm = pnorm[..., 0]
            norm_th = 0.1
            pind = pnorm < norm_th
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            pose_pts = pose_pts[pind][None]
            viewdir = viewdir[pind][None]
            pose_dirs = pose_dirs[pind][None]

        # transform points from the pose space to the tpose space
        tpose, tpose_dirs, init_bigpose, resd = self.pose_points_to_tpose_points(
            pose_pts, pose_dirs, batch)
        tpose = tpose[0]
        if cfg.tpose_viewdir:
            viewdir = tpose_dirs[0]
        else:
            viewdir = viewdir[0]
        ret = self.tpose_human(tpose, viewdir, dists, batch)

        ind = ret['sdf'][:, 0].detach().abs() < 0.02
        init_bigpose = init_bigpose[0][ind][None].detach().clone()

        if ret['raw'].requires_grad and ind.sum() != 0:
            observed_gradients, _ = self.gradient_of_deformed_sdf(
                init_bigpose, batch)
            ret.update({'observed_gradients': observed_gradients})

        tbounds = batch['tbounds'][0]
        tbounds[0] -= 0.05
        tbounds[1] += 0.05
        inside = tpose > tbounds[:1]
        inside = inside * (tpose < tbounds[1:])
        outside = torch.sum(inside, dim=1) != 3
        ret['raw'][outside] = 0

        n_batch, n_point = wpts.shape[:2]
        raw = torch.zeros([n_batch, n_point, 4]).to(wpts)
        raw[pind] = ret['raw']
        sdf = 10 * torch.ones([n_batch, n_point, 1]).to(wpts)
        sdf[pind] = ret['sdf']
        ret.update({'raw': raw, 'sdf': sdf, 'resd': resd})

        ret.update({'gradients': ret['gradients'][None]})

        return ret

    def get_sdf(self, wpts, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])

        with torch.no_grad():
            pbw, pnorm = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
            pnorm = pnorm[..., 0]
            norm_th = 0.1
            pind = pnorm < norm_th
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            pose_pts = pose_pts[pind][None]

        # initial blend weights of points at i
        pbw, _ = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
        pbw = pbw.permute(0, 2, 1)

        # transform points from i to i_0
        init_tpose = pose_points_to_tpose_points(pose_pts, pbw,
                                                 batch['A'])
        init_bigpose = tpose_points_to_pose_points(init_tpose, pbw,
                                                   batch['big_A'])
        resd = self.calculate_residual_deformation(init_bigpose, batch)
        tpose = init_bigpose + resd
        tpose = tpose[0]

        sdf_nn_output = self.tpose_human.sdf_network(tpose, batch)
        sdf = sdf_nn_output[:, 0]

        n_batch, n_point = wpts.shape[:2]
        sdf_full = 10 * torch.ones([n_batch, n_point]).to(wpts)
        sdf_full[pind] = sdf
        sdf = sdf_full.view(-1, 1)

        return sdf


class TPoseHuman(nn.Module):
    def __init__(self):
        super(TPoseHuman, self).__init__()

        self.sdf_network = SDFNetwork()
        self.beta_network = BetaNetwork()
        self.color_network = ColorNetwork()

    def sdf_to_alpha(self, sdf, beta):
        x = -sdf

        # select points whose x is smaller than 0: 1 / beta * 0.5 * exp(x/beta)
        ind0 = x <= 0
        val0 = 1 / beta * (0.5 * torch.exp(x[ind0] / beta))

        # select points whose x is bigger than 0: 1 / beta * (1 - 0.5 * exp(-x/beta))
        ind1 = x > 0
        val1 = 1 / beta * (1 - 0.5 * torch.exp(-x[ind1] / beta))

        val = torch.zeros_like(sdf)
        val[ind0] = val0
        val[ind1] = val1

        return val

    def forward(self, wpts, viewdir, dists, batch):
        # calculate sdf
        wpts.requires_grad_()
        with torch.enable_grad():
            sdf_nn_output = self.sdf_network(wpts, batch)
            sdf = sdf_nn_output[:, :1]

        feature_vector = sdf_nn_output[:, 1:]

        # calculate normal
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf,
                                        inputs=wpts,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        # gradients = self.sdf_network.gradient(wpts, batch)[:, 0]

        # calculate alpha
        wpts = wpts.detach()
        beta = self.beta_network(wpts).clamp(1e-9, 1e6)
        alpha = self.sdf_to_alpha(sdf, beta)
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(
            raw) * 0.005)
        alpha = raw2alpha(alpha[:, 0], dists)

        # calculate color
        ind = batch['latent_index']
        rgb = self.color_network(wpts, gradients, viewdir, feature_vector, ind)

        raw = torch.cat((rgb, alpha[:, None]), dim=1)
        ret = {'raw': raw, 'sdf': sdf, 'gradients': gradients}

        return ret


class SDFNetwork(nn.Module):
    def __init__(self):
        super(SDFNetwork, self).__init__()

        d_in = 3
        d_out = 257
        d_hidden = 256
        n_layers = 8

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        multires = 6
        if multires > 0:
            embed_fn, input_ch = embedder.get_embedder(multires,
                                                       input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        skip_in = [4]
        bias = 0.5
        scale = 1
        geometric_init = True
        weight_norm = True
        activation = 'softplus'

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs, batch):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x, batch):
        return self.forward(x, batch)[:, :1]

    def gradient(self, x, batch):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x, batch)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)


class BetaNetwork(nn.Module):
    def __init__(self):
        super(BetaNetwork, self).__init__()
        init_val = 0.1
        self.register_parameter('beta', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        beta = self.beta
        # beta = torch.exp(self.beta).to(x)
        return beta


class ColorNetwork(nn.Module):
    def __init__(self):
        super(ColorNetwork, self).__init__()

        self.color_latent = nn.Embedding(cfg.num_latent_code, 128)

        d_feature = 256
        mode = 'idr'
        d_in = 9
        d_out = 3
        d_hidden = 256
        n_layers = 4
        squeeze_out = True

        if not cfg.get('color_with_viewdir', True):
            mode = 'no_view_dir'

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden
                                     for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if self.mode != 'no_view_dir':
            multires_view = 4
            if multires_view > 0:
                embedview_fn, input_ch = embedder.get_embedder(multires_view)
                self.embedview_fn = embedview_fn
                dims[0] += (input_ch - 3)
        else:
            dims[0] = dims[0] - 3

        self.num_layers = len(dims)

        self.lin0 = nn.Linear(dims[0], d_hidden)
        self.lin1 = nn.Linear(d_hidden, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_hidden)
        self.lin3 = nn.Linear(d_hidden + 128, d_hidden)
        self.lin4 = nn.Linear(d_hidden, d_out)

        weight_norm = True
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors,
                latent_index):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat(
                [points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors],
                                        dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors],
                                        dim=-1)

        x = rendering_input

        net = self.relu(self.lin0(x))
        net = self.relu(self.lin1(net))
        net = self.relu(self.lin2(net))

        latent = self.color_latent(latent_index)
        latent = latent.expand(net.size(0), latent.size(1))
        features = torch.cat((net, latent), dim=1)

        net = self.relu(self.lin3(features))
        x = self.lin4(net)

        if self.squeeze_out:
            x = torch.sigmoid(x)

        return x
