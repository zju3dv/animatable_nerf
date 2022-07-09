import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.utils.blend_utils import *
from .. import embedder
from lib.utils import net_utils
from lib.utils import sample_utils


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.tpose_human = TPoseHuman()

        self.bw_latent = nn.Embedding(cfg.num_train_frame + 1, 128)

        self.actvn = nn.ReLU()

        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, 24, 1)

        if cfg.aninerf_animation:
            self.novel_pose_bw = BackwardBlendWeight()

            if 'init_aninerf' in cfg:
                net_utils.load_network(self,
                                       'data/trained_model/deform/' +
                                       cfg.init_aninerf,
                                       strict=False)

    def get_bw_feature(self, pts, ind):
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        latent = self.bw_latent(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def calculate_neural_blend_weights(self, pose_pts, smpl_bw, latent_index):
        features = self.get_bw_feature(pose_pts, latent_index)
        net = features
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        bw = self.bw_fc(net)
        bw = torch.log(smpl_bw + 1e-9) + bw
        bw = F.softmax(bw, dim=1)
        return bw

    def pose_points_to_tpose_points(self, pose_pts, pose_dirs, batch):
        """
        pose_pts: n_batch, n_point, 3
        """
        # initial blend weights of points at i
        init_pbw, _ = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
        init_pbw = init_pbw.permute(0, 2, 1)

        # neural blend weights of points at i
        if cfg.test_novel_pose:
            pbw = self.novel_pose_bw(pose_pts, init_pbw,
                                     batch['bw_latent_index'])
        else:
            pbw = self.calculate_neural_blend_weights(
                pose_pts, init_pbw, batch['latent_index'] + 1)

        # transform points from i to i_0
        tpose = pose_points_to_tpose_points(pose_pts, pbw, batch['A'])
        tpose = tpose_points_to_pose_points(tpose, pbw, batch['big_A'])

        if cfg.tpose_viewdir and pose_dirs is not None:
            init_tdirs = pose_dirs_to_tpose_dirs(pose_dirs, pbw,
                                                 batch['A'])
            tpose_dirs = tpose_dirs_to_pose_dirs(init_tdirs, pbw,
                                                 batch['big_A'])
        else:
            tpose_dirs = None

        return tpose, pbw, tpose_dirs

    def forward(self, wpts, viewdir, dists, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])
        viewdir = viewdir[None]
        pose_dirs = world_dirs_to_pose_dirs(viewdir, batch['R'])

        with torch.no_grad():
            pbw, pnorm = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
            pnorm = pnorm[..., 0]
            norm_th = cfg.norm_th
            pind = pnorm < norm_th
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            pose_pts = pose_pts[pind][None]
            viewdir = viewdir[pind][None]
            pose_dirs = pose_dirs[pind][None]
            dists = dists[pind[0]]

        # transform points from the pose space to the tpose space
        tpose, pbw, tpose_dirs = self.pose_points_to_tpose_points(pose_pts, pose_dirs, batch)

        # calculate neural blend weights of points at the tpose space
        init_tbw, _ = sample_utils.sample_blend_closest_points(tpose, batch['tvertices'], batch['weights'])
        init_tbw = init_tbw.permute(0, 2, 1)
        ind = torch.zeros_like(batch['latent_index'])
        tbw = self.calculate_neural_blend_weights(tpose, init_tbw, ind)

        tpose = tpose[0]
        if cfg.tpose_viewdir:
            viewdir = tpose_dirs[0]
        else:
            viewdir = viewdir[0]
        ret = self.tpose_human(tpose, viewdir, dists, batch)

        tbounds = batch['tbounds'][0]
        tbounds[0] -= 0.05
        tbounds[1] += 0.05
        inside = tpose > tbounds[:1]
        inside = inside * (tpose < tbounds[1:])
        outside = torch.sum(inside, dim=1) != 3
        ret['raw'][outside] = 0
        alpha = ret['raw'][..., -1]

        n_batch, n_point = wpts.shape[:2]
        raw = torch.zeros([n_batch, n_point, 4]).to(wpts)
        raw[pind] = ret['raw']
        ret.update({'raw': raw})

        alpha = alpha[None]
        alpha_ind = alpha.detach() > cfg.train_th
        max_ind = torch.argmax(alpha, dim=1)
        alpha_ind[torch.arange(alpha.size(0)), max_ind] = True
        pbw = pbw.transpose(1, 2)[alpha_ind][None]
        tbw = tbw.transpose(1, 2)[alpha_ind][None]

        ret.update({'pbw': pbw, 'tbw': tbw})

        return ret

    def get_alpha(self, wpts, batch):
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
        init_pbw, _ = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
        init_pbw = init_pbw.permute(0, 2, 1)

        # neural blend weights of points at i
        if cfg.test_novel_pose:
            pbw = self.novel_pose_bw(pose_pts, init_pbw,
                                     batch['bw_latent_index'])
        else:
            pbw = self.calculate_neural_blend_weights(
                pose_pts, init_pbw, batch['latent_index'] + 1)

        # transform points from i to i_0
        tpose = pose_points_to_tpose_points(pose_pts, pbw, batch['A'])
        tpose = tpose_points_to_pose_points(tpose, pbw, batch['big_A'])
        tpose = tpose[0]

        nerf_nn_output = self.tpose_human.nerf_network(tpose)
        alpha = nerf_nn_output[:, 0]

        n_batch, n_point = wpts.shape[:2]
        alpha_full = torch.zeros([n_batch, n_point]).to(wpts)
        alpha_full[pind] = alpha
        alpha = alpha_full.view(-1)

        return alpha


class BackwardBlendWeight(nn.Module):
    def __init__(self):
        super(BackwardBlendWeight, self).__init__()

        self.bw_latent = nn.Embedding(cfg.num_eval_frame, 128)

        self.actvn = nn.ReLU()

        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, 24, 1)

    def get_point_feature(self, pts, ind, latents):
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        latent = latents(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def forward(self, ppts, smpl_bw, latent_index):
        latents = self.bw_latent
        features = self.get_point_feature(ppts, latent_index, latents)
        net = features
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        bw = self.bw_fc(net)
        bw = torch.log(smpl_bw + 1e-9) + bw
        bw = F.softmax(bw, dim=1)
        return bw


class TPoseHuman(nn.Module):
    def __init__(self):
        super(TPoseHuman, self).__init__()

        self.nerf_network = NeRFNetwork()
        self.color_network = ColorNetwork()

    def calculate_alpha(self, tpose):
        nerf_nn_output = self.nerf_network(tpose[0])
        alpha = nerf_nn_output[:, :1]
        alpha = alpha[None].transpose(1, 2)
        return alpha

    def forward(self, wpts, viewdir, dists, batch):
        # calculate nerf
        nerf_nn_output = self.nerf_network(wpts)
        alpha = nerf_nn_output[:, 0]
        feature_vector = nerf_nn_output[:, 1:]

        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(
            raw) * dists)
        alpha = raw2alpha(alpha, dists)

        # calculate color
        ind = batch['latent_index']
        rgb = self.color_network(wpts, viewdir, feature_vector, ind)

        raw = torch.cat((rgb, alpha[:, None]), dim=1)
        ret = {'raw': raw}

        return ret


class NeRFNetwork(nn.Module):
    def __init__(self):
        super(NeRFNetwork, self).__init__()

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

    def forward(self, inputs):
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


class ColorNetwork(nn.Module):
    def __init__(self):
        super(ColorNetwork, self).__init__()

        self.color_latent = nn.Embedding(cfg.num_latent_code, 128)

        d_feature = 256
        mode = 'idr'
        d_in = 6
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

    def forward(self, points, view_dirs, feature_vectors,
                latent_index):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat(
                [points, view_dirs, feature_vectors], dim=-1)
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
