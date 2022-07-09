import torch.nn.functional as F
import torch
import numpy as np
from lib.config import cfg


def get_alpha_inter_ratio(iter_step):
    anneal_start = 0
    anneal_end = 25000
    x = np.min([1.0, (iter_step - anneal_start) / (anneal_end - anneal_start)])
    return x


def sdf_to_alpha(sdf, inv_variance, viewdir, gradients, dists, batch):
    """
    sdf: n_point, 1
    inv_variance: n_point, 1
    viewdir: n_point, 3
    gradients: n_point, 3
    dists: n_point

    prev_sigmoid = sigmoid(s * sdf_i)
    next_sigmoid = sigmoid(s * sdf_{i + 1})
    alpha = max((prev_sigmoid - next_sigmoid) / prev_sigmoid, 0)
    """
    # # calculate cos between the normal and viewding direction
    # true_dot_val = (viewdir * gradients).sum(-1, keepdim=True)

    # # soft clamp of the cos
    # alpha_inter_ratio = get_alpha_inter_ratio(batch['iter_step'])
    # iter_cos = -(
    #     F.relu(-true_dot_val * 0.5 + 0.5) * (1.0 - alpha_inter_ratio) +
    #     F.relu(-true_dot_val) * alpha_inter_ratio)  # always non-positive

    # true_estimate_sdf_half_next = sdf + iter_cos.clamp(
    #     -10.0, 10.0) * dists.reshape(-1, 1) * 0.5
    # true_estimate_sdf_half_prev = sdf - iter_cos.clamp(
    #     -10.0, 10.0) * dists.reshape(-1, 1) * 0.5

    cdf = torch.sigmoid(sdf * inv_variance)
    if 'pind' in batch:
        pind = batch['pind'].view(-1)
        n_point = pind.shape[0]
        full_cdf = torch.ones([n_point]).to(sdf)
        full_cdf[pind] = cdf[:, 0]
        cdf = full_cdf

    n_point = cdf.shape[0]
    cdf = cdf.reshape(-1, cfg.N_samples)

    residual = cdf[:, :-1] - cdf[:, 1:]
    p = torch.cat([residual, residual[:, -1:]], dim=1).reshape(n_point, 1)
    c = cdf.reshape(n_point, 1)

    # prev_cdf = torch.sigmoid(true_estimate_sdf_half_prev * inv_variance)
    # next_cdf = torch.sigmoid(true_estimate_sdf_half_next * inv_variance)

    # p = prev_cdf - next_cdf
    # c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).clamp(0.0, 1.0)

    if 'pind' in batch:
        alpha = alpha[pind]

    return alpha
