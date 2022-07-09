import torch.nn.functional as F
import torch


def sdf_mask_crit(ret, batch):
    msk_sdf = ret['msk_sdf']
    msk_label = ret['msk_label']

    alpha = 50
    alpha_factor = 2
    alpha_milestones = [10000, 20000, 30000, 40000, 50000]
    for milestone in alpha_milestones:
        if batch['iter_step'] > milestone:
            alpha = alpha * alpha_factor

    msk_sdf = -alpha * msk_sdf
    mask_loss = F.binary_cross_entropy_with_logits(msk_sdf, msk_label) / alpha

    return mask_loss


def elastic_crit(ret, batch):
    """
    resd_jacobian: n_batch, n_point, 3, 3
    """
    jac = ret['resd_jacobian']
    U, S, V = torch.svd(jac, compute_uv=True)
    log_svals = torch.log(torch.clamp(S, min=1e-6))
    elastic_loss = torch.sum(log_svals**2, dim=2).mean()
    return elastic_loss


def normal_crit(ret, batch):
    surf_normal_pred = ret['surf_normal'][ret['surf_mask']]
    surf_normal = batch['normal'][ret['surf_mask']]

    viewdir = batch['ray_d'][ret['surf_mask']]
    weights = torch.sum(-surf_normal_pred * viewdir, dim=1)
    weights = torch.clamp(weights, min=0, max=1)**2

    norm = torch.norm(surf_normal, dim=1)
    norm[norm < 1e-8] = 1e-8
    surf_normal = surf_normal / norm[..., None]

    surf_normal_pred[:, 1:] = surf_normal_pred[:, 1:] * -1

    normal_loss = torch.norm(surf_normal_pred - surf_normal, dim=1)
    normal_loss = (weights * normal_loss).mean()

    return normal_loss
