import torch.nn.functional as F
import torch
from lib.config import cfg


def raw2outputs(raw, z_vals, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    rgb = raw[..., :-1]  # [N_rays, N_samples, 3]
    alpha = raw[..., -1]

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(depth_map),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    from torchsearchsorted import searchsorted

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf],
                    -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples).to(cdf)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(cdf)

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def get_intersection_mask(sdf, z_vals):
    """
    sdf: n_batch, n_pixel, n_sample
    z_vals: n_batch, n_pixel, n_sample
    """
    sign = torch.sign(sdf[..., :-1] * sdf[..., 1:])
    ind = torch.min(sign * torch.arange(sign.size(2)).flip([0]).to(sign),
                    dim=2)[1]
    sign = sign.min(dim=2)[0]
    intersection_mask = sign == -1
    return intersection_mask, ind


def sphere_tracing(wpts, sdf, z_vals, ray_o, ray_d, decoder):
    """
    wpts: n_point, n_sample, 3
    sdf: n_point, n_sample
    z_vals: n_point, n_sample
    ray_o: n_point, 3
    ray_d: n_point, 3
    """
    sign = torch.sign(sdf[..., :-1] * sdf[..., 1:])
    ind = torch.min(sign * torch.arange(sign.size(1)).flip([0]).to(sign),
                    dim=1)[1]

    wpts_sdf = sdf[torch.arange(len(ind)), ind]
    wpts_start = wpts[torch.arange(len(ind)), ind]
    wpts_end = wpts[torch.arange(len(ind)), ind + 1]

    sdf_threshold = 5e-5
    unfinished_mask = wpts_sdf.abs() > sdf_threshold
    i = 0
    while unfinished_mask.sum() != 0 and i < 20:
        curr_start = wpts_start[unfinished_mask]
        curr_end = wpts_end[unfinished_mask]

        wpts_mid = (curr_start + curr_end) / 2
        mid_sdf = decoder(wpts_mid)[:, 0]

        ind_outside = mid_sdf > 0
        if ind_outside.sum() > 0:
            curr_start[ind_outside] = wpts_mid[ind_outside]

        ind_inside = mid_sdf < 0
        if ind_inside.sum() > 0:
            curr_end[ind_inside] = wpts_mid[ind_inside]

        wpts_start[unfinished_mask] = curr_start
        wpts_end[unfinished_mask] = curr_end
        wpts_sdf[unfinished_mask] = mid_sdf
        unfinished_mask[unfinished_mask] = (mid_sdf.abs() >
                                            sdf_threshold) | (mid_sdf < 0)

        i = i + 1

    # get intersection points
    mask = (wpts_sdf.abs() < sdf_threshold) * (wpts_sdf >= 0)
    intersection_points = wpts_start[mask]

    ray_o = ray_o[mask]
    ray_d = ray_d[mask]
    z_vals = (intersection_points[:, 0] - ray_o[:, 0]) / ray_d[:, 0]

    return intersection_points, z_vals, mask
