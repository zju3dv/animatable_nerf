# Typing

from collections import namedtuple
from typing import Callable, List

# Torch
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# Others
from tqdm import tqdm

# Utils
# from lib.utils.net_utils import l2, normalize_sum

# PyTorch3D
from pytorch3d import _C
from pytorch3d.structures import Meshes
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes, _rand_barycentric_coords


class PointMeshDistance(Function):
    # PointFaceDistance
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    RETURNS FACE_IDX AND ACTUAL DISTANCE INSTEAD OF SQUARED
    """
    @staticmethod
    def forward(ctx, points, tris, n_batch=1):
        """
        Args:
            points: FloatTensor of shape `(P, 3)`
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            n_batch: Num of batch
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the REAL! NOT SQUARED
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.
        """
        n_points = points.shape[0]
        device = points.device
        zeros = torch.zeros(n_batch, dtype=torch.long, device=device)
        dists, idxs = _C.point_face_dist_forward(
            points, zeros, tris, zeros, n_points
        )
        ctx.save_for_backward(points, tris, idxs)
        return dists.sqrt(), idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists
        )
        return grad_points, None, grad_tris, None, None


# pyre-fixme[16]: `_PointFaceDistance` has no attribute `apply`.
point_mesh_distance = PointMeshDistance.apply


def random_points_on_meshes_with_face_and_bary(meshes: Meshes, num_samples: int):
    verts = meshes.verts_packed()
    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(
            areas, mesh_to_face[meshes.valid], max_faces
        )  # (N, F)

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        sample_face_idxs = areas_padded.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)
        sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_valid_meshes, num_samples, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c
    bary = torch.stack((w0, w1, w2), dim=2)
    return samples, sample_face_idxs, bary


def get_voxel_grid_and_update_bounds(voxel_size: List, bounds: torch.Tensor):
    # now here's the problem
    # 1. if you want the voxel size to be accurate, you bounds need to be changed along with this sampling process
    #    since the F.grid_sample will treat the bounds based on align_corners=True or not
    #    say we align corners, the actual bound on the sampled tpose blend weight should be determined by the actual sampling voxels
    #    not the bound that we kind of used to produce the voxels, THEY DO NOT LINE UP UNLESS your bounds is divisible by the voxel size in every direction

    # voxel_size: [0.005, 0.005, 0.005]
    # bounds: n_batch, 2, 3, initial bounds
    ret = []
    for b in bounds:
        x = torch.arange(b[0, 0].item(), b[1, 0].item() + voxel_size[0]/2, voxel_size[0], dtype=b.dtype, device=b.device)
        y = torch.arange(b[0, 1].item(), b[1, 1].item() + voxel_size[1]/2, voxel_size[1], dtype=b.dtype, device=b.device)
        z = torch.arange(b[0, 2].item(), b[1, 2].item() + voxel_size[2]/2, voxel_size[2], dtype=b.dtype, device=b.device)
        pts = torch.stack(torch.meshgrid(x, y, z), dim=-1)
        ret.append(pts)
    pts = torch.stack(ret)  # dim 0
    bounds = torch.stack([pts[:, 0, 0, 0], pts[:, -1, -1, -1]], dim=1)  # dim 1 n_batch, 2, 3
    return pts, bounds


def optimize_until_no_nan(func: Callable[..., torch.Tensor], *args: torch.Tensor):
    # Note: assuming first value in args is to be optimized and can be NaN
    # FIXME: nasty fix for nan: repeating until it's not a nan anymore, while True is bad...
    param_to_optim = args[0]
    while True:

        param_to_optim = func(
            param_to_optim.clone(), *args[1:]
        )

        if param_to_optim.isnan().any():
            print_red("NaN detected, repeating grid optimization step...")
            continue
        else:
            break
    return param_to_optim


def optimze_samples_from_volume(grid: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, values: torch.Tensor, bounds: torch.Tensor, n_free_pts: int = 1024, n_surf_pts: int = 1024, norm_expand_step: int = 2, norm_expand_factor: float = 0.02, n_iter: int = 128,  lr: float = 1e-3, clip_grad=10):
    # post_processing: apply additional constraints on returned values (like sums to one)
    n_batch, w, h, d, D = grid.shape
    diag = bounds[:, 1] - bounds[:, 0]  # B, 3

    mesh = Meshes(verts, faces)
    norm = mesh.verts_normals_padded()

    grid.requires_grad = True
    optim = Adam([grid], lr=lr)

    p = tqdm(range(n_iter))
    with torch.enable_grad():  # you've turned off grad before main
        for _ in p:

            free_pts = torch.rand([n_batch, n_free_pts, 3], dtype=grid.dtype, device=grid.device)
            free_pts *= diag  # [0,1] -> diagonal
            free_pts += bounds[:, 0]  # diagonal -> shifted
            surf, norm = sample_points_from_meshes(mesh, n_surf_pts, True)
            surf_pts = torch.cat([surf + norm * norm_expand_factor * (torch.rand(1, dtype=grid.dtype, device=grid.device) * 2 - 1) for _ in range(norm_expand_step)], dim=1)
            pts = torch.cat([free_pts, verts, surf_pts], dim=1)  # make sure verts gets mapped correctly

            pred = sample_grid(pts, grid, bounds)
            gt, dists = sample_closest_points_on_surface(pts, verts, faces, values)

            loss = l2(pred, gt)
            p.set_description(f"l2: {loss:.6f}")

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_([grid], clip_grad)
            optim.step()

    return grid


def points_to_barycentric(points: torch.Tensor, triangles: torch.Tensor, method="cramer", eps: float = 1e-16):
    # points: n_points, 3
    # triangles: n_points, 3, 3

    def method_cross(edge_vectors: torch.Tensor, w: torch.Tensor):
        n = torch.cross(edge_vectors[:, 0], edge_vectors[:, 1])
        denominator = torch.bmm(n[:, None], n[..., None])[:, 0, 0]
        denominator[denominator.abs() < eps] = eps

        barycentric = torch.zeros((len(triangles), 3), dtype=points.dtype, device=points.device)
        barycentric[:, 2] = torch.bmm(
            torch.cross(edge_vectors[:, 0], w)[:, None], n[..., None])[:, 0, 0] / denominator
        barycentric[:, 1] = torch.bmm(
            torch.cross(w, edge_vectors[:, 1])[:, None], n[..., None])[:, 0, 0] / denominator
        barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]
        return barycentric

    def method_cramer(edge_vectors: torch.Tensor, w: torch.Tensor):
        # n_points, 1, 3 @ n_points, 3, 1 -> n_points, 1, 1
        dot00 = torch.bmm(edge_vectors[:, 0][:, None], edge_vectors[:, 0][..., None])[:, 0, 0]  # n_points
        dot01 = torch.bmm(edge_vectors[:, 0][:, None], edge_vectors[:, 1][..., None])[:, 0, 0]
        dot02 = torch.bmm(edge_vectors[:, 0][:, None],                  w[..., None])[:, 0, 0]
        dot11 = torch.bmm(edge_vectors[:, 1][:, None], edge_vectors[:, 1][..., None])[:, 0, 0]
        dot12 = torch.bmm(edge_vectors[:, 1][:, None],                  w[..., None])[:, 0, 0]

        denominator = dot00 * dot11 - dot01 * dot01
        denominator[denominator.abs() < eps] = eps

        barycentric = torch.zeros((len(triangles), 3), dtype=points.dtype, device=points.device)
        barycentric[:, 2] = (dot00 * dot12 - dot01 * dot02) / denominator
        barycentric[:, 1] = (dot11 * dot02 - dot01 * dot12) / denominator
        barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]
        return barycentric

    def constraint_barycentric(barycentric: torch.Tensor):
        barycentric[barycentric < 0] = 0
        barycentric = normalize_sum(barycentric)
        return barycentric

    edge_vectors = triangles[:, 1:] - triangles[:, :1]  # n_points, 2, 3
    w = points - triangles[:, 0].view(-1, 3)  # n_points, 3

    # trimesh.triangles.points_to_barycentric(triangles.cpu().numpy(), points.cpu().numpy())
    if method == "cramer":
        barycentric = method_cramer(edge_vectors, w)  # might be out of bound
    else:
        barycentric = method_cross(edge_vectors, w)  # might be out of bound

    barycentric = constraint_barycentric(barycentric)
    return barycentric


def sample_grid(pts: torch.Tensor, values: torch.Tensor, bounds: torch.Tensor):
    """sample blend weights for points
    https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    pts: n_batch, n_points, 3
    values: n_batch, w, h, d, n_bones
    bounds: n_batch, 2, 3

    returns:
    values: n_batch, n_points, n_bones
    """
    # interpolate blend weights
    diagonal = bounds[:, 1:] - bounds[:, :1]  # n_batch, 1, 3
    grid_coords = (pts - bounds[:, :1]) / diagonal  # n_batch, n_points, 3
    grid_coords = grid_coords * 2 - 1  # to ndc space, n_batch, n_points, 3
    # convert xyz to zyx, since the blend weight is indexed by xyz
    grid_coords = grid_coords[..., [2, 1, 0]]

    # the blend weight is indexed by xyz
    # TODO: what the heck between the dhw and whd conversion?
    values = values.permute(0, 4, 1, 2, 3)  # n_batch, n_bones, w, h, d
    grid_coords = grid_coords[:, None, None]

    values = F.grid_sample(values,  # n_batch, n_bones, w, h, d
                           grid_coords,  # n_batch, 1, 1, n_points, 3 (now indexing zyx)
                           padding_mode='border',
                           align_corners=False)  # n_batch, n_bones, w, h, d
    values = values[:, :, 0, 0].permute(0, 2, 1)  # (n_batch, n_bones, n_points) -> (n_batch, n_points, n_bones)

    return values


def expand_points_for_sampling(verts: torch.Tensor, faces: torch.Tensor, values: torch.Tensor, expand_range: List, expand_step: int):
    # prepare emission (along normal direction)
    # mesh = get_mesh(verts, faces)
    mesh = Meshes(verts, faces)
    vert_cnt = verts.shape[1]  # consider batch dimension
    vert_norms = mesh.verts_normals_padded()
    expand_len = expand_range[1] - expand_range[0]
    expand_min = expand_range[0]
    expand_cnt = expand_step + 1
    # vert_norms = torch.tensor(mesh.vertex_normals, dtype=pts.dtype, device=pts.device)[None] # add batch dimension
    expand_verts = torch.cat(
        [
            (expand_min + i * expand_len / expand_step) * vert_norms + verts for i in range(expand_cnt)
        ],
        dim=1  # verts has a batch dimension
    )
    expand_faces = torch.cat(
        [
            faces + i * vert_cnt for i in range(expand_cnt)
        ],
        dim=1,  # faces has a batch dimension
    )
    expand_values = values.repeat(1, expand_cnt, 1)

    return expand_verts, expand_faces, expand_values


def sample_grid_closest_points_on_surface(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, values: torch.Tensor):
    return [x.view(*pts.shape[:-1], -1) for x in sample_closest_points_on_surface(pts.view(pts.shape[0], -1, 3), verts, faces, values)]  # (1, whd, n_bones+3)


def sample_grid_closest_points_on_expanded_surface(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, values: torch.Tensor, expand_range: List, expand_step: int):
    expand_verts, expand_faces, expand_values = expand_points_for_sampling(verts, faces, values, expand_range, expand_step)
    return sample_grid_closest_points_on_surface(pts, expand_verts, expand_faces, expand_values)


def guard_knn_points(src, ref, K):
    ret = knn_points(src, ref, K=K)
    return namedtuple('ret', ['dists', 'idx'])(dists=ret.dists.sqrt(), idx=ret.idx)


def sample_closest_points(src: torch.Tensor, ref: torch.Tensor, values: torch.Tensor):
    n_batch, n_points, _ = src.shape
    ret = guard_knn_points(src, ref, K=1)  # (n_batch, n_points, K)
    dists, vert_ids = ret.dists, ret.idx
    values = values.view(-1, values.shape[-1])  # (n, D)
    sampled = values[vert_ids]  # (s, D)
    return sampled.view(n_batch, n_points, -1), dists.view(n_batch, n_points, 1)


def sample_blend_closest_points(src: torch.Tensor, ref: torch.Tensor, values: torch.Tensor, K: int = 5, exp: float = 1e-8):
    # not so useful to aggregate all K points
    n_batch, n_points, _ = src.shape
    ret = guard_knn_points(src, ref, K=K)
    dists, vert_ids = ret.dists, ret.idx  # (n_batch, n_points, K)
    values = values.view(-1, values.shape[-1])  # (n, D)
    # sampled = values[vert_ids]  # (n_batch, n_points, K, D)
    disp = 1 / (dists + exp)  # inverse distance: disparity
    weights = disp / disp.sum(dim=-1, keepdim=True)  # normalize distance by K
    dists = torch.einsum('ijk,ijk->ij', dists, weights)
    # sampled *= weights[..., None]  # augment weight in last dim for bones # written separatedly to avoid OOM
    # sampled = sampled.sum(dim=-2)  # sum over second to last for weighted bw
    sampled = torch.einsum('ijkl,ijk->ijl', values[vert_ids], weights)
    return sampled.view(n_batch, n_points, -1), dists.view(n_batch, n_points, 1)


def sample_closest_points_on_surface_approx(points: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, values: torch.Tensor, n_surf_pts: int = 16384):
    # even slower...
    # expect all to have batch dim
    n_batch, n_points, _ = points.shape

    # similiar to computing point mesh distance and then do a barycentric
    # but directly sample points on surface, then do a ball query for values
    mesh = Meshes(verts, faces)
    # FIXME: bary is not the same bary as in points_to_barycentric
    # surf: (b, n, 3)
    # face_ids: (b, n)
    # bary: (b, n, 3)
    surf, face_ids, bary = random_points_on_meshes_with_face_and_bary(mesh, n_surf_pts)
    # dists: (b, n, 1)
    # surf_ids: (b, n, 1)
    # nn: (b, n, 1, 3)
    dists, vert_ids = guard_knn_points(points, surf, K=1)

    values = values.view(-1, values.shape[-1])  # (n, D)
    faces = faces.view(-1, faces.shape[-1])  # (n ,3)
    bary = bary.view(-1, bary.shape[-1])  # (n, 3)
    face_ids = face_ids.view(-1)  # (f)

    sampled = torch.sum(values[faces[face_ids]] *  # (n, 3, 3)
                        bary[..., None],  # (n, 3, 1)
                        dim=1)

    vert_ids = vert_ids.view(-1)  # (s)
    sampled = sampled[vert_ids]  # (s, D)
    return sampled.view(n_batch, n_points, sampled.shape[-1]), dists.view(n_batch, n_points, 1)


def sample_closest_points_on_surface(points: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, values: torch.Tensor):
    # samples values by barycentricing the closest points on the mesh
    # points: n_batch, n_points, 3
    # verts: n_batch, n_verts, 3
    # faces: n_batch, n_faces, 3 (long)
    # values: n_batch, n_verts, D (might need to augment some)

    # FIXME: this is not gonna be differentiable
    # https://github.com/facebookresearch/pytorch3d/issues/193, pytorch doesn't expose the indices' api
    # and we decide to resort back to Trimesh for some R-trees
    # this goes inefficient, and maybe we should not calculate this on the fly?
    # TODO: test out the performance of this
    n_batch, n_points, _ = points.shape
    n_batch, n_verts, D = values.shape

    points = points.view(-1, 3)
    verts = verts.view(-1, 3)
    faces = faces.view(-1, 3)
    values = values.view(-1, D)

    # # FIXME: trimesh functions returns nothing in the eye of pylance
    # # using trimesh to access them to make pylance happy
    # import trimesh
    # mesh = trimesh.Trimesh(verts, faces)
    # closest, distance, face_id = trimesh.proximity.closest_point(mesh, points)
    # closest_verts = faces[face_id]  # (n, 3, 3)
    # barycentric = trimesh.triangles.points_to_barycentric(verts[closest_verts], closest)  # (n, 3)

    # device = points.device
    # closest_verts = torch.tensor(closest_verts).to(device)
    # barycentric = torch.tensor(barycentric).to(device)
    # values = torch.sum(values[closest_verts] *  # (n, 3, 3)
    #                    barycentric[..., None],  # (n, 3, 1)
    #                    dim=1)
    # we use pytorch for a faster cuda version instead of implementing by hand
    dists, face_ids = point_mesh_distance(points, verts[faces], n_batch)
    bary = points_to_barycentric(points, verts[faces[face_ids]])  # (n, 3)
    interp = torch.sum(values[faces[face_ids]] *  # (n, 3, 3)
                       bary[..., None],  # (n, 3, 1)
                       dim=1)

    return interp.view(n_batch, n_points, D), dists.view(n_batch, n_points, 1)
