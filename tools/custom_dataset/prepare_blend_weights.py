"""
Prepare blend weights of grid points
"""

import os
import cv2
import json
import pickle
import trimesh
import argparse
# import mesh_to_sdf
import numpy as np
import open3d as o3d
from tqdm import tqdm
from psbody.mesh import Mesh


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def get_smpl_faces(pkl_path):
    smpl = read_pickle(pkl_path)
    faces = smpl['f']
    return faces


def get_o3d_mesh(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


def process_shapedirs(shapedirs, vert_ids, bary_coords):
    arr = []
    for i in range(3):
        t = barycentric_interpolation(shapedirs[:, i, :][vert_ids],
                                      bary_coords)
        arr.append(t[:, np.newaxis, :])
    arr = np.concatenate(arr, axis=1)
    return arr


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_rigid_transformation(rot_mats, joints, parents):
    """
    rot_mats: n_bones x 3 x 3
    joints: n_bones x 3
    parents: n_bones
    """
    # obtain the relative joints
    n_bones = rot_mats.shape[0]
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([n_bones, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    # obtain the rigid transformation
    padding = np.zeros([n_bones, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    rel_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms


def get_transform_params(smpl, params):
    """ obtain the transformation parameters for linear blend skinning
    """
    v_template = np.array(smpl['v_template'])

    # add shape blend shapes
    shapedirs = np.array(smpl['shapedirs'])
    betas = params['shapes']
    n = betas.shape[-1]
    m = shapedirs.shape[-1]
    n = min(m, n)
    v_shaped = v_template + np.sum(shapedirs[..., :n] * betas[None][..., :n], axis=2)

    # add pose blend shapes
    poses = params['poses'].reshape(-1, 3)
    # 24 x 3 x 3
    rot_mats = batch_rodrigues(poses)

    # obtain the joints
    joints = smpl['J_regressor'].dot(v_shaped)

    # obtain the rigid transformation
    parents = smpl['kintree_table'][0]
    A = get_rigid_transformation(rot_mats, joints, parents)

    # apply global transformation
    R = cv2.Rodrigues(params['Rh'][0])[0]
    Th = params['Th']

    return A, R, Th, joints


def get_colored_pc(pts, rgb):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    colors = np.zeros_like(pts)
    colors += rgb
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc


def get_grid_points(xyz):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= 0.05
    max_xyz += 0.05
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    vsize = 0.025
    voxel_size = [vsize, vsize, vsize]
    x = np.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0], voxel_size[0])
    y = np.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1], voxel_size[1])
    z = np.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2], voxel_size[2])
    pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    return pts


def get_bweights(param_path, vertices_path):
    params = np.load(param_path, allow_pickle=True).item()
    vertices = np.load(vertices_path)
    faces = get_smpl_faces(smpl_path)
    mesh = get_o3d_mesh(vertices, faces)

    smpl = read_pickle(smpl_path)
    # obtain the transformation parameters for linear blend skinning
    A, R, Th, joints = get_transform_params(smpl, params)

    # transform points from the world space to the pose space
    pxyz = np.dot(vertices - Th, R)
    smpl_mesh = Mesh(pxyz, faces)

    # create grid points in the pose space
    pts = get_grid_points(pxyz)
    sh = pts.shape
    pts = pts.reshape(-1, 3)

    # obtain the blending weights for grid points
    vert_ids, norm = smpl_mesh.closest_vertices(pts, use_cgal=True)
    bweights = smpl['weights'][vert_ids]

    # closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
    # vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(
    #     closest_points, closest_face.astype('int32'))
    # bweights = barycentric_interpolation(smpl['weights'][vert_ids],
    #                                      bary_coords)

    # calculate the distance to the smpl surface
    # norm = np.linalg.norm(pts - closest_points, axis=1)

    A = np.dot(bweights, A.reshape(24, -1)).reshape(-1, 4, 4)
    can_pts = pts - A[:, :3, 3]
    R_inv = np.linalg.inv(A[:, :3, :3])
    can_pts = np.sum(R_inv * can_pts[:, None], axis=2)

    bweights = np.concatenate((bweights, norm[:, None]), axis=1)
    bweights = bweights.reshape(*sh[:3], 25).astype(np.float32)

    return bweights


def prepare_blend_weights(begin_frame, end_frame, frame_interval):
    annot_path = os.path.join(data_root, human, 'annots.npy')
    annot = np.load(annot_path, allow_pickle=True).item()
    bweight_dir = os.path.join(lbs_root, 'bweights')
    os.system('mkdir -p {}'.format(bweight_dir))

    end_frame = len(annot['ims']) if end_frame < 0 else end_frame
    for i in tqdm(range(begin_frame, end_frame, frame_interval)):
        param_path = os.path.join(param_dir, '{}.npy'.format(i))
        vertices_path = os.path.join(vertices_dir, '{}.npy'.format(i))
        bweights = get_bweights(param_path, vertices_path)
        bweight_path = os.path.join(bweight_dir, '{}.npy'.format(i))
        np.save(bweight_path, bweights)


def get_tpose_blend_weights():
    i = begin_frame
    param_path = os.path.join(param_dir, '{}.npy'.format(i))
    vertices_path = os.path.join(vertices_dir, '{}.npy'.format(i))

    params = np.load(param_path, allow_pickle=True).item()
    vertices = np.load(vertices_path)
    faces = get_smpl_faces(smpl_path)
    mesh = get_o3d_mesh(vertices, faces)

    smpl = read_pickle(smpl_path)
    # obtain the transformation parameters for linear blend skinning
    A, R, Th, joints = get_transform_params(smpl, params)

    parent_path = os.path.join(lbs_root, 'parents.npy')
    np.save(parent_path, smpl['kintree_table'][0])
    joint_path = os.path.join(lbs_root, 'joints.npy')
    np.save(joint_path, joints)

    # transform points from the world space to the pose space
    pxyz = np.dot(vertices - Th, R)
    smpl_mesh = Mesh(pxyz, faces)

    bweights = smpl['weights']
    A = np.dot(bweights, A.reshape(24, -1)).reshape(-1, 4, 4)
    can_pts = pxyz - A[:, :3, 3]
    R_inv = np.linalg.inv(A[:, :3, :3])
    pxyz = np.sum(R_inv * can_pts[:, None], axis=2)

    tvertices_path = os.path.join(lbs_root, 'tvertices.npy')
    np.save(tvertices_path, pxyz)

    smpl_mesh = Mesh(pxyz, faces)

    # create grid points in the pose space
    pts = get_grid_points(pxyz)
    sh = pts.shape
    pts = pts.reshape(-1, 3)

    # obtain the blending weights for grid points
    closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
    vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(
        closest_points, closest_face.astype('int32'))
    bweights = barycentric_interpolation(smpl['weights'][vert_ids],
                                         bary_coords)

    # calculate the distance to the smpl surface
    norm = np.linalg.norm(pts - closest_points, axis=1)

    bweights = np.concatenate((bweights, norm[:, None]), axis=1)
    bweights = bweights.reshape(*sh[:3], 25).astype(np.float32)
    bweight_path = os.path.join(lbs_root, 'tbw.npy')
    np.save(bweight_path, bweights)

    return bweights


def compute_unit_sphere_transform(mesh):
    """
    returns translation and scale, which is applied to meshes before computing their SDF cloud
    """
    # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
    translation = -mesh.bounding_box.centroid
    scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
    return translation, scale


def get_bigpose_blend_weights():
    i = begin_frame
    param_path = os.path.join(param_dir, '{}.npy'.format(i))
    vertices_path = os.path.join(vertices_dir, '{}.npy'.format(i))

    params = np.load(param_path, allow_pickle=True).item()
    vertices = np.load(vertices_path)
    faces = get_smpl_faces(smpl_path)
    mesh = get_o3d_mesh(vertices, faces)

    smpl = read_pickle(smpl_path)
    # obtain the transformation parameters for linear blend skinning
    A, R, Th, joints = get_transform_params(smpl, params)

    parent_path = os.path.join(lbs_root, 'parents.npy')
    np.save(parent_path, smpl['kintree_table'][0])
    joint_path = os.path.join(lbs_root, 'joints.npy')
    np.save(joint_path, joints)

    # transform points from the world space to the pose space
    pxyz = np.dot(vertices - Th, R)
    smpl_mesh = Mesh(pxyz, faces)

    bweights = smpl['weights']
    A = np.dot(bweights, A.reshape(24, -1)).reshape(-1, 4, 4)
    can_pts = pxyz - A[:, :3, 3]
    R_inv = np.linalg.inv(A[:, :3, :3])
    pxyz = np.sum(R_inv * can_pts[:, None], axis=2)

    # calculate big pose
    poses = params['poses'].reshape(-1, 3)
    big_poses = np.zeros_like(poses).ravel()
    angle = 30
    big_poses[5] = np.deg2rad(angle)
    big_poses[8] = np.deg2rad(-angle)
    # big_poses = big_poses.reshape(-1, 3)
    # big_poses[1] = np.array([0, 0, 7. / 180. * np.pi])
    # big_poses[2] = np.array([0, 0, -7. / 180. * np.pi])
    # big_poses[16] = np.array([0, 0, -55. / 180. * np.pi])
    # big_poses[17] = np.array([0, 0, 55. / 180. * np.pi])

    big_poses = big_poses.reshape(-1, 3)
    rot_mats = batch_rodrigues(big_poses)
    parents = smpl['kintree_table'][0]
    big_A = get_rigid_transformation(rot_mats, joints, parents)
    big_A = np.dot(bweights, big_A.reshape(24, -1)).reshape(-1, 4, 4)

    bigpose_vertices = np.sum(big_A[:, :3, :3] * pxyz[:, None], axis=2)
    bigpose_vertices = bigpose_vertices + big_A[:, :3, 3]

    bigpose_vertices_path = os.path.join(lbs_root, 'bigpose_vertices.npy')
    np.save(bigpose_vertices_path, bigpose_vertices)

    faces_path = os.path.join(lbs_root, 'faces.npy')
    np.save(faces_path, faces)

    smpl_mesh = Mesh(bigpose_vertices, faces)

    # create grid points in the pose space
    pts = get_grid_points(bigpose_vertices)
    sh = pts.shape
    pts = pts.reshape(-1, 3)

    # obtain the blending weights for grid points
    closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
    vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(
        closest_points, closest_face.astype('int32'))
    bweights = barycentric_interpolation(smpl['weights'][vert_ids],
                                         bary_coords)

    # calculate the distance to the smpl surface
    norm = np.linalg.norm(pts - closest_points, axis=1)

    bweights = np.concatenate((bweights, norm[:, None]), axis=1)
    bweights = bweights.reshape(*sh[:3], 25).astype(np.float32)
    bweight_path = os.path.join(lbs_root, 'bigpose_bw.npy')
    np.save(bweight_path, bweights)

    # # calculate sdf
    # mesh = trimesh.Trimesh(bigpose_vertices, faces)
    # points, sdf = mesh_to_sdf.sample_sdf_near_surface(mesh,
    #                                                   number_of_points=250000)
    # translation, scale = compute_unit_sphere_transform(mesh)
    # points = (points / scale) - translation
    # sdf /= scale
    # sdf_path = os.path.join(lbs_root, 'bigpose_sdf.npy')
    # np.save(sdf_path, {'points': points, 'sdf': sdf})

    return bweights


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/light_stage')
parser.add_argument('--humans', type=str, nargs='+', default=['CoreView_313', 'CoreView_315', "CoreView_377", "CoreView_386", "CoreView_387", "CoreView_390", "CoreView_392", "CoreView_393", "CoreView_394"])
parser.add_argument('--smpl', type=str, default='smpl', choices=['smplh', 'smpl'])
parser.add_argument('--begin_frames', type=int, nargs='+', default=[1, 1, 0, 0, 0, 0, 0, 0, 0])
parser.add_argument('--lbs', type=str, default='lbs')
parser.add_argument('--params', type=str, default='params')
parser.add_argument('--vertices', type=str, default='vertices')
args = parser.parse_args()

data_root = args.data_dir
tpose_geometry = True
frame_interval = 1

for human_ind in tqdm(range(len(args.humans))):
    try:
        human = args.humans[human_ind]
        begin_frame = args.begin_frames[human_ind]
        lbs_root = os.path.join(args.data_dir, human, args.lbs)
        param_dir = os.path.join(args.data_dir, human, args.params)
        vertices_dir = os.path.join(args.data_dir, human, args.vertices)
        smpl_path = f'data/smplx/{args.smpl}/{"SMPLH_male.pkl" if "smplh" in args.smpl else "SMPL_NEUTRAL.pkl"}'  # used in EasyMocap

        if args.smpl == 'smpl':
            n_bones = 24
        else:
            n_bones = 52

        os.system('mkdir -p {}'.format(lbs_root))

        last_frame = len(os.listdir(param_dir)) + begin_frame

        get_bigpose_blend_weights()
        get_tpose_blend_weights()
        prepare_blend_weights(begin_frame, last_frame, frame_interval)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        continue
