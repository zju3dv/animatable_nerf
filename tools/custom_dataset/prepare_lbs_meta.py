"""
Prepare blend weights parameters (no blend weight field) for lbs related stuff in project
This should call easymocap-public
"""

import tqdm
import argparse
from functools import lru_cache

import os
import numpy as np
import cv2

import pickle


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


@lru_cache()  # don't load again and again pls
def read_smpl_file(smpl_path):
    if smpl_path.endswith('.pkl'):
        return read_pickle(smpl_path)
    else:
        smpl = np.load(smpl_path)
        smpl = {**smpl}  # disable lazy loading of actual content
        return smpl


@lru_cache()  # don't load again and again pls
def get_smpl_faces(smpl_path):
    smpl = read_smpl_file(smpl_path)
    faces = smpl['f']
    return faces


def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


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
    shapedirs = np.array(smpl['shapedirs'])  # qing only used the first 10 shape
    betas = params['shapes']
    v_shaped = v_template + np.sum(shapedirs[..., :betas.shape[-1]] * betas[None], axis=2)

    # add pose blend shapes
    poses = params['poses'].reshape(-1, 3).astype(np.float32)
    # n_bones x 3 x 3
    rot_mats = batch_rodrigues(poses)

    # obtain the joints
    joints = smpl['J_regressor'].dot(v_shaped)[:n_bones]

    # obtain the rigid transformation
    parents = smpl['kintree_table'][0][:n_bones]
    A = get_rigid_transformation(rot_mats, joints, parents)

    # apply global transformation
    R = cv2.Rodrigues(params['Rh'][0])[0]
    Th = params['Th']

    return A, R, Th, joints, parents, v_shaped


def get_tpose_blend_weights():
    i = args.begin_frame
    param_path = os.path.join(param_dir, '{}.npy'.format(i))
    vertices_path = os.path.join(vertices_dir, '{}.npy'.format(i))

    params = np.load(param_path, allow_pickle=True).item()
    vertices = np.load(vertices_path)
    faces = get_smpl_faces(smpl_path)

    smpl = read_smpl_file(smpl_path)
    # obtain the transformation parameters for linear blend skinning
    A, R, Th, joints, parents, v_shaped = get_transform_params(smpl, params)

    parent_path = os.path.join(lbs_root, 'parents.npy')
    np.save(parent_path, parents)
    joint_path = os.path.join(lbs_root, 'joints.npy')
    np.save(joint_path, joints)

    # transform points from the world space to the pose space
    pxyz = np.dot(vertices - Th, R)

    if args.smpl == 'smpl' and ('SMPLH' in smpl_path or 'smplh' in smpl_path): 
        weight_smpl_path = f'data/smpl-all/{args.smpl}/{"SMPLH_male.pkl" if "smplh" in args.smpl else "SMPL_NEUTRAL.pkl"}'  # used in EasyMocap
        smpl = read_smpl_file(weight_smpl_path)
    bweights = smpl['weights'][..., :n_bones]
    bweights = bweights / bweights.sum(axis=-1, keepdims=True)  # FIXME: hand weight assignment not right

    weight_path = os.path.join(lbs_root, 'weights.npy')
    np.save(weight_path, bweights)

    A = np.dot(bweights, A.reshape(A.shape[0], -1)).reshape(-1, 4, 4)
    can_pts = pxyz - A[:, :3, 3]
    R_inv = np.linalg.inv(A[:, :3, :3])
    pxyz = np.sum(R_inv * can_pts[:, None], axis=2)

    tvertices_path = os.path.join(lbs_root, 'tvertices.npy')
    np.save(tvertices_path, pxyz)


def get_bigpose_blend_weights():
    i = args.begin_frame
    param_path = os.path.join(param_dir, '{}.npy'.format(i))
    vertices_path = os.path.join(vertices_dir, '{}.npy'.format(i))

    params = np.load(param_path, allow_pickle=True).item()
    vertices = np.load(vertices_path)
    faces = get_smpl_faces(smpl_path)

    smpl = read_smpl_file(smpl_path)
    # obtain the transformation parameters for linear blend skinning
    A, R, Th, joints, parents, v_shaped = get_transform_params(smpl, params)

    parent_path = os.path.join(lbs_root, 'parents.npy')
    np.save(parent_path, parents)
    joint_path = os.path.join(lbs_root, 'joints.npy')
    np.save(joint_path, joints)

    # transform points from the world space to the pose space
    pxyz = np.dot(vertices - Th, R)
    
    if args.smpl == 'smpl' and ('SMPLH' in smpl_path or 'smplh' in smpl_path): 
        weight_smpl_path = f'data/smpl-all/{args.smpl}/{"SMPLH_male.pkl" if "smplh" in args.smpl else "SMPL_NEUTRAL.pkl"}'  # used in EasyMocap
        smpl = read_smpl_file(weight_smpl_path)
    bweights = smpl['weights'][..., :n_bones]
    bweights = bweights / bweights.sum(axis=-1, keepdims=True)  # FIXME: hand weight assignment not right
    
    A = np.dot(bweights, A.reshape(A.shape[0], -1)).reshape(-1, 4, 4)
    can_pts = pxyz - A[:, :3, 3]
    R_inv = np.linalg.inv(A[:, :3, :3])
    pxyz = np.sum(R_inv * can_pts[:, None], axis=2)

    # calculate big pose
    poses = params['poses'].reshape(-1, 3).astype(np.float32)
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
    big_A = get_rigid_transformation(rot_mats, joints, parents)
    big_A = np.dot(bweights, big_A.reshape(big_A.shape[0], -1)).reshape(-1, 4, 4)

    bigpose_vertices = np.sum(big_A[:, :3, :3] * pxyz[:, None], axis=2)
    bigpose_vertices = bigpose_vertices + big_A[:, :3, 3]

    bigpose_vertices_path = os.path.join(lbs_root, 'bigpose_vertices.npy')
    np.save(bigpose_vertices_path, bigpose_vertices)

    faces_path = os.path.join(lbs_root, 'faces.npy')
    np.save(faces_path, faces)

    return bweights


def get_all_params():
    all_params = {}
    for i in tqdm.tqdm(range(args.begin_frame, len(os.listdir(param_dir)))):
        param_path = os.path.join(param_dir, '{}.npy'.format(i))
        params = np.load(param_path, allow_pickle=True).item()
        for k in params:
            if k not in all_params:
                all_params[k] = []
            all_params[k].append(params[k])
    for k in all_params:
        all_params[k] = np.concatenate(all_params[k], axis=0)
    all_params_path = os.path.join(lbs_root, 'smpl_params.npy')
    np.save(all_params_path, all_params)


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/light_stage')
parser.add_argument('--cfg_model', type=str, default='output/cfg_model.yml')
parser.add_argument('--humans', type=str, nargs='+', default=['CoreView_377'])
parser.add_argument('--begin_frame', type=int, default=0)
parser.add_argument('--smpl', type=str, default='smpl', choices=['smplh', 'smpl'])
parser.add_argument('--lbs', type=str, default='lbs')
parser.add_argument('--params', type=str, default='new_params')
parser.add_argument('--vertices', type=str, default='new_vertices')
args = parser.parse_args()

smpl_path = ''
for human_ind in range(len(args.humans)):
    try:
        human = args.humans[human_ind]
        lbs_root = os.path.join(args.data_dir, human, args.lbs)
        param_dir = os.path.join(args.data_dir, human, args.params)
        vertices_dir = os.path.join(args.data_dir, human, args.vertices)
        cfg_model_path = os.path.join(args.data_dir, human, args.cfg_model)

        if os.path.exists(cfg_model_path):
            from easymocap.config.baseconfig import Config
            smpl_path = Config.load(cfg_model_path).args.model_path  # this will still load smplh if its a smpl
        if not os.path.exists(smpl_path):
            smpl_path = f'data/smplx/{args.smpl}/{"SMPLH_male.pkl" if "smplh" in args.smpl else "SMPL_NEUTRAL.pkl"}'  # used in EasyMocap

        if args.smpl == 'smpl':
            n_bones = 24
        else:
            n_bones = 52

        os.system('mkdir -p {}'.format(lbs_root))

        get_bigpose_blend_weights()
        get_tpose_blend_weights()
        get_all_params()
    except:
        import traceback
        print(traceback.format_exc())
        continue

# python tools/prepare_param_verts.py --data_dir $data --humans $(ls $data) --easy_output easymocap/output-smpl-3d --smpl smpl --use_easymocap_vertices --params params --vertices vertices && python tools/prepare_lbs_meta.py --data_dir $data --cfg_model easymocap/output-smpl-3d/cfg_model.yml --humans $(ls $data) --params params --vertices vertices --lbs lbs && python tools/prepare_h36m_uv.py --data_dir $data --humans lan_images620_1300 marc_images35000_36200 --begin_frame 0 --lbs lbs --vertices vertices --params params --cfg_model easymocap/output-smpl-3d/cfg_model.yml && python tools/prepare_h36m_uv.py --data_dir $data --humans olek_images0812 vlad_images1011 --begin_frame 1 --lbs lbs --vertices vertices --params params --cfg_model easymocap/output-smpl-3d/cfg_model.yml
