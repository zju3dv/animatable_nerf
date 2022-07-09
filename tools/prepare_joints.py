import os
import numpy as np
import cv2
import sys
sys.path.append('.')
sys.path.append('..')
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils


def get_joints(frame_index):
    inds = os.listdir(param_root)
    inds = sorted([int(ind[:-4]) for ind in inds])
    frame_index = inds[frame_index]

    # transform smpl from the world coordinate to the smpl coordinate
    params_path = os.path.join(param_root, '{}.npy'.format(frame_index))
    params = np.load(params_path, allow_pickle=True).item()
    Rh = params['Rh'].astype(np.float32)
    Th = params['Th'].astype(np.float32)

    # prepare sp input of param pose
    R = cv2.Rodrigues(Rh)[0].astype(np.float32)

    # calculate the skeleton transformation
    poses = params['poses'].reshape(-1, 3)
    A, canonical_joints = if_nerf_dutils.get_rigid_transformation(
        poses, joints, parents, return_joints=True)

    posed_joints = np.dot(canonical_joints, R.T) + Th

    return posed_joints


# data_path = 'data/h36m/{}/Posing'
# human_frames = [['S1', 150], ['S5', 250], ['S6', 150], ['S7', 300], ['S8', 250], ['S9', 260], ['S11', 200]]
# frame_interval = 5

# data_path = 'data/deepcap/{}'
# human_frames = [['lan_images620_1300', 300], ['marc_images35000_36200', 300], ['olek_images0812', 300], ['vlad_images1011', 300]]
# frame_interval = 1

data_path = 'data/light_stage/CoreView_{}'
# human_frames = [['313', 60], ['315', 400], ['377', 300], ['386', 300], ['387', 300], ['392', 300], ['393', 300], ['394', 300]]
human_frames = [['390', 300]]
frame_interval = 1

for human_frame in human_frames:
    data_root = data_path.format(human_frame[0])
    num_train_frame = human_frame[1]

    lbs_root = os.path.join(data_root, 'lbs')
    joints = np.load(os.path.join(lbs_root, 'joints.npy'))
    joints = joints.astype(np.float32)
    parents = np.load(os.path.join(lbs_root, 'parents.npy'))
    param_root = os.path.join(data_root, 'new_params')

    training_joints = []
    for i in range(0, num_train_frame * frame_interval, frame_interval):
        posed_joints = get_joints(i)
        training_joints.append(posed_joints)
    training_joints = np.stack(training_joints)

    np.save(os.path.join(lbs_root, 'training_joints.npy'), training_joints)
