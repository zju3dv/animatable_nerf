import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
import trimesh


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        if len(cfg.test_view) == 0:
            test_view = [
                i for i in range(num_cams) if i not in cfg.training_view
            ]
            if len(test_view) == 0:
                test_view = [0]
        else:
            test_view = cfg.test_view
        view = cfg.training_view if split == 'train' else test_view

        i = cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame
        if cfg.test_novel_pose:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_eval_frame
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.num_cams = len(view)

        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))
        # self.nrays = cfg.N_rand

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, 'mask',
                self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, 'mask_cihp',
                    self.ims[index])[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        msk_cihp = (msk_cihp >125).astype(np.uint8)
        msk = msk_cihp
        return msk

    def prepare_input(self, i):
        # read xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)

        # calculate the skeleton transformation
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)

        big_poses = np.zeros_like(poses).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        big_poses = big_poses.reshape(-1, 3)
        big_A = if_nerf_dutils.get_rigid_transformation(
            big_poses, joints, parents)
        pbw = np.load(os.path.join(self.lbs_root, 'bweights/{}.npy'.format(i)))
        pbw = pbw.astype(np.float32)

        return wxyz, pxyz, A, big_A, pbw, Rh, Th

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk = self.get_mask(index)
        if len(msk.shape) == 3:
            msk = msk[:,:,0]
        msk = msk.astype(np.float32)
        H, W = img.shape[:2]
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.
        RT = np.concatenate([R, T], axis=1)
        RT = np.concatenate([RT, [[0, 0, 0, 1]]], axis=0)
        RT = np.linalg.inv(RT).astype(np.float32)

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
        K[:2] = K[:2] * cfg.ratio

        i = int(os.path.basename(img_path)[:-4])
        frame_index = i
        # feature,  can_bounds, Rh, Th = self.prepare_input(
            # i)
        wpts, ppts, A, big_A, pbw, Rh, Th = self.prepare_input(i)
        wbounds = if_nerf_dutils.get_bounds(wpts)
        vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        tpose = np.load(vertices_path).astype(np.float32)
        tbounds = if_nerf_dutils.get_bounds(tpose)
        pose = np.concatenate([R, T], axis=1)
        mask_at_box = if_nerf_dutils.get_bound_2d_mask(wbounds, K, pose, H,
                                                       W)
        tbw = np.load(os.path.join(self.lbs_root, 'bigpose_bw.npy'))
        tbw = tbw.astype(np.float32)

        img = img.transpose(2, 0, 1)
        K = K.astype(np.float32)
        ret = {
            'img': img,
            'msk': msk,
            'K': K,
            'RT': RT,
            'mask_at_box': mask_at_box,
            'big_A': big_A,
            'A': A,
            'H': H,
            'W': W,
            'tbw': tbw,
            'tbounds': tbounds,
            'tpose': tpose
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        # if cfg.test_novel_pose:
            # i = index // self.num_cams + cfg.skip_test_ni
        # else:
            # i = index // self.num_cams
        i = int(os.path.basename(img_path)[:-4])
        frame_index = i
        meta = {
            'R': R,
            'Th': Th,
            'i': i,
            'cam_ind': cam_ind,
            'frame_index': frame_index
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
