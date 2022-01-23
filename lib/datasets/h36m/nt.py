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
                test_view = [3]
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
        self.nrays = cfg.N_rand

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index])[:-4] + '.png'
        msk = imageio.imread(msk_path)
        msk = (msk != 0).astype(np.uint8)
        return msk

    def get_uvmap(self, index, cam):
        ind = int(os.path.basename(self.ims[index])[:-4])
        uv_path = os.path.join(os.path.join(self.data_root, 'uv/{}'.format(cfg.exp_name[-3:])), 'frame{:04d}_view{:04d}.png'.format(int(ind), int(cam)))
        uv = imageio.imread(uv_path)
        uv = (uv / 255.).astype(np.float32)
        uv = cv2.resize(uv, (1000, 1002), interpolation=cv2.INTER_AREA)
        uv = uv[..., :2]
        uv_msk = (np.sum(uv, axis=2) != 0).astype(np.float32)
        return uv, uv_msk

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # transformation augmentation
        xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        nxyz = nxyz.astype(np.float32)
        feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans

    def random_crop(self, img, msk, uv, uv_msk, crop_size):
        h, w = img.shape[:2]
        crop_h = crop_size
        crop_w = crop_size
        yx = np.argwhere(msk)
        yx = yx[np.random.randint(0, len(yx))]
        h1 = min(yx[0], h - crop_h)
        w1 = min(yx[1], w - crop_w)
        img = img[h1:h1 + crop_h, w1:w1 + crop_w, :]
        msk = msk[h1:h1 + crop_h, w1:w1 + crop_w]
        uv = uv[h1:h1 + crop_h, w1:w1 + crop_w, :]
        uv_msk = uv_msk[h1:h1 + crop_h, w1:w1 + crop_w]
        return img, msk, uv, uv_msk

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk = self.get_mask(index)
        cam_ind = self.cam_inds[index]
        uv, uv_msk = self.get_uvmap(index, cam_ind)

        img = cv2.resize(img, (1000, 1002), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (1000, 1002), interpolation=cv2.INTER_NEAREST)

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
        uv = cv2.resize(uv, (W, H), interpolation=cv2.INTER_AREA)
        uv_msk = cv2.resize(uv_msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
        K[:2] = K[:2] * cfg.ratio

        i = int(os.path.basename(img_path)[:-4])
        feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans = self.prepare_input(
            i)
        pose = np.concatenate([R, T], axis=1)
        mask_at_box = if_nerf_dutils.get_bound_2d_mask(can_bounds, K, pose, H,
                                                       W)

        # if self.split == 'train':
        #     img, msk, uv, uv_msk = self.random_crop(img, msk, uv, uv_msk, 256)

        img = img.transpose(2, 0, 1)
        K = K.astype(np.float32)
        ret = {
            'img': img,
            'msk': msk,
            'uv': uv,
            'uv_msk': uv_msk,
            'mask_at_box': mask_at_box,
            'H': H,
            'W': W,
        }

        meta = {'frame_index': i, 'cam_ind': cam_ind, 'i':i}
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
