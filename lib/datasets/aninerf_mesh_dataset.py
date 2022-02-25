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
    def __init__(self, data_root, human, ann_file, split, **kwargs):
        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        view = cfg.training_view

        i = cfg.begin_ith_frame
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + cfg.num_train_frame *
                                          cfg.frame_interval]
        ])

        if cfg.vis_tpose_mesh:
            self.ims = self.ims[:1]

        self.Ks = np.array(self.cams['K'])[cfg.training_view].astype(
            np.float32)
        self.Rs = np.array(self.cams['R'])[cfg.training_view].astype(
            np.float32)
        self.Ts = np.array(self.cams['T'])[cfg.training_view].astype(
            np.float32) / 1000.
        self.Ds = np.array(self.cams['D'])[cfg.training_view].astype(
            np.float32)

        self.num_cams = 1

        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))

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

        pbw = np.load(os.path.join(self.lbs_root, 'bweights/{}.npy'.format(i)))
        pbw = pbw.astype(np.float32)

        return wxyz, pxyz, A, pbw, Rh, Th

    def get_mask(self, i, nv):
        im = self.ims[i, nv]

        msk_path = os.path.join(self.data_root, 'mask_cihp', im)[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, im.replace(
                'images', 'mask'))[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        msk = (msk_cihp != 0).astype(np.uint8)

        msk = cv2.undistort(msk, self.Ks[nv], self.Ds[nv])

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk = cv2.dilate(msk.copy(), kernel)

        return msk

    def prepare_inside_pts(self, pts, i):
        sh = pts.shape
        pts3d = pts.reshape(-1, 3)

        inside = np.ones([len(pts3d)]).astype(np.uint8)
        for nv in range(self.ims.shape[1]):
            ind = inside == 1
            pts3d_ = pts3d[ind]

            RT = np.concatenate([self.Rs[nv], self.Ts[nv]], axis=1)
            pts2d = base_utils.project(pts3d_, self.Ks[nv], RT)

            msk = self.get_mask(i, nv)
            H, W = msk.shape
            pts2d = np.round(pts2d).astype(np.int32)
            pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
            pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
            msk_ = msk[pts2d[:, 1], pts2d[:, 0]]

            inside[ind] = msk_

        inside = inside.reshape(*sh[:-1])

        return inside

    def __getitem__(self, index):
        latent_index = index
        img_path = os.path.join(self.data_root, self.ims[index][0])
        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

        vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        tpose = np.load(vertices_path).astype(np.float32)
        tbounds = if_nerf_dutils.get_bounds(tpose)
        tbw = np.load(os.path.join(self.lbs_root, 'tbw.npy'))
        tbw = tbw.astype(np.float32)

        wpts, ppts, A, pbw, Rh, Th = self.prepare_input(i)
        pbounds = if_nerf_dutils.get_bounds(ppts)
        wbounds = if_nerf_dutils.get_bounds(wpts)

        voxel_size = cfg.voxel_size
        x = np.arange(wbounds[0, 0], wbounds[1, 0] + voxel_size[0],
                      voxel_size[0])
        y = np.arange(wbounds[0, 1], wbounds[1, 1] + voxel_size[1],
                      voxel_size[1])
        z = np.arange(wbounds[0, 2], wbounds[1, 2] + voxel_size[2],
                      voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)

        inside = self.prepare_inside_pts(pts, index)

        ret = {'pts': pts}
        meta = {
            'A': A,
            'pbw': pbw,
            'tbw': tbw,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': tbounds,
            'inside': inside
        }
        ret.update(meta)

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
