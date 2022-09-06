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
from lib.utils import render_utils


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        test_view = [3]
        view = cfg.training_view if split == 'train' else test_view
        self.num_cams = len(view)
        K, RT = render_utils.load_cam(ann_file)
        render_w2c = render_utils.gen_path(RT)

        i = cfg.begin_ith_frame
        self.ims = np.array([
            np.array(ims_data['ims'])[cfg.training_view]
            for ims_data in annots['ims'][:cfg.num_train_frame *
                                          cfg.frame_interval]
        ])

        self.K = K[0]
        self.render_w2c = render_w2c
        img_root = 'data/render/{}'.format(cfg.exp_name)
        # base_utils.write_K_pose_inf(self.K, self.render_w2c, img_root)

        self.Ks = np.array(K)[cfg.training_view].astype(np.float32)
        self.RT = np.array(RT)[cfg.training_view].astype(np.float32)
        self.Ds = np.array(self.cams['D'])[cfg.training_view].astype(
            np.float32)

        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))
        weights = np.load(os.path.join(self.lbs_root, 'weights.npy'))
        self.weights = weights.astype(np.float32)
        self.nrays = cfg.N_rand
        self.big_A = self.load_bigpose()
        
        if cfg.test_novel_pose or cfg.aninerf_animation:
            training_joints_path = os.path.join(self.lbs_root, 'training_joints.npy')
            if os.path.exists(training_joints_path):
                self.training_joints = np.load(training_joints_path)

    def load_bigpose(self):
        big_poses = np.zeros([len(self.joints), 3]).astype(np.float32).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        big_poses = big_poses.reshape(-1, 3)
        big_A = if_nerf_dutils.get_rigid_transformation(
            big_poses, self.joints, self.parents)
        big_A = big_A.astype(np.float32)
        return big_A

    def prepare_input(self, i):
        if self.human in ['CoreView_313', 'CoreView_315']:
            i = i + 1
        # read xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        if not os.path.exists(vertices_path):
            vertices_path = os.path.join(self.data_root, cfg.vertices,
                                         '{:06d}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        if not os.path.exists(params_path):
            params_path = os.path.join(self.data_root, cfg.params,
                                       '{:06d}.npy'.format(i))
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
        A, canonical_joints = if_nerf_dutils.get_rigid_transformation(
            poses, joints, parents, return_joints=True)

        posed_joints = np.dot(canonical_joints, R.T) + Th

        # find the nearest training frame
        if (cfg.test_novel_pose or cfg.aninerf_animation) and hasattr(self, "training_joints"):
            nearest_frame_index = np.linalg.norm(self.training_joints -
                                                 posed_joints[None],
                                                 axis=2).mean(axis=1).argmin()
        else:
            nearest_frame_index = 0

        poses = poses.ravel().astype(np.float32)

        return wxyz, pxyz, A, Rh, Th, poses, nearest_frame_index

    def get_mask(self, i):
        ims = self.ims[i]
        msks = []

        for nv in range(len(ims)):
            im = ims[nv]

            msk_path = os.path.join(self.data_root, 'mask_cihp',
                                    im)[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(self.data_root, 'mask',
                                        im)[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(
                    self.data_root, im.replace('images', 'mask'))[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(
                    self.data_root, im.replace('images', 'mask'))[:-4] + '.jpg'
            msk_cihp = imageio.imread(msk_path)
            if len(msk_cihp.shape) == 3:
                msk_cihp = msk_cihp[..., 0]
            msk_cihp = (msk_cihp != 0).astype(np.uint8)

            msk = msk_cihp.astype(np.uint8)

            K = self.Ks[nv].copy()
            K[:2] = K[:2] / cfg.ratio
            msk = cv2.undistort(msk, K, self.Ds[nv])

            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk = cv2.dilate(msk.copy(), kernel)

            msks.append(msk)

        return msks

    def __getitem__(self, index):
        view_index = index
        latent_index = cfg.begin_ith_frame
        frame_index = cfg.begin_ith_frame * cfg.frame_interval

        # read v_shaped
        if cfg.get('use_bigpose', False):
            vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        else:
            vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        tvertices = np.load(vertices_path).astype(np.float32)
        tbounds = if_nerf_dutils.get_bounds(tvertices)

        wpts, pvertices, A, Rh, Th, poses, nearest_frame_index = self.prepare_input(frame_index)

        pbounds = if_nerf_dutils.get_bounds(pvertices)
        wbounds = if_nerf_dutils.get_bounds(wpts)

        msks = self.get_mask(frame_index)

        # reduce the image resolution by ratio
        img_path = os.path.join(self.data_root, self.ims[0][0])
        img = imageio.imread(img_path)
        H, W = img.shape[:2]
        H, W = int(H * cfg.ratio), int(W * cfg.ratio)
        msks = [
            cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            for msk in msks
        ]
        msks = np.array(msks)

        K = self.K
        RT = self.render_w2c[index]
        R, T = RT[:3, :3], RT[:3, 3:]
        ray_o, ray_d, near, far, mask_at_box = if_nerf_dutils.get_rays_within_bounds(
            H, W, K, R, T, wbounds)
        # ray_o, ray_d, near, far, center, scale, mask_at_box = render_utils.image_rays(
        #         RT, K, wbounds)

        ret = {
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        # blend weight
        meta = {
            'A': A,
            'big_A': self.big_A,
            'poses': poses,
            'weights': self.weights,
            'tvertices': tvertices,
            'pvertices': pvertices,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': tbounds
        }
        ret.update(meta)

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = min(latent_index, cfg.num_train_frame - 1)
        meta = {
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': view_index
        }
        ret.update(meta)

        meta = {'msks': msks, 'Ks': self.Ks, 'RT': self.RT, 'H': H, 'W': W}
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.render_w2c)
