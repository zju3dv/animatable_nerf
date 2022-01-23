import numpy as np
from lib.config import cfg
from skimage.measure import compare_ssim
import os
import cv2


class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, rgb_pred, rgb_gt, batch):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(batch['H']), int(batch['W'])
        mask_at_box = mask_at_box.reshape(H, W)
        # convert the pixels into an image
        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box == 1] = rgb_pred
        img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box == 1] = rgb_gt
        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = img_pred[y:y + h, x:x + w]
        img_gt = img_gt[y:y + h, x:x + w]
        # compute the ssim
        ssim = compare_ssim(img_pred, img_gt, multichannel=True)
        return ssim

    def evaluate(self, output, batch):
        msk = output['mask'][0]
        img_pred = (output['rgb'][0] * msk).permute(1, 2, 0).detach().cpu().numpy()
        img_gt = (batch['img'][0] * msk).permute(1, 2, 0).detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        rgb_pred = img_pred[mask_at_box == 1]
        rgb_gt = img_gt[mask_at_box == 1]

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
        self.ssim.append(ssim)
        result_dir = os.path.join('data/result/deform', cfg.exp_name)
        frame_index = batch['i'].item()
        view_index = batch['cam_ind'].item()
        if os.path.exists(result_dir) == False:
            os.system('mkdir -p {}'.format(result_dir))
        result_dir = os.path.join('data/result/deform', cfg.exp_name,'comparison')
        img_path = os.path.join(result_dir, 'frame{:04d}_view{:04d}.png'.format(frame_index,
                                                  view_index))
        img_gt_path = os.path.join(result_dir, 'frame{:04d}_view{:04d}_gt.png'.format(frame_index,
                                                  view_index))
        cv2.imwrite(img_path, (img_pred * 255)[..., [2, 1, 0]])
        cv2.imwrite(img_gt_path, (img_gt * 255)[..., [2, 1, 0]])

    def summarize(self):
        result_path = os.path.join(cfg.result_dir, 'metrics.npy')
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim}
        np.save(result_path, self.mse)
        print('mse: {}'.format(np.mean(self.mse)))
        print('psnr: {}'.format(np.mean(self.psnr)))
        print('ssim: {}'.format(np.mean(self.ssim)))
        self.mse = []
        self.psnr = []
        self.ssim = []
