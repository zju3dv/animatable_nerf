import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import os
import cv2


class Visualizer:
    def __init__(self):
        self.time = []
    def visualize(self, output, batch):
        img_pred = output['rgb'][0].permute(1, 2, 0).detach().cpu().numpy()
        mask = output['mask'][0, 0].detach().cpu().numpy()
        img_pred[mask < 0.5] = 0

        img_gt = batch['img'][0].permute(1, 2, 0).detach().cpu().numpy()

        result_dir = os.path.join('data/result/nhr', cfg.exp_name)
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        result_dir = os.path.join(result_dir, 'comparison')
        os.system('mkdir -p {}'.format(result_dir))
        img_pred = (img_pred * 255)[..., [2, 1, 0]]
        cv2.imwrite('{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                   view_index), img_pred)
        img_gt = (img_gt * 255)[..., [2, 1, 0]]
        cv2.imwrite('{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
                                                      view_index), img_gt)
        self.time.append(output['time'])
        print("avg_time:{}".format(np.array(self.time).mean().item()))
        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(img_pred)
        # ax2.imshow(img_gt)
        # plt.show()
