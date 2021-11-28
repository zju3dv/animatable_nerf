import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import cv2
import os
from termcolor import colored
import torch


class Visualizer:
    def __init__(self):
        data_dir = 'data/pose_sequence/{}'.format(cfg.exp_name)
        print(colored('the results are saved at {}'.format(data_dir),
                      'yellow'))

    def visualize(self, output, batch):
        if torch.is_tensor(output['rgb_map']):
            rgb_pred = output['rgb_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred
        img_pred = img_pred[..., [2, 1, 0]]

        view_index = batch['view_index'].item()
        frame_root = 'data/pose_sequence/{}/view{:04d}'.format(
            cfg.exp_name, view_index)
        os.system('mkdir -p {}'.format(frame_root))
        frame_index = batch['frame_index'].item()
        cv2.imwrite(
            os.path.join(frame_root, 'frame{:04d}.png'.format(frame_index)),
            img_pred * 255)
