import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder


class Renderer:
    def __init__(self, net):
        self.net = net

    def render(self, batch):
        rgb_mask = self.net(batch)
        ret = {'rgb': rgb_mask[:, :3], 'mask': rgb_mask[:, 3:]}
        return ret
