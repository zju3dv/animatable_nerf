import torch.nn as nn
from .texture import Texture
from lib.networks.nhr.unet_model import UNet


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        W = 1024
        H = 1024
        feature_dim = 16
        self.texture = Texture(W, H, feature_dim)
        self.unet = UNet(feature_dim,
                         3, [64, 128, 256, 512, 512, 256, 128, 64, 32],
                         use_maks=True,
                         no_modified=False)

    def forward(self, batch):
        x = self.texture(batch['uv'], batch['uv_msk'])
        res = self.unet(x)
        return res
