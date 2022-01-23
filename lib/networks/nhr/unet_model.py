# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import sys
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 layers,
                 use_maks,
                 no_modified=False):
        super(UNet, self).__init__()

        if len(layers) != 9:
            print('error on layers definination.')
            sys.exit(0)
        self.layers = layers

        self.inc = inconv(n_channels, layers[0], no_modified=no_modified)
        self.down1 = down(layers[0], layers[1], no_modified=no_modified)
        self.down2 = down(layers[1], layers[2], no_modified=no_modified)
        self.down3 = down(layers[2], layers[3], no_modified=no_modified)
        self.down4 = down(layers[3], layers[4], no_modified=no_modified)
        self.up1 = up(layers[3] + layers[4],
                      layers[5],
                      no_modified=no_modified)
        self.up2 = up(layers[5] + layers[2],
                      layers[6],
                      no_modified=no_modified)
        self.up3 = up(layers[6] + layers[1],
                      layers[7],
                      no_modified=no_modified)
        self.up4 = up(layers[7] + layers[0],
                      layers[8],
                      no_modified=no_modified)

        self.use_maks = use_maks

        if use_maks:
            self.outc = outconv(layers[8], n_classes + 1)
        else:
            self.outc = outconv(layers[8], n_classes)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        #x = torch.sigmoid(x)

        if self.use_maks:
            x[:, 3, :, :] = torch.sigmoid(x[:, 3, :, :])

        return x
