# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import models_lpf


class gated_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super(gated_conv, self).__init__()
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.conv2_gate = nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    padding=padding)

    def forward(self, x):

        feat = self.conv2(x)
        mask = self.conv2_gate(x)

        return torch.sigmoid(mask) * feat


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, no_modified=False):
        super(double_conv, self).__init__()

        if no_modified:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(gated_conv(in_ch, out_ch, 3, padding=1),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True),
                                      gated_conv(out_ch, out_ch, 3, padding=1),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, no_modified=False):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, no_modified)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, no_modified=False):
        super(down, self).__init__()

        if no_modified:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                double_conv(in_ch, out_ch, no_modified))
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2, stride=1),
                models_lpf.Downsample(channels=in_ch, filt_size=3, stride=2),
                double_conv(in_ch, out_ch, no_modified))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, no_modified=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, no_modified)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv2(x)
        return x1 + x2
