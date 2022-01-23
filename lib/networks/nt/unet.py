import torch
import torch.nn as nn
import torch.nn.functional as F


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, output_pad=0, concat=True, final=False):
        super(up, self).__init__()
        self.concat = concat
        self.final = final
        if self.final:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad),
                nn.Tanh()
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x1, x2):
        if self.concat:
            x1 = torch.cat((x2, x1), dim=1)
        x1 = self.conv(x1)
        return x1


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()
        self.down1 = down(input_channels, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)
        self.up1 = up(512, 512, concat=False)
        self.up2 = up(1024, 512)
        self.up3 = up(768, 256)
        self.up4 = up(384, 128)
        self.up5 = up(192, output_channels, final=True)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, None)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        return x
