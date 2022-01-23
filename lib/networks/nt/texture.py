import torch
import torch.nn as nn
import torch.nn.functional as F


class Texture(nn.Module):
    def __init__(self, W, H, feature_dim):
        super(Texture, self).__init__()
        self.layer1 = nn.Parameter(torch.randn(1, feature_dim, W, H))
        self.layer2 = nn.Parameter(torch.randn(1, feature_dim, W // 2, H // 2))
        self.layer3 = nn.Parameter(torch.randn(1, feature_dim, W // 4, H // 4))
        self.layer4 = nn.Parameter(torch.randn(1, feature_dim, W // 8, H // 8))

    def forward(self, x, x_msk):
        batch = x.shape[0]
        x = x * 2.0 - 1.0
        y1 = F.grid_sample(self.layer1.repeat(batch, 1, 1, 1),
                           x,
                           align_corners=True)
        y2 = F.grid_sample(self.layer2.repeat(batch, 1, 1, 1),
                           x,
                           align_corners=True)
        y3 = F.grid_sample(self.layer3.repeat(batch, 1, 1, 1),
                           x,
                           align_corners=True)
        y4 = F.grid_sample(self.layer4.repeat(batch, 1, 1, 1),
                           x,
                           align_corners=True)
        y = y1 + y2 + y3 + y4
        y = y * x_msk[:, None]
        return y
