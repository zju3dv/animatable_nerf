import torch


# (batch, c, h, w)
def instant_norm_2d(data):

    channels = data.size(1)
    batch = data.size(0)
    h = data.size(2)
    w = data.size(3)
    t = data.view(batch, channels, -1)
    x_var, x_mean = torch.std_mean(
        t + 1e-4 * torch.randn(t.size(), device=t.device), dim=2, keepdim=True)

    x_mean = x_mean.unsqueeze(2).repeat(1, 1, h, w)
    x_var = x_var.unsqueeze(2).repeat(1, 1, h, w)

    return (data - x_mean) / (x_var + 1e-6)


# (batch,num, c)
def instant_norm_1d(data):
    batch = data.size(0)
    num = data.size(1)
    channels = data.size(2)
    x_var, x_mean = torch.std_mean(data, dim=1, keepdim=True)

    x_mean = x_mean.repeat(1, num, 1)
    x_var = x_var.repeat(1, num, 1)

    return (data - x_mean) / (x_var + 1e-6)
