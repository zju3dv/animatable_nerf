import torch
import torch.nn.functional as F
import torchvision.models.vgg as vgg
from collections import namedtuple


def gradient_1order(x, h_x=None, w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow(
        torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
    return xgrad


LossOutput = namedtuple("LossOutput", ["relu1", "relu2"])

#LossOutput = namedtuple(
#    "LossOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])


class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """
    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        '''
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }
        '''

        self.layer_name_mapping = {'3': "relu1", '8': "relu2"}

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
            if name == '8':
                break
        return LossOutput(**output)


class Perceptual_loss(torch.nn.Module):
    def __init__(self):
        super(Perceptual_loss, self).__init__()

        #self.model = models_lpf.resnet50(filter_size = 5)
        #self.model.load_state_dict(torch.load('/data/wmy/NR/models/resnet50_lpf5.pth.tar')['state_dict'])
        self.model = LossNetwork()
        self.model.cuda()
        self.model.eval()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, x, target):
        x_feature = self.model(x[:, 0:3, :, :])
        target_feature = self.model(target[:, 0:3, :, :])

        if x.size(1) > 3:
            x_mask_feature = self.model(x[:, 3:4, :, :].repeat(1, 3, 1, 1))
            target_mask_feature = self.model(target[:, 3:4, :, :].repeat(
                1, 3, 1, 1))

        #xgrad_x = gradient_1order(x[:,0:3,:,:])
        #xgrad_target = gradient_1order(target[:,0:3,:,:])

        feature_loss = (self.loss(x_feature.relu1, target_feature.relu1) +
                        self.loss(x_feature.relu2, target_feature.relu2)) / 2.0

        if x.size(1) > 3:
            feature_loss = feature_loss + (self.loss(
                x_mask_feature.relu1, target_mask_feature.relu1) + self.loss(
                    x_mask_feature.relu2, target_mask_feature.relu2)) / 2.0

        return feature_loss, self.loss(x, target), torch.Tensor([0])
