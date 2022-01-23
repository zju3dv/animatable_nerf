import imp
from lib.config import cfg
import os
if cfg.train.use_vgg:
    from .perceptual_loss import VGGPerceptualLoss
from .ssim import SSIMLoss
from .discriminator import GANLoss, NLayerDiscriminator
# def get_loss(name):
#     path = os.path.join('lib/losses', cfg.task, name+'.py')
#     module = '.{}.{}'.format(cfg.task, name)
#     loss = imp.load_source(module, path).Loss()
#     return loss
