import torch
from torch import nn
import torchvision

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg_network = nn.Sequential(*vgg.features).eval()
        self.mse_loss = nn.MSELoss()
        self.tv_loss = 

    def forward(self, fake_out, fake_img, real_img):
        vgg_loss = self.
        adversarial_loss = -torch.log(fake_out)
        mse_loss = self.mse_loss(fake_img, real_img)
        tv_loss = self.tv_loss(fake_img, real_img)



        
class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, fake_out, real_out):
        return 1 - (torch.log(real_out) + torch.log(1 - fake_out))