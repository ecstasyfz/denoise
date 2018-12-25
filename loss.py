import torch
from torch import nn
import torchvision


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg_network = nn.Sequential(*vgg.features).eval()
        self.mse_loss = nn.MSELoss()
        # loss factors
        self.adversarial_loss_factor = 0.5
        self.mse_loss_factor = 1.0
        self.vgg_loss_factor = 1.0
        self.tv_loss_factor = 1e-4

    def forward(self, fake_out, fake_img, real_img):
        adversarial_loss = -torch.log(fake_out)
        mse_loss = self.mse_loss(fake_img, real_img)
        vgg_loss = self.mse_loss(
            self.vgg_network(fake_img), self.vgg_network(real_img))
        tv_loss = self.tv_loss(fake_img)

        return self.vgg_loss_factor * vgg_loss \
            + self.adversarial_loss_factor * adversarial_loss \
            + self.mse_loss_factor * mse_loss \
            + self.tv_loss_factor * tv_loss

    def tv_loss(self, x, tv_loss_weight=1):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, fake_out, real_out):
        return - torch.log(real_out) - torch.log(1 - fake_out)
