import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, num_channels):
        super(Generator, self).__init__()
        self.block1 = ConvBlock(num_channels, 32, 9, 1, 4, nn.LeakyReLU(0.2))
        self.block2 = ConvBlock(32, 64, 3, 1, 1, nn.LeakyReLU(0.2))
        self.block3 = ConvBlock(64, 128, 3, 1, 1, nn.LeakyReLU(0.2))
        self.block4 = ResidualBlock(128)
        self.block5 = ResidualBlock(128)
        self.block6 = ResidualBlock(128)
        self.block7 = ResidualBlock(128)
        self.block8 = ConvBlock(128, 64, 3, 1, 1, nn.LeakyReLU(0.2))
        self.block9 = ConvBlock(64, 32, 3, 1, 1, nn.LeakyReLU(0.2))
        self.block10 = ConvBlock(32, num_channels, 9, 1, 4, nn.Tanh())

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        x7 = self.block7(x6)
        x8 = self.block8(x7) + x2
        x9 = self.block9(x8) + x1
        x10 = self.block10(x9) + x
        return (x10 + 1) / 2


class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        self.seq = nn.Sequential(
            ConvBlock(num_channels, 64, 4, 2, 2, nn.LeakyReLU(0.2)),
            ConvBlock(64, 128, 4, 2, 2, nn.LeakyReLU(0.2)),
            ConvBlock(128, 256, 4, 2, 2, nn.LeakyReLU(0.2)),
            ConvBlock(256, 512, 4, 1, 2, nn.LeakyReLU(0.2)),
            nn.AdaptiveAvgPool2d(1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.seq(x)
        # return batch size length 1-d tensor
        return x.view(x.shape[0])


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, act_function):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_function

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual
