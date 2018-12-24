import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, num_channels):
        super(Generator, self).__init__()
        self.seq = nn.Sequential(
            ConvBlock(num_channels, 32, 9, 4, nn.ReLU()),
            ConvBlock(32, 64, 3, 1, nn.ReLU()),
            ConvBlock(64, 128, 3, 1, nn.ReLU()),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 64, 3, 1, nn.ReLU()),
            ConvBlock(64, 32, 3, 1, nn.ReLU()),
            ConvBlock(32, num_channels, 9, 4, nn.Tanh()),
        )

    def forward(self, x):
        x = self.seq(x)
        return (x + 1) / 2


class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        self.seq = nn.Sequential(
            ConvBlock(num_channels, 64, 3, 1, nn.ReLU()),
            ConvBlock(64, 128, 3, 1, nn.ReLU()),
            ConvBlock(128, 256, 3, 1, nn.ReLU()),
            ConvBlock(256, 512, 3, 1, nn.ReLU()),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.seq(x)
        # return batch size length 1-d tensor
        return x.view(x.shape[0])


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, act_function):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
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
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual
