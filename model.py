import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, num_channels):
        super(Generator, self).__init__()
        self.block1 = ConvBlock(num_channels, 32, 9, 4)
        self.block2 = ConvBlock(32, 64, 3, 1)
        self.block3 = ConvBlock(64, 128, 3, 1)
        self.block4 = ResidualBlock(128)
        self.block5 = ResidualBlock(128)
        self.block6 = ResidualBlock(128)
        self.block7 = ResidualBlock(128)
        self.block8 = ConvBlock(128, 64, 3, 1)
        self.block9 = ConvBlock(64, 32, 3, 1)
        self.last_conv = nn.Conv2d(32, num_channels, 9, 4)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block2(block2)
        block4 = self.block2(block3)
        block5 = self.block2(block4)
        block6 = self.block2(block5)
        block7 = self.block2(block6)
        block8 = self.block2(block7)
        block9 = self.block9(block8)
        x = 1

        return (F.tanh(x) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.seq = nn.Sequential(

        )

    def forward(self, x):
        pass


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation_function):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation_function()

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


if __name__ == '__main__':
    print(Generator())
