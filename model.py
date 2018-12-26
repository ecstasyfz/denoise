import torch
import torch.nn as nn
import torch.nn.functional as F


# class Generator(nn.Module):
#     def __init__(self, num_channels):
#         super(Generator, self).__init__()
#         # self.block1 = ConvBlock(num_channels, 32, 9, 1, 4, nn.LeakyReLU(0.2))
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 32, 9, 1, padding=4),
#             # nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2)
#         )
#         self.block2 = ConvBlock(32, 64, 3, 1, 1, nn.LeakyReLU(0.2))
#         self.block3 = ConvBlock(64, 128, 3, 1, 1, nn.LeakyReLU(0.2))
#         self.block4 = ResidualBlock(128)
#         self.block5 = ResidualBlock(128)
#         self.block6 = ResidualBlock(128)
#         self.block7 = ResidualBlock(128)
#         self.block8 = ConvBlock(128, 64, 3, 1, 1, nn.LeakyReLU(0.2))
#         self.block9 = ConvBlock(64, 32, 3, 1, 1, nn.LeakyReLU(0.2))
#         # self.block10 = ConvBlock(32, num_channels, 9, 1, 4, nn.Tanh())
#         self.block10 = nn.Sequential(
#             nn.Conv2d(32, 32, 3, 1, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 3, 3, 1, padding=1),
#         )

#     def forward(self, x):
#         x1 = self.block1(x)
#         x2 = self.block2(x1)
#         x3 = self.block3(x2)
#         x4 = self.block4(x3)
#         x5 = self.block5(x4)
#         x6 = self.block6(x5)
#         x7 = self.block7(x6)
#         x8 = self.block8(x7)
#         x9 = self.block9(x8)
#         x10 = self.block10(x9) + x
#         return (F.tanh(x10) + 1) / 2

# 下面是抄的ImagingDenosie
class Generator(nn.Module):
    def __init__(self, num_channels):
        super(Generator, self).__init__()
        self.block1 = ConvBlock(3, 32, 9, 1, 4, nn.LeakyReLU(0.2))
        self.block2 = ConvBlock(32, 64, 3, 1, 1, nn.LeakyReLU(0.2))
        self.block3 = ConvBlock(64, 128, 3, 1, 1, nn.LeakyReLU(0.2))
        self.block4 = ResidualBlock(128)
        self.block5 = ResidualBlock(128)
        self.block6 = ResidualBlock(128)
        self.block7 = ConvBlock(128, 64, 3, 1, 1, nn.LeakyReLU(0.2))
        self.block8 = ConvBlock(64, 32, 3, 1, 1, nn.LeakyReLU(0.2))
        self.block9 = ConvBlock(32, 3, 9, 1, 4, nn.Tanh())

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)

        x7 = self.block7(x6)
        x8 = self.block8(x7)

        x8 = x7 + x8
        x9 = self.block9(x8) + x
        return (x9+1)/2


class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        # self.seq = nn.Sequential(
        #     ConvBlock(num_channels, 64, 4, 2, 1, nn.LeakyReLU(0.2)),
        #     ConvBlock(64, 128, 4, 2, 1, nn.LeakyReLU(0.2)),
        #     ConvBlock(128, 256, 4, 2, 1, nn.LeakyReLU(0.2)),
        #     ConvBlock(256, 512, 4, 1, 1, nn.LeakyReLU(0.2)),
        #     ConvBlock(512, 1, 4, 1, 1, nn.LeakyReLU(0.2)),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Sigmoid()
        # )
        self.seq = nn.Sequential(
            ConvBlock(3, 64, 3, 1, 1, nn.LeakyReLU(0.2)),
            ConvBlock(64, 64, 3, 2, 1, nn.LeakyReLU(0.2)),
            ConvBlock(64, 128, 3, 1, 1, nn.LeakyReLU(0.2)),
            ConvBlock(128, 128, 3, 2, 1, nn.LeakyReLU(0.2)),
            ConvBlock(128, 256, 3, 1, 1, nn.LeakyReLU(0.2)),
            ConvBlock(256, 256, 3, 2, 1, nn.LeakyReLU(0.2)),
            ConvBlock(256, 512, 3, 1, 1, nn.LeakyReLU(0.2)),
            ConvBlock(512, 512, 3, 2, 1, nn.LeakyReLU(0.2)),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
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
        self.block1 = ConvBlock(channels, channels, 3, 1, 1, nn.LeakyReLU(0.2))
        self.block2 = ConvBlock(channels, channels, 3, 1, 1, nn.LeakyReLU(0.2))

    def forward(self, x):
        residual = self.block1(x)
        residual = self.block2(residual)
        return x + residual
