# -*- coding: utf-8 -*-
import argparse
import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets. transforms

from loss import *
from model import *
from utils import *

# hyper parameters
BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_WORKERS = 4
CROP_SIZE = 64
NUM_CHANNELS = 3
LR = 0.002

# decide device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# define image transform
transform = transforms.Compose([
    transforms.RandomCrop(CROP_SIZE),
    transforms.ToTensor()
])

# create the dataset
train_set = datasets.ImageFolder(root='data/VOC2012/train', transform=transform)
val_set = datasets.ImageFolder(root='data/VOC2012/val', transform=transform)
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# create the networks
netG = Generator(NUM_CHANNELS).to(device)
netD = Discriminator().to(device)

# create the loss function
generator_criterion = GeneratorLoss().to(device)
discriminator_criterion = DiscriminatorLoss().to(device)

# create the optimizers
optimizerG = optim.Adam(netG.parameters(), lr=LR)
optimizerD = optim.Adam(netD.parameters(), lr=LR)

# main loop
print('Starting Training Loop...')
for epoch in range(NUM_EPOCHS):
    print('Starting epoch %d...' % epoch)
    # train part
    train_bar = tqdm(train_loader)
    # set train mode
    netG.train()
    netD.train()
    # train loop
    for image in train_bar:
        # update D network
        netD.zero_grad()

        real_image = image.to(device)
        noised_image = add_gaussian_noise(real_image)
        fake_image = netG(noised_image)

        real_out = netD(real_image)
        fake_out = netD(fake_image)

        D_loss = discriminator_criterion(fake_image, real_image)
        D_loss.backward(retain_variables=True)

        optimizerD.step()

        # updata G network
        netG.zero_grad()

        G_loss = generator_criterion()
        G_loss.backward()

        optimizerG.step()

    # validation part
    val_bar = tqdm(val_loader)
    # set validation mode
    netG.eval()
    for image in val_bar:
        pass


    # save part
    pass