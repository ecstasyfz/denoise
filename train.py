# -*- coding: utf-8 -*-
import os
import argparse
from datetime import datetime
from math import log10
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_ssim
from loss import *
from model import *
from utils import *


def main():
    # hyper parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    NUM_WORKERS = 4
    CROP_SIZE = 64
    NUM_CHANNELS = 3
    LR = 0.002

    # create output path
    now = datetime.now().strftime(r'%g%m%d_%H%M%S')
    output_path = os.path.join('train_output', now)
    mkdir('train_output')
    mkdir(output_path)
    mkdir(os.path.join(output_path, 'images'))
    mkdir(os.path.join(output_path, 'params'))
    mkdir(os.path.join(output_path, 'plot'))

    # decide device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create the dataset
    train_set = TrainDatasetFromFolder(root='data/VOC2012/train', crop_size=CROP_SIZE, sigma=20)
    val_set = ValDatasetFromFolder(root='data/VOC2012/val')
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=True, num_workers=NUM_WORKERS)

    # create the networks
    netG = Generator(NUM_CHANNELS).to(device)
    netD = Discriminator(NUM_CHANNELS).to(device)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    # create the loss function
    generator_criterion = GeneratorLoss().to(device)
    discriminator_criterion = DiscriminatorLoss().to(device)

    # create the optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=LR)
    optimizerD = optim.Adam(netD.parameters(), lr=LR)

    # create the dataforms
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    train_columns = ['epoch', 'batch', 'G_loss', 'D_loss', 'D(real)', 'D(fake)']
    val_columns = ['epoch', 'filename', 'sigma', 'PSNR', 'SSIM']

    # main loop
    print('Starting Training Loop...')
    for epoch in range(NUM_EPOCHS):
        # create empty epoch results
        train_result = []
        val_result = []
        # train part
        train_bar = tqdm(train_loader)
        # set train mode
        netG.train()
        netD.train()
        # train loop
        for batch, (real_image, noisy_image) in enumerate(train_bar):
            # move to GPU
            real_image = real_image.to(device)
            noisy_image = noisy_image.to(device)
            # update D network
            netD.zero_grad()

            fake_image = netG(noisy_image)
            fake_out = netD(fake_image).mean()
            real_out = netD(real_image).mean()

            D_loss = discriminator_criterion(fake_out, real_out)
            D_loss.backward(retain_graph=True)

            optimizerD.step()

            # updata G network
            netG.zero_grad()

            G_loss = generator_criterion(fake_out, fake_image, real_image)
            G_loss.backward()

            optimizerG.step()

            # convert scalar tensor
            G_loss = G_loss.item()
            D_loss = D_loss.item()
            real_out = real_out.item()
            fake_out = fake_out.item()

            # append result
            train_result.append([epoch, batch, G_loss, D_loss, real_out, fake_out])

            # update progress bar
            train_bar.set_description(
                desc='[Train %d/%d] | G_loss: %.4f | D_loss: %.4f | D(fake): %.4f | D(real): %.4f' % (
                    epoch, NUM_EPOCHS, G_loss, D_loss, fake_out, real_out))

        # stop grad for validation
        with torch.no_grad():
            # validation part
            val_bar = tqdm(val_loader)
            # set validation mode
            netG.eval()
            netD.eval()
            # validation loop
            for filename, images in val_bar:
                # move tensors to GPU
                ground_images = images.to(device).squeeze(0)
                # excess the network
                denoised_images = netG(ground_images)
                # chunk tensor
                ground_images = ground_images.chunk(6)
                denoised_images = denoised_images.chunk(6)
                # measure
                psnrs = [10*log10(1/((denoised_images[i]-ground_images[0])**2).mean()) for i in range(6)]
                ssims = [pytorch_ssim.ssim(ground_images[0], denoised_images[i]).item() for i in range(6)]
                # move tensors back to CPU
                ground_images = [image.cpu() for image in ground_images]
                denoised_images = [image.cpu() for image in denoised_images]
                # generate imaging to output/images
                filename = filename[0] # filename return from loader is a tuple
                png_name = filename[:-len(filename.split('.')[-1])-1] + '.png' # jpg is not supported by torchvision
                print(ground_images[0].shape)
                save_validation_images(os.path.join(output_path, 'images', 'epoch_%d_%s' % (epoch, png_name)), ground_images, denoised_images, psnrs, ssims)
                # append result
                for i in range(6):
                    val_result.append([epoch, filename, i * 10, psnrs[i], ssims[i]])

                # update progress bar
                val_bar.set_description(
                    desc='[Validation for sigma=20 %d/%d] | PSNR: %.4f | SSIM: %.4f' % (
                        epoch, NUM_EPOCHS, psnrs[2], ssims[2]))


        # save part
        # save model parameters
        torch.save(netG.state_dict(), os.path.join(output_path, 'params', 'netG_epoch_%d.pth' % epoch))
        torch.save(netD.state_dict(), os.path.join(output_path, 'params', 'netD_epoch_%d.pth' % epoch))
        # update dateframe
        train_df = train_df.append(pd.DataFrame(train_result, columns=train_columns))
        val_df = val_df.append(pd.DataFrame(val_result, columns=val_columns))

    # end loop
    # save csv
    train_df.set_index('epoch', inplace=True)
    val_df.set_index('epoch', inplace=True)
    train_df.to_csv(os.path.join(output_path, 'train_result.csv'))
    val_df.to_csv(os.path.join(output_path, 'validation_result.csv'))

    # append finished task name
    open('finished_tasks.txt', 'a').write(now + '\n')

    print('Training Finished!')


if __name__ == '__main__':
    main()
