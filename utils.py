import os
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils

font = ImageFont.truetype('NotoSansMono.ttf', 35)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def add_gaussian_noise(image, sigma):
    gaussian_mat = torch.randn(image.shape[1:], device='cpu')
    gaussian_mat.mul_(sigma/255)
    gaussian_mat = torch.stack([gaussian_mat] * image.shape[0])
    result = image + gaussian_mat
    result = result.clamp(0, 1)
    return result


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, root, crop_size, sigma):
        self.crop_size = crop_size
        self.sigma = sigma
        self.filenames = [os.path.join(root, x) for x in os.listdir(root) if is_image_file(x)]

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        image = transforms.RandomCrop(self.crop_size)(image)
        image = transforms.ToTensor()(image)
        noisy_image = add_gaussian_noise(image, self.sigma)
        return image, noisy_image

    def __len__(self):
        return len(self.filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, root):
        self.filenames = [os.path.join(root, x) for x in os.listdir(root) if is_image_file(x)]

    def __getitem__(self, index):
        filename = self.filenames[index]
        image = Image.open(filename)
        image = transforms.CenterCrop(min(image.size))(image)
        image = transforms.ToTensor()(image)
        images = [image]
        for sigma in range(10, 51, 10):
            images.append(add_gaussian_noise(image, sigma))
        images = torch.stack(images)

        filename = os.path.split(filename)[-1]
        return filename, images

    def __len__(self):
        return len(self.filenames)


def save_validation_images(filename, ground_images, denoised_images, psnrs, ssims):
    layers = []
    for i in range(6):
        ground_image = transforms.ToPILImage()(ground_images[i].cpu().squeeze(0))
        denoised_image = transforms.ToPILImage()(denoised_images[i].cpu().squeeze(0))
        ground_image = transforms.Resize(400)(ground_image)
        denoised_image = transforms.Resize(400)(denoised_image)
        layer = Image.new('RGB', (400, 895), color='white')
        layer.paste(ground_image, (0, 0))
        layer.paste(denoised_image, (0, 405))
        draw = ImageDraw.Draw(layer)
        draw.rectangle((0, 400, 400, 405), fill='black')
        draw.rectangle((0, 805, 400, 810), fill='black')
        draw.text(
            xy=(90, 805),
            text='PSNR %.4f\nSSIM %.4f' % (psnrs[i], ssims[i]),
            fill='black',
            font=font)
        layers.append(transforms.ToTensor()(layer))
    layers = torch.stack(layers)
    utils.save_image(layers, filename, nrow=3, padding=5)
