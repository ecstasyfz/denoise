import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def add_gaussian_noise(image):
    sigma = 50
    gaussian_mat = torch.randn(image.shape[:2], device=device)
    gaussian_mat = gaussian_mat * sigma
    gaussian_mat = gaussian_mat.expand(3, *image.shape[:2])
    result = image + gaussian_mat
    result = result.clamp(0, 255)
    return result
