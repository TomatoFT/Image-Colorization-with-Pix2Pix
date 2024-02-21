import os
from glob import glob
from statistics import mean
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from IPython import display
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from methods.Pix2Pix.constants.config import Hyperparamater


def read_path(filepath) -> List[str]:
    root_path = "/kaggle/working/Image-Colorization-with-Pix2Pix/dataset/dataset"
    path = os.path.join(root_path, filepath)
    dataset = []
    for p in glob(path+"/"+"*.jpg"):
        dataset.append(p)
    return dataset


class Transform():
    def __init__(self, resize=Hyperparamater.RESIZE, 
                 mean=Hyperparamater.MEAN, 
                 std=Hyperparamater.STD):
        self.data_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img: Image.Image):
        return self.data_transform(img)


class Dataset(object):
    def __init__(self, files: List[str]):
        self.files = files
        self.trasformer = Transform()

    def _separate(self, img) -> Tuple[Image.Image, Image.Image]:
        img = np.array(img, dtype=np.uint8)
        h, w, _ = img.shape
        w = int(w/2)
        return Image.fromarray(img[:, w:, :]), Image.fromarray(img[:, :w, :])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.files[idx])
        input, output = self._separate(img)
        input_tensor = self.trasformer(input)
        output_tensor = self.trasformer(output)
        return input_tensor, output_tensor

    def __len__(self):
        return len(self.files)

def show_img_sample(img: torch.Tensor, img1: torch.Tensor):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    ax = axes.ravel()
    ax[0].imshow(img.permute(1, 2, 0))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("input image", c="g")
    ax[1].imshow(img1.permute(1, 2, 0))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("label image", c="g")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

