import os
from glob import glob
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from config.constants import KAGGLE_PATH, MEAN, RESIZE, ROOT_PATH, STD
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms


def read_path(filepath) -> List[str]:
    root_path = KAGGLE_PATH + "/dataset"
    path = os.path.join(root_path, filepath)
    dataset = []
    for p in glob(path+"/"+"*.jpg"):
        dataset.append(p)
    return dataset


class Transform():
    def __init__(self, resize=RESIZE, 
                 mean=MEAN, 
                 std=STD):
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
        # Image.fromarray(np.array(img, dtype=np.uint8))
        img = np.array(img, dtype=np.uint8)
        _, w, _ = img.shape
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
    _, axes = plt.subplots(1, 2, figsize=(15, 8))
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
    plt.savefig("Image_Sample.jpg")
