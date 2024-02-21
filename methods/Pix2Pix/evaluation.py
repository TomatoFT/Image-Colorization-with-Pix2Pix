from glob import glob
from statistics import mean
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from config.constants import DEVICE, MEAN, STD
from IPython import display
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


def de_norm(img):
    img_ = img.mul(torch.FloatTensor(STD).view(3, 1, 1))
    img_ = img_.add(torch.FloatTensor(MEAN).view(3, 1, 1)).detach().numpy()
    img_ = np.transpose(img_, (1, 2, 0))
    return img_

def evaluate(val_dl, name, G):
    with torch.no_grad():
        fig, axes = plt.subplots(6, 5, figsize=(8, 12))
        ax = axes.ravel()
#         G = load_model(name)
        for input_img, real_img in tqdm(val_dl):
            input_img = input_img.to(DEVICE)
            real_img = real_img.to(DEVICE)

            fake_img = G(input_img)
            batch_size = input_img.size()[0]
            batch_size_2 = batch_size * 2

            for i in range(batch_size):
                ax[i].imshow(input_img[i].permute(1, 2, 0))
                ax[i+batch_size].imshow(de_norm(real_img[i]))
                ax[i+batch_size_2].imshow(de_norm(fake_img[i]))
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                ax[i+batch_size].set_xticks([])
                ax[i+batch_size].set_yticks([])
                ax[i+batch_size_2].set_xticks([])
                ax[i+batch_size_2].set_yticks([])
                if i == 0:
                    ax[i].set_ylabel("Input Image", c="g")
                    ax[i+batch_size].set_ylabel("Real Image", c="g")
                    ax[i+batch_size_2].set_ylabel("Generated Image", c="r")
            plt.subplots_adjust(wspace=0, hspace=0)
            break

