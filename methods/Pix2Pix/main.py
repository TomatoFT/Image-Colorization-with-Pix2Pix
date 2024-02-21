from glob import glob
from statistics import mean
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from config.constants import BATCH_SIZE
from dataloader import Dataset, read_path
from evaluation import evaluate
from IPython import display
from matplotlib import pyplot as plt
from model import Discriminator, Generator
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from train import train_loop, train_show_img

train = read_path("train")
val = read_path("val")

train_ds = Dataset(train)
val_ds = Dataset(val)

torch.manual_seed(0)
np.random.seed(0)


train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                       shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, 
                    shuffle=False, drop_last=False)

G = Generator()
D = Discriminator()
EPOCH = 30
trained_G, trained_D = train_loop(train_dl, G, D, EPOCH)

train_show_img(5, trained_G)
evaluate(val_dl, 5, trained_G)