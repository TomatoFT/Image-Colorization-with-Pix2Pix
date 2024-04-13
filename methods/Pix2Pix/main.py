import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.constants import BATCH_SIZE, EPOCH
from dataloader import Dataset, read_path
from evaluation import evaluate
from model import Discriminator, Generator
from train import load_model, train_loop, train_show_img

train = read_path("train")
val = read_path("val")
test = read_path("test")

selected_test = random.sample(test, 10)
print(len(selected_test))

train_ds = Dataset(train)
val_ds = Dataset(val)
inf_ds = Dataset(selected_test)

torch.manual_seed(0)
np.random.seed(0)


train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                       shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, 
                    shuffle=False, drop_last=False)
inf_dl = DataLoader(inf_ds, batch_size=BATCH_SIZE, 
                    shuffle=False, drop_last=False)
G = Generator()
D = Discriminator()
# trained_G, trained_D = train_loop(train_dl, G, D, EPOCH)

G = load_model('')

# train_show_img(1, trained_G)
evaluate(inf_dl, '', G)

print("Done in implementation in the whole Pix2Pix progress")