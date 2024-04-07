import numpy as np
import torch
from config.constants import BATCH_SIZE, EPOCH
from dataloader import Dataset, read_path
from evaluation import evaluate
from model import Discriminator, Generator
from torch.utils.data import DataLoader
from train import load_model, train_loop, train_show_img

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
trained_G, trained_D = train_loop(train_dl, G, D, EPOCH)

G = load_model(45)

# train_show_img(1, trained_G)
evaluate(val_dl, 45, G)

print("Done in implementation in the whole Pix2Pix progress")