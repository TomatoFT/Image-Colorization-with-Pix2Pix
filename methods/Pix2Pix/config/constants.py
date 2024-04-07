import torch

EPOCH = 45
MEAN = (0.5, 0.5, 0.5,)
STD = (0.5, 0.5, 0.5,)
RESIZE = 64
BATCH_SIZE = 16

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ROOT_PATH = './'

KAGGLE_PATH = '/kaggle/working/Image-Restoration/'