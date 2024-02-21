from enum import Enum

import torch

MEAN = (0.5, 0.5, 0.5,)
STD = (0.5, 0.5, 0.5,)
RESIZE = 64
BATCH_SIZE = 16

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

