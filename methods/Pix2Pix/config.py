from enum import Enum
import torch

class Hyperparamater(Enum):
    MEAN = (0.5, 0.5, 0.5,)
    STD = (0.5, 0.5, 0.5,)
    RESIZE = 64
    BATCH_SIZE = 16

class Device(Enum):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

