import torch, pickle
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class Model(nn.Module):
    def __init__(self, inputs, outputs):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(12,20)
        self.linear2 = nn.Linear(20, 8)

    def forward(self, x):
        x = torch.reshape(x, (-1,))
        x = self.linear1(x)
        out = self.linear2(x)
        return out