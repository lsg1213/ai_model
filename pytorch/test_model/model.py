import torch, pickle
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class Model(nn.Module):
    def __init__(self, inputs, outputs):
        super(Model, self).__init__()
        # self.conv1 = nn.Conv1d(12,128,kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(inputs,128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 20)
        self.linear4 = nn.Linear(20, 64)
        self.linear5 = nn.Linear(64, 128)
        self.linear6 = nn.Linear(128, outputs)

    def forward(self, x):
        
        
        # x = self.conv1(x.type(torch.float))
        x = torch.reshape(x.type(torch.float), (x.size(0),-1))
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        out = self.linear6(x)
        out = torch.reshape(out, (out.size(0), -1, 8))
        return out.type(torch.double)