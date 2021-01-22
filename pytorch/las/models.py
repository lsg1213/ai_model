import torch
import torch.nn as nn

class Listener(nn.Module):
    def __init__(self, config):
        super(Listener, self).__init__()

class Speller(nn.Module):
    def __init__(self, config):
        super(Speller, self).__init__()

class LAS(nn.Module):
    def __init__(self, config, device):
        super(LAS, self).__init__()

        self.config = config
        self.Listener = Listener(self.config)
        self.Speller = Speller(self.config)

    def forward(self, x):
        x = Listener