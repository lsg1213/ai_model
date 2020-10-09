import torch, pickle, torchaudio, pdb
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def cal_outputs_conv(inputs, layer):
    if len(layer.kernel_size) == 1:
        return (inputs + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
    elif len(layer.kernel_size) == 2:
        
        return ((inputs[0] + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1,
                (inputs[1] + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1)
class CombineAutoencoder(nn.Module):
    def __init__(self, inputs, outputs, inch, outch, config):
        super(CombineAutoencoder, self).__init__()
        self.config = config
        # self.conv1 = nn.Conv1d(inch, 128, 3, stride=2, padding=1)
        # self.batchnorm1 = nn.BatchNorm1d(128)
        # self.dropout1 = nn.Dropout(p=0.2)
        # self.conv2 = nn.Conv1d(128, 8, 1)
        # self.batchnorm2 = nn.BatchNorm1d(8)
        # self.dropout2 = nn.Dropout(p=0.2)
        if config.feature == 'mel':
            self.conv1 = nn.Conv2d(inch, 64, 3, padding=1)
            self.batchnorm1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.batchnorm2 = nn.BatchNorm2d(128)
        elif config.feature == 'wav':
            self.conv1 = nn.Conv1d(inch, 64, 3, padding=1)
            self.batchnorm1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
            self.batchnorm2 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

        _inputs = cal_outputs_conv(cal_outputs_conv(inputs, self.conv1), self.conv2)
        if config.feature == 'wav':
            self.back = FCAutoencoder(_inputs, config.len, self.conv2.out_channels, outch, config)
        elif config.feature == 'mel':
            # mel: inputs=(frames, 12), outputs=(window_size, 8), inch=(self.conv2 filter number), outch=(frames)
            self.back = FCAutoencoder(_inputs, (config.nfft, 8), self.conv2.out_channels, outch, config)

        if config.weight:
            with torch.no_grad():
                self.conv1.weight = torch.nn.Parameter(torch.zeros_like(self.conv1.weight) + 1e-5)
                self.conv2.weight = torch.nn.Parameter(torch.zeros_like(self.conv2.weight) + 1e-5)

    
    def forward(self, x):
        x = self.conv1(x.type(torch.float32))
        x = F.relu(self.dropout1(self.batchnorm1(x)))
        # x = F.relu(self.dropout1(x))
        x = self.conv2(x)
        x = F.relu(self.dropout2(self.batchnorm2(x)))
        # x = F.relu(self.dropout2(x))
        x = self.back(x)
        return x

class CNN(nn.Module):
    def __init__(self, inputs, outputs, inch, outch, config):
        super(CNN, self).__init__()
        self.config = config
        self.conv1 = nn.Conv1d(inch, 16, 3, padding=1).double()
        self.bn1 = nn.BatchNorm1d(16).double()
        self.do1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(16, 8, 3, padding=1).double()
        self.bn2 = nn.BatchNorm1d(8).double()
        self.do2 = nn.Dropout(0.1)
        self.li1 = nn.Linear(inputs,100,bias=True).double()
        self.li2 = nn.Linear(100,outputs,bias=False).double()

    def forward(self, x):
        x = self.conv1(x)
        x = self.do1(self.bn1(x))
        x = self.conv2(x)
        x = self.do2(self.bn2(x))

        x = torch.relu(self.li1(x))
        # x = torch.tanh(self.li2(x))
        x = self.li2(x)

        return x.transpose(1,2)

class FCAutoencoder(nn.Module):
    def __init__(self, inputs, outputs, inch, outch, config):
        super(FCAutoencoder, self).__init__()
        self.config = config
        if len(inputs) > 1:
            _in = 1
            for i in inputs:
                _in *= i
            inputs = _in
        # mel: inputs=(frames, 12), outputs=(window_size, 8), inch=(self.conv2 filter number), outch=(frames)
        # mel: (batch, 128, frames, 12)
        self.linear1 = nn.Linear(inputs * inch,256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 30)
        self.linear5 = nn.Linear(30, 64)
        self.linear6 = nn.Linear(64, 128)
        self.linear7 = nn.Linear(128, 256)
        if config.feature == 'wav':
            self.linear8 = nn.Linear(256, outputs * outch)
        elif config.feature == 'mel':
            self.linear8 = nn.Linear(256, outputs[0] * outputs[1] * outch)
            # self.linear8 = nn.Linear(256, self.config.nmels * 8 * (self.config.len // (self.config.nfft // 2) + 1))

        if config.weight:
            layers = dict(self._modules)
            for layer in layers.keys():
                la = getattr(self, layer).weight
                la = torch.nn.Parameter(torch.zeros_like(la) + 1e-5)

    def forward(self, x):
        x = torch.reshape(x, (x.size(0), -1))
        if self.config.relu:
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            x = F.relu(self.linear3(x))
            x = F.relu(self.linear4(x))
            x = F.relu(self.linear5(x))
            x = F.relu(self.linear6(x))
            x = F.relu(self.linear7(x))
        else:
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            x = self.linear4(x)
            x = self.linear5(x)
            x = self.linear6(x)
            x = self.linear7(x)
        # out = self.linear8(x).transpose(1,2)
        out = self.linear8(x)
        if self.config.feature == 'wav':
            out = torch.reshape(out, (out.size(0), -1, 8))
        elif self.config.feature == 'mel':
            # make mel output
            # out = torch.reshape(out, (out.size(0), self.config.nmels, 8, -1))
            out = torch.reshape(out, (out.size(0), -1, self.config.nfft, 8))

        return out.type(torch.double)   