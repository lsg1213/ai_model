import torch, pickle, torchaudio, pdb
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class CombineAutoencoder(nn.Module):
    def __init__(self, inputs, outputs, inch, outch, config):
        super(CombineAutoencoder, self).__init__()
        self.config = config
        self.conv1 = nn.Conv1d(inch, 64, 3, padding=1).double()
        self.batchnorm1 = nn.BatchNorm1d(64).double()
        self.dropout1 = nn.Dropout(p=0.2).double()
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1).double()
        self.batchnorm2 = nn.BatchNorm1d(128).double()
        self.dropout2 = nn.Dropout(p=0.2).double()
        self.pool = nn.MaxPool1d(2, 2)
        if config.feature == 'mel':
            outputs = config.n_mels
        self.back = FCAutoencoder(inputs // 4, outputs, 128, outch, config)
        if config.weight:
            with torch.no_grad():
                self.conv1.weight = torch.nn.Parameter(torch.zeros_like(self.conv1.weight) + 1e-5)
                self.conv2.weight = torch.nn.Parameter(torch.zeros_like(self.conv2.weight) + 1e-5)

    def inverse_mel(self, x):
        pdb.set_trace()
        shape = x.size()
        x = x.clone().cpu().detach()
        x = torchaudio.transforms.InverseMelScale(self.config.b + self.config.len,n_mels=160, sample_rate=8192).double()(x)
        
        # x = torchaudio.functional.istft()
        return x
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.dropout1(self.batchnorm1(x)))
        # x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(self.dropout2(self.batchnorm2(x)))
        # x = F.relu(x)
        x = self.pool(x)
        x = self.back(x)
        if self.config.feature == 'mel':
            x = self.inverse_mel(x)
        return x

class ConvAutoencoder(nn.Module):
    def __init__(self, inputs, outputs):
        super(ConvAutoencoder, self).__init__()
        self.inputs = inputs
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv1d(4, 16, 3, padding=1).double()
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv1d(16, 4, 3, padding=1).double()
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool1d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose1d(4, 16, 2, stride=2).double()
        self.t_conv2 = nn.ConvTranspose1d(16, 8, 2, stride=2).double()

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))
                
        return x.transpose(1,2)

class FCAutoencoder(nn.Module):
    def __init__(self, inputs, outputs, inch, outch, config):
        super(FCAutoencoder, self).__init__()
        self.config = config
        # self.conv1 = nn.Conv1d(12,128,kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(inputs * inch,256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 30)
        self.linear5 = nn.Linear(30, 64)
        self.linear6 = nn.Linear(64, 128)
        self.linear7 = nn.Linear(128, 256)
        self.linear8 = nn.Linear(256, outputs * outch)
        if config.weight:
            layers = dict(self._modules)
            for layer in layers.keys():
                la = getattr(self, layer).weight
                la = torch.nn.Parameter(torch.zeros_like(la) + 1e-5)

    def forward(self, x):
        # x = self.conv1(x.type(torch.float))
        x = torch.reshape(x.type(torch.float), (x.size(0),-1))
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
        out = self.linear8(x)
        out = torch.reshape(out, (out.size(0), -1, 8))
        return out.type(torch.double)   