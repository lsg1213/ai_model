import torch, pickle, pdb
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class CombineAutoencoder(nn.Module):
    def __init__(self, inputs, outputs, inch, outch):
        super(CombineAutoencoder, self).__init__()
        self.conv1 = nn.Conv1d(12, 64, 3, padding=1).double()
        self.batchnorm1 = nn.BatchNorm1d(64).double()
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1).double()
        self.batchnorm2 = nn.BatchNorm1d(128).double()
        self.pool = nn.MaxPool1d(2, 2)
        self.back = FCAutoencoder(inputs // 4, outputs, 128, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.batchnorm1(x))
        # x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))
        # x = F.relu(x)
        x = self.pool(x)
        x = self.back(x)
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
    def __init__(self, inputs, outputs, inch, outch):
        super(FCAutoencoder, self).__init__()
        # self.conv1 = nn.Conv1d(12,128,kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(inputs * inch,256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 30)
        self.linear5 = nn.Linear(30, 64)
        self.linear6 = nn.Linear(64, 128)
        self.linear7 = nn.Linear(128, 256)
        self.linear8 = nn.Linear(256, outputs * outch)

    def forward(self, x):
        # x = self.conv1(x.type(torch.float))
        x = torch.reshape(x.type(torch.float), (x.size(0),-1))
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