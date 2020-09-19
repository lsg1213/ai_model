import torch, math, pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class time_distributed(nn.Module):
    def __init__(self, module, dropout=None, batch_first=True):
        super().__init__()
        self.module = module
        self.dropout = dropout
        self.batch_first = batch_first
    
    def forward(self, x):
        """Input should be (batch, time, channels)"""
        x_size = x.size()
        
        if len(x_size) <= 2:
            return self.module(x)
        
        x_reshaped = x.contiguous().view(-1, x_size[-1])
        x_reshaped = self.module(x_reshaped)
        if self.dropout is not None:
            x_reshaped = self.dropout(x_reshaped)

        if self.batch_first:
            y = x_reshaped.contiguous().view(x_size[0], -1, x_reshaped.size(-1))
        else:
            y = x_reshaped.contiguous().view(-1, x_size[0], x_reshaped.size(-1))
        
        return y

class spectral_attention_module(nn.Module):
    def __init__(self, input_ch, n_chan):
        super(spectral_attention_module, self).__init__()
        self.conv1 = nn.Conv2d(input_ch, n_chan, 3)
        self.conv2 = nn.Conv2d(input_ch, n_chan, 3)
        self.maxpool = nn.MaxPool2d((1,2))

    def forward(self, x):
        x = F.pad(x,(1,1,1,1))
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out2 = torch.sigmoid(out2)
        out = out1 * out2
        out = self.maxpool(out)
        return out

class st_attention(nn.Module):
    def __init__(self, window_size = 7, specn_chan = 16, device=torch.device('cpu')):
        super(st_attention, self).__init__()
        self.spectral_attention1 = spectral_attention_module(1, specn_chan)
        for i in range(2,5):
            self.add_module(f'spectral_attention{i}', spectral_attention_module(specn_chan, specn_chan * 2))
            specn_chan *= 2
        self.linear1 = time_distributed(nn.Linear(5*128,256))
        self.batchnorm1 = nn.BatchNorm1d(window_size)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = time_distributed(nn.Linear(256,256))
        self.batchnorm2 = nn.BatchNorm1d(window_size)
        self.linear3 = nn.Linear(256,128,bias=False)
        self.linear4 = nn.Linear(256,128,bias=False)
        self.linear5 = nn.Linear(256,128,bias=False)
        self.linear6 = nn.Linear(128,256)
        self.linear7 = nn.Linear(256,256)
        # self.linear6 = time_distributed(nn.Linear(128,256))
        # self.linear7 = time_distributed(nn.Linear(256,256))
        self.batchnorm3 = nn.BatchNorm1d(window_size)
        self.dropout2 = nn.Dropout(0.5)
        self.batchnorm4 = nn.BatchNorm1d(window_size)
        self.dropout3 = nn.Dropout(0.5)
        self.linear8 = nn.Linear(256,1)
        self.linear9 = nn.Linear(128,1)
        self.linear10 = nn.Linear(256,1)
        self.device = device
        
    def forward(self, x):
        out = torch.unsqueeze(x, 1)
        out = self.spectral_attention1(out)
        out = self.spectral_attention2(out)
        out = self.spectral_attention3(out)
        out = self.spectral_attention4(out)
        out = torch.reshape(out, (x.shape[0],x.shape[1],-1))
        #pipenet
        out = self.linear1(out)
        out = F.relu(self.batchnorm1(out))
        out = self.dropout1(out)
        out = self.linear2(out)
        out = F.relu(self.batchnorm2(out))
        pipenet_logits = self.dropout2(out)
        pipenet_soft = torch.sigmoid(self.linear8(pipenet_logits))

        #temporal attention
        n_heads, head_size = 4, 32
        g = torch.mean(pipenet_logits,dim=1)
        query = torch.sigmoid(self.linear3(g))
        key = torch.sigmoid(self.linear4(pipenet_logits))
        value = torch.sigmoid(self.linear5(pipenet_logits))

        heads = []
        for i in range(n_heads):
            qu = query[:, head_size*i:head_size*(i+1)] # (batch, head_size)
            k = key[:, :, head_size*i:head_size*(i+1)] # (batch, window, head_size)
            k = k.transpose(1,2) #(batch, head_size, l)
            head = qu.unsqueeze(1).bmm(k).squeeze(1) / np.sqrt(32) # (batch, window)
            head = torch.softmax(head,dim=1)
            head = head.unsqueeze(-1)
            head = head * value[:, :, head_size*i:head_size*(i+1)]
            heads.append(head)

        multihead_logits = torch.cat(heads, -1)
        multihead_soft = torch.sigmoid(self.linear9(multihead_logits))

        #postnet
        out = self.linear6(multihead_logits)
        out = self.batchnorm3(out)
        out = self.dropout2(F.relu(out))
        out = self.linear7(out)
        out = self.batchnorm4(out)
        post_logits = self.dropout3(F.relu(out))
        post_soft = torch.sigmoid(self.linear10(post_logits))

        pipenet_soft = torch.squeeze(pipenet_soft, -1)
        multihead_soft = torch.squeeze(multihead_soft, -1)
        post_soft = torch.squeeze(post_soft, -1)
        return pipenet_soft, multihead_soft, post_soft