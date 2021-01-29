import torch, pdb
import torch.nn as nn
from data_utils import onehot
from utils import characterMap
import numpy as np
cmap = characterMap()

# BLSTM layer for pBLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through BLSTM
# Note the input should have timestep%2 == 0
class pBLSTMLayer(nn.Module):
    def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim*2,hidden_dim,1, bidirectional=True, 
                                   dropout=dropout_rate,batch_first=True)
    
    def forward(self,input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
        # Bidirectional RNN
        output,hidden = self.BLSTM(input_x)
        return output,hidden

class Attention(nn.Module):
    def __init__(self, config, classnum):
        super(Attention, self).__init__()
        self.config = config
        # self.W = nn.Linear(,bias=False)
        # self.V = nn.Linear(,bias=False)

    def forward(self, key, value, last_attention):
        '''
        key: 이전 hidden state
        value: feature
        last_attention: 이전 attention score
        '''
        pdb.set_trace()
        context = torch.bmm(key, value) * last_attention

class Listener(nn.Module):
    def __init__(self, input_shape, config):
        super(Listener, self).__init__()
        self.config = config
        self.pBLSTMs = nn.ModuleList().append(pBLSTMLayer(input_shape[0], self.config.hidden))
        for i in range(2):
            self.pBLSTMs.append(pBLSTMLayer(self.config.hidden * 2, self.config.hidden))
        self.resolution = 2 ** len(self.pBLSTMs)

    def forward(self, x):
        # x = (batch, channel, time)
        x = x.transpose(1,2) # (batch, time, channel)
        x = x[:,:x.shape[1] // self.resolution * self.resolution]
        for pBLSTM in self.pBLSTMs:
            x, _ = pBLSTM(x)
        return x

class Speller(nn.Module):
    def __init__(self, input_shape, config, max_len_y):
        super(Speller, self).__init__()
        self.config = config
        self.classnum = len(cmap.encodemap.keys())
        self.max_len_y = max_len_y
        self.lstm = nn.LSTM(self.config.hidden * 2 + self.classnum, self.config.hidden, num_layers=2, batch_first=True)
        self.attention = Attention(self.config, self.classnum)
        self.CharacterDistribution = nn.Linear(self.config.hidden * 2, self.classnum)

    def forward(self, s):
        sos = torch.from_numpy(onehot(np.ones((s.shape[0],1), dtype=np.int) * cmap.encode('<sos>'))).to(s.device).type(s.dtype)
        attention_score = torch.zeros((s.shape[0], self.config.hidden), device=s.device)
        
        feature, hid = self.lstm(torch.cat([s[:,:1],sos],dim=-1)) # rnn
        y = self.CharacterDistribution(torch.cat([feature.squeeze(1),  attention_score], dim=-1))
        y_all = [y]
        for i, _ in enumerate(range(self.max_len_y)):
            pdb.set_trace()
            x, hid = self.lstm(torch.cat([s[:,i], y0], dim=-1), hid)
            self.attention(hid, s, attention_score)
        return 

class LAS(nn.Module):
    def __init__(self, input_shape, config, max_len_y):
        super(LAS, self).__init__()
        # input_shape = (channel, windows)
        self.config = config
        self.Listener = Listener(input_shape, self.config)
        self.Speller = Speller(input_shape, self.config, max_len_y)

    def forward(self, x):
        # x=(batch,channel,time)
        x = self.Listener(x) # x = (batch, time, feature)
        x = self.Speller(x)
        return x