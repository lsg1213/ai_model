from collections import deque
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

class makeDataset(Dataset):
    def __init__(self, accel, sound, transform=None):
        self.transform = transform
        self.accel = accel
        self.sound = sound
    
    def __len__(self):
        return len(self.accel)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        accel = self.accel[idx]
        sound = self.sound[idx]

        if self.transform:
            return self.transform(accel), self.transform(sound)
        return accel, sound

def dataSplit(data, data_length=500, device=torch.device('cpu')):
    num = sum([data[i].shape[0] for i in range(len(data))])
    data_ = torch.zeros((int((num + data_length - (num % data_length)) / data_length) * data_length, data[0].shape[-1])).to(device)
    i = 0
    while i < num:
        for j in tqdm(range(len(data))):
            for k in range(len(data[j])):
                data_[i] = torch.from_numpy(data[j][k])
                i += 1
    return data_.reshape((int((num + data_length - (num % data_length)) / data_length), data_length, data[0].shape[-1])).cpu()

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return deque(factors)

class mu_encode(nn.Module):
    def __init__(self, quant_chan=256, rescale_factor=10):
        super(mu_encode, self).__init__()
        self.quant_chan = quant_chan
        self.rescale_factor = rescale_factor
    
    def forward(self, x):
        x /= self.rescale_factor
        return mu_law_encoding(x, self.quant_chan)

def mu_law_encoding(x, quantization_channels):
    mu = quantization_channels - 1.
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return torch.round(((x_mu + 1) / 2 * mu + 0.5)).type(torch.float)

def mu_law_expansion(x_mu, quantization_channels):
    mu = quantization_channels - 1.
    x = ((x_mu) / mu) * 2 - 1.
    return np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu



def time_to_batch(sigs, sr):
    '''Adds zero padding to inputs and reshapes by sample rate.  This essentially
    rebatches the input into one second batches.

    Used to perform 1D dilated convolution

    Args:
      sig: (tensor) in (Bx)SxC; B = # batches, S = # samples, C = # channels
      sr: (int) sample rate of audio signal
    Outputs:
      sig: (tensor) also in SecBatchesx(B x sr)xC, SecBatches = # of seconds in
            padded sample
    '''

    unsqueezed = False

    # check if sig is a batch, if not make a batch of 1
    if len(sigs.size()) == 1:
        sigs.unsqueeze_(0)
        unsqueezed = True

    assert len(sigs.size()) == 3

    # pad to the second (i.e. sample rate)
    b_num, s_num, c_num = sigs.size()
    width_pad = int(sr * np.ceil(s_num / sr + 1))
    lpad_len = width_pad - s_num
    lpad = torch.zeros(b_num, pad_left_len, c_num)
    sigs = torch.cat((lpad, sigs), 1) # concat on sample dimension

    # reshape to batches of one second each
    secs_num = width_pad // sr
    sigs = sigs.view(secs_num, -1, c_num) # seconds x (batches*rate) x channels

    return sigs

def batch_to_time(sigs, sr, lcrop=0):
    ''' Reshape to 1d signal from batches of 1 second.

    I'm using the same variable names as above as opposed to the original
    author's variables

    Used to perform dilated conv1d

    Args:
      sig: (tensor) second_batches_num x (batch_size x sr) x channels
      sr: (int)
      lcrop: (int)
    Outputs:
      sig: (tensor) batch_size x # of samples x channels
    '''

    assert len(sigs.size()) == 3

    secs_num, bxsr, c_num = sigs.size()
    b_num = bxsr // sr
    width_pad = int(secs_num * sr)

    sigs = sigs.view(-1, width_pad, c_num) # missing dim should be b_num

    assert sigs.size(0) == b_num

    if lcrop > 0:
        sigs = sigs[:,lcrop:, :]

    return sigs
