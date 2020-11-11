import torch, torchaudio, pdb
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
from scipy.signal import welch, hann, windows
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch.nn.functional as F
from scipy.io.wavfile import write
from math import ceil


def data_spread(data,data_length):
    if type(data) == list:
        res = torch.cat([torch.tensor(i[:(len(i) // data_length) * data_length]) for i in data])
        res = torch.reshape(res, (-1, data_length, res.size(-1)))
    return res

def inverse_mel(data, sr=8192, n_mels=160):
    # data = (batch, )
    # data = 
    pass


def write_wav(data, sr=8192, name='test_gen'):
    data = data.type(torch.float32).numpy()
    data = data - np.min(data)
    data /= np.max(data)
    write(name+'.wav', sr, data)
    return data

class testDataset(Dataset):
    def __init__(self, accel, sound, config):
        self.config = config
        self.accel = torch.from_numpy(np.array(self.flatten(accel)))
        self.sound = torch.from_numpy(np.array(self.flatten(sound)))
        
        self.mode = 'end'

        self.split_num = config.split_number
        self.mode = 'center' # split place of out
        self.index = torch.arange(0, len(self.sound), self.split_num)
        # self.accel = self.split(self.accel)
        # self.sound = self.split(self.sound)

    def flatten(self, data):
        return [x for y in data for x in y]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        index = self.index[idx]
        frame_size = self.config.b
        if self.config.future:
            frame_size += self.config.len
        # sound_index = index + frame_size + self.config.latency + (self.config.len // 2)
        sound_index = index + frame_size + self.config.latency
        
        accel = self.accel[index:index + frame_size]
        
        sound = self.split(self.sound[sound_index:sound_index + self.config.len], self.config.len // 2 - (self.config.split_number // 2), self.config.len // 2 + (self.config.split_number // 2))
        if accel.size(0) != frame_size:
            accel = torch.cat([accel,torch.zeros((frame_size - accel.size(0), accel.size(1)), device=accel.device, dtype=accel.dtype)])
        if sound.size(0) != self.split_num:
            sound = torch.cat([sound,torch.zeros((self.split_num - sound.size(0), sound.size(1)), device=sound.device, dtype=sound.dtype)])
        return accel.transpose(0,1), sound

    def split(self, data, start, end):
        if len(data.shape) == 2:
            return data[start:end]
        elif len(data.shape) == 3:
            return data[:,start:end]

def padding(signal, Ls):
    _pad = torch.zeros((signal.size(0), Ls, signal.size(2)), device=signal.device, dtype=signal.dtype)
    return torch.cat([_pad, signal],1)
    
def conv_with_S(signal, S_data, config, device=torch.device('cpu')):
    # S_data(Ls, K, M)
    if config.ema:
        signal = ema(signal, n=2)
    S_data = torch.tensor(S_data.transpose(0,1).cpu().numpy()[:,::-1,:].copy(),device=signal.device)
    Ls = S_data.size(1)
    K = S_data.size(-1)
    signal = padding(signal, Ls)
    if signal.size(1) != K:
        signal = signal.transpose(1,2)
    
    out = F.conv1d(signal, S_data.permute([2,0,1]).type(signal.dtype)).transpose(1,2)[:,:-1,:]
    
    return out 

# FILTERA Generates an A-weighting filter.
#    FILTERA Uses a closed-form expression to generate
#    an A-weighting filter for arbitary frequencies.
#
# Author: Douglas R. Lanman, 11/21/05

# Define filter coefficients.
# See: http://www.beis.de/Elektronik/AudioMeasure/
# WeightingFilters.html#A-Weighting
def filter_A(F):
    c1 = 3.5041384e16
    c2 = 20.598997 ** 2
    c3 = 107.65265 ** 2
    c4 = 737.86223 ** 2
    c5 = 12194.217 ** 2

    f = F
    arr = np.array(list(map(lambda x : 1e-17 if x == 0 else x, f)))

    num = np.sqrt(c1 * (arr ** 8))
    den = (f ** 2 + c2) * np.sqrt((f ** 2 + c3) * (f**2 + c4)) * (f ** 2 + c5)
    A = num / den
    return A

def dBA_metric(y, gt, plot=True):
    """
    |args|
    :y: generated sound data, it's shape should be (time, 8)
    :gt: ground truth data, it's shape should be (time, 8)
    :plot: if True, plot graph of each channels
    """
    d = gt
    e = gt - y
    Tn = y.shape[0]
    K = 8
    M = 8
    """Post processing : performance metric and plots"""
    p_ref = 20e-6
    fs = 2000
    Nfft = fs
    noverlap = Nfft / 2
    t = (np.arange(Tn) / fs)[np.newaxis, :]
    #win = hann(fs, False)
    win = windows.hamming(fs)
    #autopower calculation
    D = np.zeros((int(Nfft/2 + 1), M))
    E = np.zeros((int(Nfft/2 + 1), M))
    for m in range(M):
        F, D[:,m] = welch(d[:, m], fs=fs, window=win, noverlap=noverlap, nfft=Nfft, return_onesided=True, detrend=False)
        F, E[:,m] = welch(e[:, m], fs=fs, window=win, noverlap=noverlap, nfft=Nfft, return_onesided=True, detrend=False)
    
    A = filter_A(F)
    AA = np.concatenate([[A]] * M, axis=0).transpose(1,0)
    D_A = D * AA ** 2 / p_ref ** 2
    E_A = E * AA ** 2 / p_ref ** 2
    
    # perfomance metric calculation
    D_A_dBA_Sum = np.zeros((1,M))
    E_A_dBA_Sum = np.zeros((1,M))
    freq_range = np.arange(500)
    result = []
    E_result = []
    for m in range(M):
        D_A_dBA_Sum[0, m] = 10 * np.log10(np.sum(D_A[freq_range, m]))
        E_A_dBA_Sum[0, m] = 10 * np.log10(np.sum(E_A[freq_range, m]))
        result.append(D_A_dBA_Sum[0,m] - E_A_dBA_Sum[0,m])
        E_result.append(E_A_dBA_Sum[0, m])
    result.append(np.array(np.mean(result)))
    avg_result = np.mean(result)
    #E_result = np.array(np.mean(E_result))
    E_re = np.mean(np.array(E_result))
    
    gs = gridspec.GridSpec(nrows= int(ceil(M/4)), # row 몇 개 
                       ncols=4, # col 몇 개 
                      )
    fig = plt.figure(figsize=(16,10))
    if plot:
        for m in range(M):
            plt.subplot(gs[m])
            da = D_A[:,m]
            ea = E_A[:,m]
            plt.plot(F, 10 * np.log10(da), color="red", label=f'D{D_A_dBA_Sum[0, m]:.2f}dBA')
            plt.plot(F, 10 * np.log10(ea), color="blue", label=f'E{E_A_dBA_Sum[0, m]:.2f}dBA')
            plt.legend()
            plt.ylim((10, 60))
            plt.xlim((20,1000))
            plt.xlabel('Frequency(Hz)')
            plt.ylabel('Level(dBA)')
        plt.tight_layout()
        plt.show()
    
    return avg_result, E_re