import torch, pdb
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F

# def dataSplit(data, takebeforetime, data_length=40, expand=True):
#     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#     # data shape list(25, np(987136, 12)), accel, 주의: 샘플별로 안 섞이게 하기
#     # 이걸 자르기, (index, window, channel)
#     # data_length = int(hp.audio.sr * hp.audio.win_length / 1000000)
#     if expand:
#         if takebeforetime != 0:
#             data = [np.concatenate([np.zeros((takebeforetime, i.shape[1]),dtype='float'), i]) for i in data]
#         splited_data = torch.cat([torch.cat([torch.from_numpy(_data[idx-takebeforetime:idx+data_length][np.newaxis, ...]) for idx in range(takebeforetime, len(_data) - data_length)]) for _data in data[:1]])
#     else:
#         splited_data = torch.cat([torch.cat([torch.from_numpy(_data[idx:idx+data_length][np.newaxis, ...]) for idx in range(len(_data) - data_length)]) for _data in data])
#     return splited_data.cpu()

def data_spread(data,data_length):
    res = torch.cat([torch.tensor(i[:(len(i) // data_length) * data_length]) for i in data])
    res = torch.reshape(res, (-1, data_length, res.size(-1)))
    return res
    
class makeDataset(Dataset):
    def __init__(self, accel, sound, takebeforetime=40, data_length=40, train=True):
        if takebeforetime % data_length != 0:
            raise ValueError(f'takebeforetime must be the multiple of data_length, {takebeforetime}')
        self.accel = data_spread(accel, data_length)
        self.sound = data_spread(sound, data_length)
        self.takebeforetime = takebeforetime
        self.data_length = data_length
        self.perm = torch.arange(len(self.accel))
        if train:
            self.shuffle()
        if len(accel) < (takebeforetime // data_length) + 1:
            raise ValueError(f'Dataset is too small, {len(accel)}')
    
    def shuffle(self):
        self.perm = torch.randperm(len(self.accel))

    def __len__(self):
        return len(self.accel)

    def __getitem__(self, idx):
        if self.perm[idx] - (self.takebeforetime // self.data_length) < 0:
            return torch.cat([torch.zeros((((self.takebeforetime // self.data_length) - self.perm[idx]) * self.accel.size(1),) + self.accel.shape[2:],dtype=self.accel.dtype,device=self.accel.device),self.accel[self.perm[idx]]]), self.sound[self.perm[idx]]
        return torch.reshape(self.accel[self.perm[idx] - (self.takebeforetime // self.data_length): self.perm[idx] + 1], (-1, self.accel.size(-1))), self.sound[self.perm[idx]]


def padding(signal, Ls, device):
    _pad = torch.zeros((signal.size(0), Ls, signal.size(2)), device=device, dtype=signal.dtype)
    return torch.cat([_pad, signal],1)
    
def conv_with_S(signal, S_data, device=torch.device('cpu')):
    # S_data(Ls, K, M)
    S_data = torch.tensor(S_data.transpose(0,1).cpu().numpy()[:,::-1,:].copy(),device=device,dtype=signal.dtype)
    Ls = S_data.size(1)
    K = S_data.size(-1)
    signal = padding(signal, Ls, device)
    if signal.size(1) != K:
        signal = signal.transpose(1,2)
    out = F.conv1d(signal, S_data.permute([2,0,1]))

    return out.transpose(1,2)[:,:-1,:]

import numpy as np
import scipy
from scipy.signal import welch, hann
import matplotlib.pyplot as plt

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

def write_wav(data, sr=8192, name='test_gen'):
    data = data.type(torch.float32).numpy()
    data = data - np.min(data)
    data /= np.max(data)
    write(name+'.wav', sr, data)
    return data

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
    win = hann(fs, False)
    #autopower calculation
    D = np.zeros((int(Nfft/2 + 1), M))
    E = np.zeros((int(Nfft/2 + 1), M))
    for m in range(M):
        F, D[:,m] = welch(d[:, m], fs, window=win, noverlap=noverlap, nfft=Nfft, return_onesided=True, detrend=False)
        F, E[:,m] = welch(e[:, m], fs, window=win, noverlap=noverlap, nfft=Nfft, return_onesided=True, detrend=False)
    
    A = filter_A(F)
    AA = np.concatenate([[A]] * M, axis=0).transpose(1,0)
    D_A = D * AA ** 2 / p_ref ** 2
    E_A = E * AA ** 2 / p_ref ** 2
    
    # perfomance metric calculation
    D_A_dBA_Sum = np.zeros((1,M))
    E_A_dBA_Sum = np.zeros((1,M))
    freq_range = np.arange(500)
    result = []
    for m in range(M):
        D_A_dBA_Sum[0, m] = 10 * np.log10(np.sum(D_A[freq_range, m]))
        E_A_dBA_Sum[0, m] = 10 * np.log10(np.sum(E_A[freq_range, m]))
        result.append(D_A_dBA_Sum[0,m] - E_A_dBA_Sum[0,m])
    result.append(np.array(np.mean(result)))
    avg_result = np.mean(result)
    
    if plot:
        for m in range(M):
            plt.subplot(2, 4, m+1)
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
    
    return avg_result

def ema(data, n=40):
    '''
    exponential mov
    '''
    smoothing_factor = 2. / (n + 1)
    #get n sma first and calculate the next n period ema
    ema = torch.zeros_like(data, dtype=data.dtype, device=data.device)
    ema[:n] = torch.mean(data[:n])

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (s[n] - sma) * multiplier) + sma)
    ema[n] = ((data[n] - ema[n-1]) * smoothing_factor) + ema[n-1]
    for i,j in enumerate(data[n:]):
        ema[i] = ((j - ema[i-1]) * smoothing_factor) + ema[i-1]

    return ema