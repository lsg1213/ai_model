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

        self.mode = 'center' # split place of out
        self.index = torch.arange(0, len(self.sound), self.config.len)
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
        
        sound = self.sound[sound_index:sound_index + self.config.len]
        if accel.size(0) != frame_size:
            accel = torch.cat([accel,torch.zeros((frame_size - accel.size(0), accel.size(1)), device=accel.device, dtype=accel.dtype)])
        if sound.size(0) != self.config.len:
            sound = torch.cat([sound,torch.zeros((self.config.len - sound.size(0), sound.size(1)), device=sound.device, dtype=sound.dtype)])
        return accel.transpose(0,1), sound

    def split(self, data, start, end):
        if len(data.shape) == 2:
            return data[start:end]
        elif len(data.shape) == 3:
            return data[:,start:end]




class makeDataset(Dataset):
    def __init__(self, accel, sound, config, train=True):
        self.config = config
        self.takebeforetime = config.b
        self.data_length = config.len
        if self.takebeforetime % self.data_length != 0:
            raise ValueError(f'takebeforetime must be the multiple of data_length, {takebeforetime}')
        if config.feature == 'mel':
            if type(accel) == list:
                melaccel = torch.from_numpy(np.concatenate(accel)).type(torch.float)
                tomel = torchaudio.transforms.MelSpectrogram(sample_rate=8192, n_fft=config.b + config.len, hop_length=config.b, n_mels=config.n_mels)
                self.accel = torch.cat([tomel(melaccel[:,i]).unsqueeze(0) for i in range(melaccel.shape[-1])]).type(torch.double).transpose(0,2)
            if type(sound) == list:
                self.sound = data_spread(sound, self.data_length)

        elif config.feature == 'wav':
            self.accel = data_spread(accel, self.data_length)
            self.sound = data_spread(sound, self.data_length)
        else:
            raise ValueError(f'invalid feature {config.feature}')
        self.perm = torch.arange(len(self.accel))
        if train:
            self.shuffle()
        if len(accel) < (self.takebeforetime // self.data_length) + 1:
            raise ValueError(f'Dataset is too small, {len(accel)}')
    
    def shuffle(self):
        self.perm = torch.randperm(len(self.accel))

    def __len__(self):
        return len(self.accel)

    def __getitem__(self, idx):
        if self.config.feature == 'wav':
            if self.perm[idx] - (self.takebeforetime // self.data_length) < 0:
                return torch.cat([torch.zeros((((self.takebeforetime // self.data_length) - self.perm[idx]) * self.accel.size(1),) + self.accel.shape[2:],dtype=self.accel.dtype,device=self.accel.device),self.accel[self.perm[idx]]]).transpose(0,1), self.sound[self.perm[idx]]
            return torch.reshape(self.accel[self.perm[idx] - (self.takebeforetime // self.data_length): self.perm[idx] + 1], (-1, self.accel.size(-1))).transpose(0,1), self.sound[self.perm[idx]]
        elif self.config.feature == 'mel':
            if self.perm[idx] - (self.takebeforetime // self.data_length) < 0:
                return torch.cat([torch.zeros((((self.takebeforetime // self.data_length) - self.perm[idx]) * self.accel.size(1),) + self.accel.shape[2:],dtype=self.accel.dtype,device=self.accel.device),self.accel[self.perm[idx]]]).transpose(0,1), self.sound[self.perm[idx]]
            return torch.reshape(self.accel[self.perm[idx] - (self.takebeforetime // self.data_length): self.perm[idx] + 1], (-1, self.accel.size(-1))).transpose(0,1), self.sound[self.perm[idx]]
        

'''
config.sr = 8192, config.range = "90~100", wav = (batch, channel, data)
사용법
filter = bandPassFilter(config)
wave = filter(wave)
'''
def highPassFilter(config):
    def _highPassFilter(wav):
        wav = torchaudio.functional.highpass_biquad(wav, config.sr, int(config.range.split('~')[-1]))
        return wav
    return _highPassFilter

def lowPassFilter(config):
    def _lowPassFilter(wav):
        wav = torchaudio.functional.lowpass_biquad(wav, config.sr, int(config.range.split('~')[0]))
        return wav
    return _lowPassFilter

def bandPassFilter(config):
    lowpass = lowPassFilter(config)
    highpass = highPassFilter(config)
    def _bandPassFilter(wav):
        wav = lowpass(wav)
        wav = highpass(wav)
        return wav
    return _bandPassFilter

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