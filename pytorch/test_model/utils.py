import torch, torchaudio, pdb, librosa
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from scipy.io.wavfile import write
import concurrent.futures as fu
import librosa
import random

feature_list = ['wav', 'mel', 'stft']

class _Loss(torch.nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class CustomLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(CustomLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return customLoss(input, target)

def stft_extractor(n_fft=128, win_length=128, hop_length=128//2):
    def _get_stft(x):
        """x should be (time, chan)"""
        time, chan = x.shape
        stft = []
        for i in range(chan):
            _stft = librosa.stft(x[:, i],
                                n_fft=n_fft,
                                win_length=win_length,
                                hop_length=hop_length)
            stft.append(_stft[np.newaxis])
        stft = np.concatenate(stft, axis=0) #(chan, freq, time)
        return stft
    return _get_stft

def istft_extractor(win_length=128, hop_length=128//2):
    def _get_istft(x):
        """x should be (chan, freq, time)"""
        chan, freq, time = x.shape
        stft = []
        for i in range(chan):
            _stft = librosa.istft(x[i, :, :],
                                win_length=win_length,
                                hop_length=hop_length)
            stft.append(_stft[np.newaxis])
        istft = np.concatenate(stft, axis=0) #(chan, freq, time)
        return istft
    return _get_istft

def customLoss(y, y_pred):
    vy = y - torch.mean(y)
    vyy = y_pred - torch.mean(y_pred)

    cost = torch.sum(vy * vyy) / (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vyy ** 2)))
    return - cost

def data_spread(data, data_length, config):
    '''
    (number of file, frames, channel) => (all frames, channel)
    and cut wave frames by data_length
    '''
    if type(data) == list:
        res = torch.cat([torch.tensor(i) for i in data])
    return res

def get_diff(data):
    return data[:,1:] - data[:,:-1]

class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, x, y):
        super(IterableDataset).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            x = self.x
            y = self.y
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(x) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(x))
        return zip(x,y)

class makeGenerator():
    def __init__(self, accel, sound, config, device=torch.device('cpu')):
        self.accel = data_spread(accel, config.len, config)
        self.sound = data_spread(sound, config.len, config)
        self.config = config
        self.device = device

        self.batch_size = config.batch
        self.data_per_epoch = config.data_per_epoch

        self.len = len(self.accel) - config.b - config.len - config.latency
        if self.config.future:
            self.len -= self.config.len
        self.idx = torch.arange(0,self.len,dtype=torch.int32)
    
    def next_loader(self):
        x = []
        y = []
        while True:
            random.shuffle(self.idx)
            for idx in self.idx:
                index = idx + self.config.latency
                frame_size = self.config.b
                if self.config.future:
                    frame_size += self.config.len
                x.append(self.accel[idx:idx + self.config.b + self.config.len].transpose(0,1))
                y.append(self.sound[index + frame_size:index + frame_size + self.config.len])
                
                if len(x) >= self.config.data_per_epoch:
                    x = torch.stack(x,0)
                    y = torch.stack(y,0)
                    dataset = IterableDataset(x,y)
                    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                            batch_size = self.config.batch
                                                        )
                    x = []
                    y = []
                    yield data_loader

class makeDataset(Dataset):
    def __init__(self, accel, sound, config, device, train=True):
        self.config = config
        self.takebeforetime = config.b
        self.data_length = config.len
        self.device = device

        self.accel = data_spread(accel, self.data_length, config).to(device)
        self.sound = data_spread(sound, self.data_length, config).to(device)

        self.perm = torch.arange(len(self.accel) - self.config.latency - self.config.b - 2 * self.config.len if self.config.future else len(self.accel))
        if train:
            self.shuffle()
        self.len = len(self.accel) - config.b - config.len - config.latency
        if self.config.future:
            self.len -= self.config.len
    
    def shuffle(self):
        if self.config.feature in feature_list:
            self.perm = torch.randperm(len(self.accel) - self.config.latency - self.config.b - 2 * self.config.len if self.config.future else len(self.accel) - self.config.latency - self.config.b - self.config.len)


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx = self.perm[idx]
        index = idx + self.config.latency
        frame_size = self.config.b
        if self.config.future:
            frame_size += self.config.len
        accel = self.accel[idx:idx + self.config.b + self.config.len].transpose(0,1)
        sound = self.sound[index + frame_size:index + frame_size + self.config.len]
        return accel, sound

def padding(signal, Ls):
    _pad = torch.zeros((signal.size(0), Ls, signal.size(2)), device=signal.device, dtype=signal.dtype)
    return torch.cat([_pad, signal],1)

def meltowav(mel, config):
    # mel shape = (batch, frames, n_mels, channel=8)
    if len(mel.shape) == 4:
        mel = mel.permute((0,3,2,1))  # (batch, 8, n_mels, frames)
    else:
        raise ValueError(f'mel dimension must be 4, now {len(mel.shape)}')

    mid = torchaudio.transforms.InverseMelScale(config.nfft // 2 + 1, config.nmels, sample_rate=config.sr).to(mel.device)(mel)
    wav = torchaudio.transforms.GriffinLim(config.nfft).to(mel.device)(mid)
    return wav

def conv_with_S(signal, S_data, config, device=torch.device('cpu')):
    # S_data(Ls, K, M)
    if config.ema:
        signal = ema(signal, n=2)
    
    Ls = S_data.size(1)
    K = S_data.size(-1)
    signal = padding(signal, Ls)
    if signal.size(1) != K:
        signal = signal.transpose(1,2)
    
    out = F.conv1d(signal, S_data.permute([2,0,1]).type(signal.dtype)).transpose(1,2)[:,:-1,:]
    
    return out 

def ema(data, n=2):
    '''
    exponential mov
    '''
    smoothing_factor = 2. / (n + 1)
    #get n sma first and calculate the next n period ema
    ema = torch.zeros_like(data, dtype=data.dtype, device=data.device)
    ema[:n] = torch.mean(data[:n])

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    for i,j in enumerate(data[n:]):
        ema[i] = ((j - ema[i-1]) * smoothing_factor) + ema[i-1]

    return ema


def wavToSTFT(config, device=torch.device('cpu')):
    def _wavToSTFT(wav):
        '''wav (channel, time)'''
        return torch.functional.stft(wav.to(device), n_fft=config.nfft, win_length=config.nfft, hop_length=config.nfft // 2, return_complex=True)
    '''output stft (channel, nfft // 2 + 1, time, 2->real,imag)'''
    return _wavToSTFT 

def STFTToWav(config, device=torch.device('cpu')):
    def _STFTToWav(stft):
        '''stft (channel, config.nfft // 2 + 1, time, 2->real,imag)'''
        return torch.functional.istft(stft.to(device), n_fft=config.nfft, win_length=config.nfft, hop_length=config.nfft // 2)
    '''output wav (channel, time)'''
    return _STFTToWav 