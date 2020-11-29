import torch
import numpy as np
import torch.nn.functional as F

def data_spread(data, config):
    '''
    (number of file, frames, channel) => (all frames, channel)
    '''
    if type(data) == list:
        data = torch.from_numpy(np.concatenate(data))
    return data

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


def customLoss(y, y_pred):
    vy = y - torch.mean(y)
    vyy = y_pred - torch.mean(y_pred)

    cost = torch.sum(vy * vyy) / (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vyy ** 2)))
    return - cost

def padding(signal, Ls):
    _pad = torch.zeros((signal.size(0), Ls - 1, signal.size(2)), device=signal.device, dtype=signal.dtype)
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
    # S_data(Ls, K, M), signal(batch, frame, K)
    if config.ema:
        signal = ema(signal, n=2)
    Ls = S_data.size(0)
    K = S_data.size(1)
    signal = padding(signal, Ls)
    # conv1d (batch, inputchannel, W), (outputchannel, inputchannel, W)
    out = F.conv1d(signal.transpose(1,2), S_data.permute([2,1,0]).type(signal.dtype))
    
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
        return torch.functional.stft(wav.to(device), n_fft=config.nfft, win_length=config.win_len, hop_length=config.hop_len, return_complex=True)
    '''output stft (channel, nfft // 2 + 1, time, 2->real,imag)'''
    return _wavToSTFT 

def STFTToWav(config, device=torch.device('cpu')):
    def _STFTToWav(stft):
        '''stft (channel, config.nfft // 2 + 1, time, 2->real,imag)'''
        return torch.functional.istft(stft.to(device), n_fft=config.nfft, win_length=config.win_len, hop_length=config.hop_len,)
    '''output wav (channel, time)'''
    return _STFTToWav

def filterWithSTFT(config, device=torch.device('cpu')):
    stft = wavToSTFT(config, device)
    istft = STFTToWav(config, device)
    low, high = config.range.split('~')
    nbins = config.nfft // 2 + 1
    low = int(int(low) / (config.sr / nbins))
    high = int(int(high) / (config.sr / nbins))
    def _filter(inputs): 
        '''wav (batch, channel, nbins, time)'''
        if inputs.shape[-2] == nbins:
            # stft
            st = inputs
        else:
            # wav
            if len(inputs.shape) == 2:
                st = stft(inputs)
            elif len(inputs.shape) == 3:
                st = torch.stack(list(map(stft, inputs)))
        if len(st.shape) == 3:
            st[:,:max(low - 1,0),:] *= 0
            st[:,min(high + 1, st.shape[1]):,:] *= 0
            return istft(st)
        elif len(st.shape) == 4:
            st[:,:,:max(low - 1,0),:] *= 0
            st[:,:,min(high + 1, st.shape[1]):,:] *= 0
            return torch.stack(list(map(istft, st)))
    return _filter