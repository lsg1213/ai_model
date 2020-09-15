import torch, torchaudio, pdb
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from scipy.io.wavfile import write

def data_spread(data,data_length):
    if type(data) == list:
        res = torch.cat([torch.tensor(i[:(len(i) // data_length) * data_length]) for i in data])
        res = torch.reshape(res, (-1, data_length, res.size(-1)))
    return res

def inverse_mel(data, sr=8192, n_mels=160):
    # data = (batch, )
    # data = 
    pass
    
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