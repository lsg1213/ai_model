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
    def __init__(self, accel, sound, takebeforetime=40, data_length=40):
        if takebeforetime % data_length != 0:
            raise ValueError(f'takebeforetime must be the multiple of data_length, {takebeforetime}')
        self.accel = data_spread(accel, data_length)
        self.sound = data_spread(sound, data_length)
        self.takebeforetime = takebeforetime
        self.data_length = data_length
        self.perm = torch.randperm(len(self.accel))
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


def Conv_S(y, s_filter, device='cpu'):
    # defined as a function
    ## New Conv S => Validated
    Ls, K, M = s_filter.shape
    Tn = y.shape[0]
    y_buffer = torch.zeros((Ls, K), device=device)
    y_p = torch.zeros(y.size(), device=device)
    #e = torch.zeros(y.size())

    for n in range(Tn):
        for k in range(K):
            for m in range(M):
                y_p[n,m] += torch.dot(y_buffer[:, k], s_filter[:, k, m])
        #e[n, :] = d[n, :] - y_p[n, :]
        y_buffer[1:, :] = y_buffer[:-1, :].clone().to(device)
        y_buffer[0, :] = y[n , :]
    return y_p

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