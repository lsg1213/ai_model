import torch
from torch.utils.data import DataLoader, Dataset

def dataSplit(data, data_length=40):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # data shape list(25, np(987136, 12)), accel, 주의: 샘플별로 안 섞이게 하기
    # 이걸 자르기, (index, window, channel)
    # data_length = int(hp.audio.sr * hp.audio.win_length / 1000000)
    splited_data = torch.cat([torch.cat([torch.from_numpy(_data[idx:idx+data_length][np.newaxis, ...]) for idx in range(len(_data) // data_length)]) for _data in data])
    
    return splited_data.cpu()

class makeDataset(Dataset):
    def __init__(self, accel, sound, train=True):
        self.accel = accel
        self.sound = sound
        if train:
            perm = torch.randperm(len(self.accel))
            self.accel = self.accel[perm]
            self.sound = self.sound[perm]
    
    def __len__(self):
        return len(self.accel)

    def __getitem__(self, idx):
        return self.accel[idx], self.sound[idx]

