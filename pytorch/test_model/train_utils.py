import torch, torchaudio, pdb
from torch.utils.data import DataLoader, Dataset
from utils import data_spread
import numpy as np

feature_list = ['wav', 'mel', 'stft']



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
        self.accel = data_spread(accel, config)
        self.sound = data_spread(sound, config)
        self.config = config
        self.device = device

        self.batch_size = config.batch
        self.data_per_epoch = config.data_per_epoch

        self.len = len(self.accel) - config.b - config.len - config.latency
        if self.config.future:
            self.len -= self.config.len
        self.idx = torch.arange(0,self.len,dtype=torch.int32)
    
    def next_loader(self,train=True):
        x = []
        y = []
        while True:
            if train:
                self.idx = torch.randperm(len(self.idx))
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

        self.accel = data_spread(accel, config).to(device)
        self.sound = data_spread(sound, config).to(device)

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



if __name__ == "__main__":
    from params import get_arg
    import sys, pdb, torchaudio
    config = get_arg(sys.argv[1:])
    config.nfft = 1024
    config.len = 2048
    filt = filterWithSTFT(config, device)
    spt = torchaudio.transforms.Spectrogram(n_fft=config.nfft, win_length=config.win_len, hop_length=config.hop_len)
    config.hop_len = config.nfft // 4
    stft = wavToSTFT(config)
    wave = torch.rand(8,10000)
    st = stft(wave)

    pdb.set_trace()