import torch, pdb;
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader

class customDataset(Dataset):
    def __init__(self, dataset, config, device):
        self.config = config
        self.dataset = dataset
        self.device = device
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.dataset[idx][1] != self.config.sr:
            resample = Resample(self.data[idx][1], self.config.sr)
            x = resample(self.dataset[idx, 0].to(self.device))


        return x.cpu(), y.cpu()