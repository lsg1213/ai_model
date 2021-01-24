import torch, pdb;
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from utils import characterMap

cmap = characterMap()

def padding(max_len):
    def _pad(x):
        if type(x) == np.ndarray:
            return np.pad(x, ((0,0),(0,max_len - x.shape[-1])))
        elif type(x) == list:
            for i in range(max_len - len(x)):
                x.append(0)
            return x
    return _pad

def onehot(y):
    return np.eye(len(cmap.encodemap.keys()))[y]

class customDataset(Dataset):
    def __init__(self, x, y, config, max_len_x, max_len_y):
        self.config = config
        self.x = np.array(list(map(padding(max_len_x), x)), dtype=np.float32) # (batch, time, feature)
        self.y = np.array(list(map(padding(max_len_y), y))) # (batch, label)
        self.y = onehot(self.y)
        print(f'input shape: ({self.x.shape})')
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.x[idx]
        y = self.y[idx]
        return x, y