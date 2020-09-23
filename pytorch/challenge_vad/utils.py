import torch, pickle, pdb, joblib
from glob import glob
import numpy as np
from random import choice
from glob import glob
import pdb
import concurrent.futures, multiprocessing
from sklearn.metrics import auc, roc_curve
# PATH = '/root/datasets/ai_challenge/ST_attention_dataset'
# x = pickle.load(open(PATH+'/timit_noisex_x_mel.pickle', 'rb'))
# y = pickle.load(open(PATH+'/timit_noisex_y_mel.pickle', 'rb'))
# val_x = pickle.load(open(PATH+'/libri_aurora_val_x_mel.pickle', 'rb'))
# val_y = pickle.load(open(PATH+'/libri_aurora_val_y_mel.pickle', 'rb'))
# for i in range(len(x)):
#     x[i] = x[i][:, :len(y[i])]
# for i in range(len(val_x)):
#     val_x[i] = val_x[i][:, :len(val_y[i])]

# np.concatenate(list(map(preprocess_spec(config, feature=config.feature), x[x_index[index - 1]:x_index[index]])), axis=0)

EPSILON = torch.tensor(1e-8)
LOG_EPSILON = torch.log(EPSILON)

def getAUC(pred, truth):
    fpr, tpr, thresholds = roc_curve(truth.numpy(), pred.numpy(), pos_label=1)
    return auc(fpr, tpr)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, train=True, transform=None):
        self.transform = transform
        self.train = train
        self.x_data = x
        self.y_data = y

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # x = self.x[idx, :, :]
        # y = self.y[idx, :]
        # pickle을 여기에서 불러와서 각 pickle별로 index에 변화를 주어 만들기
        x = self.x_data[idx, :, :].type(torch.float)
        y = self.y_data[idx, :].type(torch.float)

        return x, y


class Dataloader_generator():
    def __init__(self, data, labels, transform, config, divide=10, train=True, device=torch.device('cpu'), batch_size=512, n_data_per_epoch=10000):
        self.transform = transform
        self.batch_size = batch_size
        self.config = config
        self.data = data
        self.labels = labels
        self.n_data_per_epoch = n_data_per_epoch
        self.device = device
        self.train = train
        self.divide = divide
        self.perm = torch.randperm(len(self.data))
        self.win = WindowUtils(config.pad_size, config.step_size, device)
    
    def shuffle(self):
        self.perm = torch.randperm(len(self.data))

    def next_loader(self, idx):
        data = [self.data[i] for i in self.perm[idx * (len(self.data) // self.divide): (idx + 1) * (len(self.data) // self.divide)]]
        label = [self.labels[i] for i in self.perm[idx * (len(self.data) // self.divide): (idx + 1) * (len(self.data) // self.divide)]]

        while True:
            perm = torch.randperm(len(data)).to(self.device)
            
            data = torch.cat(list(map(self.win.preprocess_spec(self.config, feature=self.config.feature), [torch.from_numpy(data[i]) for i in perm])), axis=0)
            labels = torch.cat(list(map(self.win.label_to_window(self.config), [torch.from_numpy(label[i]) for i in perm])), dim=0)
            # with concurrent.futures.ThreadPoolExecutor() as pool:
            #     data = torch.cat(list(pool.map(self.win.preprocess_spec(self.config, feature=self.config.feature), [torch.from_numpy(i) for i in data])), axis=0)
            # with concurrent.futures.ThreadPoolExecutor() as pool:  
            #     labels = torch.cat(list(pool.map(self.win.label_to_window(self.config), [torch.from_numpy(i) for i in label])), dim=0)

            if len(data) != len(labels):
                raise ValueError(f'data {data.shape}, labels {labels.shape}')

            dataset = Dataset(data, labels, transform=self.transform)
            data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size = self.batch_size,
                                                    shuffle=True if self.train else False,
                                                    drop_last=True
                                                )
            del data
            del labels
            torch.cuda.empty_cache()
            yield data_loader
                

class WindowUtils():
    def __init__(self, pad_size, step_size, device):
        self.pad_size = pad_size
        self.step_size = step_size
        self.window = torch.cat([torch.arange(-pad_size, -step_size, step_size), torch.tensor([-1, 0, 1]), torch.arange(step_size+1, pad_size+1, step_size)], dim=0)
        self.device = device

    def sequence_to_windows(self, sequence, 
                            pad_size, 
                            step_size, 
                            skip=1,
                            padding=True, 
                            const_value=0):
        '''
        SEQUENCE: (time, ...)
        PAD_SIZE:  int -> width of the window // 2
        STEP_SIZE: int -> step size inside the window
        SKIP:      int -> skip windows...
            ex) if skip == 2, total number of windows will be halved.
        PADDING:   bool -> whether the sequence is padded or not
        CONST_VALUE: (int, float) -> value to fill in the padding
        RETURN: (time, window, ...)
        '''
        assert (pad_size-1) % step_size == 0
        if pad_size == self.pad_size and step_size == self.step_size:
            window = self.window.to(sequence.device)
        else:
            window = torch.cat([torch.arange(-pad_size, -step_size, step_size, device=sequence.device), torch.tensor([-1, 0, 1], device=sequence.device), torch.arange(step_size+1, pad_size+1, step_size, device=sequence.device)], dim=0)
            window += pad_size
        output_len = len(sequence) if padding else len(sequence) - 2*pad_size
        window = window.unsqueeze(0) + torch.arange(0, output_len, skip, device=sequence.device).unsqueeze(-1)

        if padding:
            pad = torch.ones((pad_size, *sequence.shape[1:]), dtype=sequence.dtype, device=sequence.device)
            pad = pad * const_value
            sequence = torch.cat([pad, sequence, pad], dim=0)
        return sequence[window]

    def windows_to_sequence(self, windows,
                            pad_size,
                            step_size):
        sequence = torch.zeros((windows.shape[0],), dtype=torch.float32, device=windows.device)
        indices = torch.arange(1, windows.shape[0]+1, device=windows.device)
        
        indices = self.sequence_to_windows(indices, pad_size, step_size, True)
        
        for i in range(windows.shape[0]):
            pred = windows[torch.where(indices == i + 1)]
            sequence[i] = pred.mean()
        
        return torch.round(sequence)

    # TODO (test is required)
    def label_to_window(self, config, skip=1):
        def _preprocess_label(label):
            label = self.sequence_to_windows(
                label, config.pad_size, config.step_size, skip, True)
            return label
        return _preprocess_label

    def preprocess_spec(self, config, feature='mel', skip=1):
        if feature not in ['spec', 'mel', 'mfcc']:
            raise ValueError(f'invalid feature - {feature}')

        def _preprocess_spec(spec):
            if feature in ['spec', 'mel']:
                spec = torch.log(spec + EPSILON)
            if feature == 'mel':
                if config.norm == 'timit':
                    spec = (spec - 4.5252) / 2.6146 # normalize
                elif config.norm == 'paper':
                    spec -= torch.min(spec)
                    spec /= torch.max(spec) + 1e-6
            spec = spec.transpose(1, 0) # to (time, freq)
            windows = self.sequence_to_windows(spec, 
                                        config.pad_size, config.step_size,
                                        skip, True, LOG_EPSILON)
            return windows
        return _preprocess_spec

def getDataFromPath(path):
    data = joblib.load(open(path, 'rb'))
    return data