from torch.utils.data import Dataset
from . import data_preprocess, label_preproces

class Custom_Dataset(Dataset):
    def __init__(self, x, y, config):
        super(Custom_Dataset, self).__init__()
        self.x = x
        self.y = y
        '''
        config.preprocess 형태
        [
            '전처리 이름1': {전처리 argument들},
            '전처리 이름2': {전처리 argument들}
        ]
        '''
        self.data_preprocess = config.data_preprocess
        self.label_preprocess = config.label_preprocess

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        for key in self.data_preprocess.keys():
            x = getattr(data_preprocess, key)(self.x[idx], **self.data_preprocess[key])

        for key in self.label_preprocess.keys():
            y = getattr(label_preproces, key)(self.y[idx], **self.label_preprocess[key])
        
        return x, y
