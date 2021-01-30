from data_utils import dataset, dataIO
import torch, torchaudio
from torch.utils.data import DataLoader
from params import get_param
from model import get_model

def main(config):
    model = get_model(config)
    x = [1,2,3]
    y = [0,1,1]
    trainset = dataset.Custom_Dataset(x, y, config)
    for i,j in trainset:
        print(i,j)

if __name__=='__main__':
    import sys
    main(get_param(sys.argv[1:]))