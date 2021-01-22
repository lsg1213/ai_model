import os, sys
import torch, torchaudio, joblib, pdb
from torchaudio.transforms import Spectrogram
from params import getParam
from python_speech_features import logfbank
from torchaudio.transforms import Resample
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from utils import characterMap



class Datasets:
    def __init__(self):
        self.train = torchaudio.datasets.LIBRISPEECH(datapath, url='dev-clean', download=True)
        self.test = torchaudio.datasets.LIBRISPEECH(datapath, url='test-clean', download=True)

def resampling(config, device='cpu'):
    def _resampling(data):
        res = data[0]
        if data[1] != config.sr:
            resample = Resample(data[1], config.sr).to(device)
            res = resample(res.to(device)).cpu()
        return res
    return _resampling

def preprocess(data):
    return logfbank(data)

def preencoding(data, cmap):
    label = data[2].lower()
    res = []
    for i, j in enumerate(label):
        res.append(cmap.encode(j))
    res = [cmap.encode('<sos>')] + res + [cmap.encode('<eos>')]
    return res

    
if __name__ == "__main__":
    config = getParam(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    datapath = os.path.join(config.abspath, 'datasets/lsj')
    Dataset = Datasets()
    charmap = characterMap()

    for dataset in ['train', 'test']:
        data = getattr(Dataset, dataset)
        
        # x = list(map(resampling(config), data))
        # # data preprocess
        # with ProcessPoolExecutor(max_workers=cpu_count() // 2) as pool:
        #     x = list(pool.map(preprocess, x))
        
        y = list(map(preencoding, data, charmap))
        # label preprocess
        with ProcessPoolExecutor(max_workers=cpu_count() // 2) as pool:
            pass