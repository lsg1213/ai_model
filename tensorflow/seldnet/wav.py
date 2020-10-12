import joblib, os
import librosa
from glob import glob
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
import IPython.display as ipd
from scipy.io import loadmat


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
datapath = '/root/datasets/ai_challenge/interspeech20'
# datapath = '/root/datasets/DCASE2020/mic_dev'
SR = 16000
def loading(_path):
    data, sr = librosa.load(_path, sr=None, mono=False)
    num = int(_path.split('.')[-2][-6:])
    data = librosa.resample(data, sr, SR)

    return {'data': data, 'index': num}

with ThreadPoolExecutor(max_workers=cpu_count() // 2) as pool:
    a = glob(os.path.join(datapath, 'train/*.wav'))
    _trainset = list(pool.map(loading, a))
with ThreadPoolExecutor(max_workers=cpu_count() // 2) as pool:
    a = glob(os.path.join(datapath, 'test/*.wav'))
    _testset = list(pool.map(loading, a))
trainset = len(_trainset) * [None]
testset = len(_testset) * [None]
for i in _trainset:
    trainset[i['index'] - 1] = i['data']
for i in _testset:
    testset[i['index'] - 1] = i['data']
trainlabel = loadmat(os.path.join(datapath,'train/metadata_wavs.mat'))['phi'][0]
testlabel = loadmat(os.path.join(datapath,'test/metadata_wavs.mat'))['phi'][0]
datapath = '/root/datasets/ai_challenge/interspeech20/seld'

joblib.dump(trainset, open(os.path.join(datapath, 'train_x.joblib'), 'wb'))
joblib.dump(trainlabel, open(os.path.join(datapath, 'train_y.joblib'), 'wb'))
joblib.dump(testset, open(os.path.join(datapath, 'test_x.joblib'), 'wb'))
joblib.dump(testlabel, open(os.path.join(datapath, 'test_y.joblib'), 'wb'))