import os, argparse, pickle, librosa
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import torch
from glob import glob
from eval_utils import *
import models
from tqdm import tqdm
from time import time
import numpy as np

args = argparse.ArgumentParser()
args.add_argument('--gpus', type=str, default='0')
args.add_argument('--batch', type=int, default=1)
args.add_argument('--len', type=int, default=200)
args.add_argument('--b', type=int, default=200)
args.add_argument('--latency', type=int, default=5)
args.add_argument('--mode', type=str, default='sj_S')
args.add_argument('--model', type=str, default='FCAutoencoder')
args.add_argument('--weight', action='store_true')
args.add_argument('--ema', action='store_true')
args.add_argument('--relu', action='store_true')
args.add_argument('--future', action='store_true')
args.add_argument('--feature', type=str, default='wav', choices=['wav', 'mel'])
config = args.parse_known_args(['--relu', '--future'])[0]
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
SR = 8192
WINDOW_SIZE = 500 # us
data_length = config.len
BATCH_SIZE = config.batch
K, m = 8, 8
ls = 128

ABSpath = '.'
if not os.path.exists(ABSpath):
    ABSpath = '/root'
path = os.path.join(ABSpath, 'ai_model/pytorch/test_model')
data_path = os.path.join(ABSpath,'data')
if not os.path.exists(data_path):
    data_path = os.path.join(ABSpath, 'datasets')
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
accel_raw_data = pickle.load(open(os.path.join(data_path,'stationary_accel_data.pickle'),'rb'))
sound_raw_data = pickle.load(open(os.path.join(data_path,'stationary_sound_data.pickle'),'rb'))
transfer_f = np.array(pickle.load(open(os.path.join(data_path,'transfer_f.pickle'),'rb')))
transfer_f = torch.from_numpy(transfer_f).to(device)
transfer_f.requires_grad = False

dataset = makeDataset(accel_raw_data, sound_raw_data, config, False)
# name = f'CombineAutoencoder_sj_S_b{config.b}_d{config.len}_lat{config.latency}_adam_0.001_decay0.7071_featurewav_future/*'
# name = 'FCAutoencoder_sj_S_40_40_adam_0.001_decay0.7071/*'
modelsavepath = sorted(glob(os.path.join(path, 'model_save/*')), key=lambda x: float(x.split('/')[-1].split('_')[-1].split('.pt')[0]), reverse=True)
model = getattr(models, name.split('_')[0])(dataset[0][0].shape[1], dataset[0][1].shape[0], dataset[0][0].shape[0], dataset[0][1].shape[1], config).to(device)


dataset_generator = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
for modelweight in modelsavepath:
    model.load_state_dict(torch.load(modelweight)['model'])
    data_res, sound_res = [], []
    times = []
    model.eval()
    with torch.no_grad():
        for data, sound in tqdm(dataset_generator):
            start = time()
            re = conv_with_S(model(data.to(device)), S_data=transfer_f, device=device, config=config).cpu()
            times.append(time() - start)
            data_res.append(re)
            sound_res.append(sound)
    data_res = torch.cat(data_res)
    sound_res = torch.cat(sound_res)
    data_res = torch.reshape(data_res, (-1,data_res.size(-1)))
    sound_res = torch.reshape(sound_res, (-1,data_res.size(-1)))
    score = dBA_metric(data_res, sound_res, False)
    print(modelweight)
    print(f'mean prediction time: {np.mean(times):0.4}, score: {score}')