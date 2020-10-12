import torchaudio, pickle, os, pdb, joblib, argparse, torch
import numpy as np
import concurrent.futures as fu
from math import ceil
args = argparse.ArgumentParser()
args.add_argument('--nmels', type=int, default=80)
args.add_argument('--nfft', type=int, default=512)
args.add_argument('--sr', type=int, default=8192)
args.add_argument('--divide', type=int, default=100)
args.add_argument('--feature', type=str, default='mel')
config = args.parse_args()

ABSpath = '/home/skuser'
if not os.path.exists(ABSpath):
    ABSpath = '/root'
data_path = os.path.join(ABSpath,'data')
if not os.path.exists(data_path):
    data_path = os.path.join(ABSpath, 'datasets/hyundai')

accel_raw_data = torch.from_numpy(np.concatenate([x[np.newaxis,...] for y in pickle.load(open(os.path.join(data_path,'stationary_accel_data.pickle'),'rb')) for x in y]))
sound_raw_data = torch.from_numpy(np.concatenate([x[np.newaxis,...] for y in pickle.load(open(os.path.join(data_path,'stationary_sound_data.pickle'),'rb')) for x in y]))
print('data load done')
config.divide = ceil(len(accel_raw_data) / config.divide)
if config.divide == 0:
    config.divide = 1
feat_path = os.path.join(data_path, f'{config.feature}_{config.nfft}_{config.nmels}')
if not os.path.exists(feat_path):
    os.makedirs(feat_path)

def _transform(config):
    if config.feature == 'mel':
        return torchaudio.transforms.MelSpectrogram(sample_rate=config.sr, n_fft=config.nfft, win_length=config.nfft, hop_length=config.nfft//2, n_mels=config.nmels)

def seqtowin(seq, config):
    windows = []
    #(frame, channel)
    hop = config.nfft // 2
    for i in range(0, len(seq), hop):
        win = seq[i:i + config.nfft]
        if len(win) != config.nfft:
            win = torch.cat([win, torch.zeros((config.nfft - len(win), seq.shape[-1]), dtype=win.dtype, device=win.device)])
        windows.append(win)
    pdb.set_trace()
    return torch.cat([i.unsqueeze(0) for i in windows])

def transform(accel, sound, config):
    accel = accel.split(config.divide)
    sound = sound.split(config.divide)
    transfunc = _transform(config)
    for idx, (da, so) in enumerate(zip(accel, sound)):
        da = da.transpose(0,1).type(torch.float32)
        with fu.ThreadPoolExecutor() as pool:
            adata = list(pool.map(transfunc, da))
        adata = torch.cat([i.unsqueeze(0) for i in adata]).transpose(0,2) # (frames, nmels, channel)
        sound = seqtowin(so, config) # (frames, )
        adata = adata[:len(sound)]
        joblib.dump(adata, open(os.path.join(feat_path, f'{idx:04}_accel_data.joblib'), 'wb'))
        del adata
        joblib.dump(sound, open(os.path.join(feat_path, f'{idx:04}_sound_data.joblib'), 'wb'))
        del sound

transform(accel_raw_data, sound_raw_data, config)
print('transform done')
