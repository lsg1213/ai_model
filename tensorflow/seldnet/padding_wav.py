import torch, os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torchaudio, pdb, joblib
import tensorflow as tf
from glob import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from tqdm import tqdm
from scipy.io import loadmat

''' This code is working on tf 2.3 '''
class zipping(object):
    def __init__(self, org_wav, path, trans_funcs):
        self.org_wav = org_wav
        self.path = path
        self.trans_funcs = trans_funcs
        self.idx = -1
    
    def __len__(self):
        return len(self.trans_funcs)

    def __iter__(self):
        return self

    def __next__(self):
        self.idx = self.idx + 1
        yield (self.org_wav, self.path, self.idx, self.trans_funcs[self.idx])

def _get_pad_size(wav_len, total_sec, sr=16000):
    return int(total_sec * sr - wav_len)

def pad_wav(wav,
            label,
            datatype,
            total_sec=None, time_axis=1):
    # wav = (channel, time), label = (1)
    r = 16000
    if total_sec is None: # if None, double the wav
        total_sec = wav.shape[time_axis] * 2 / r

    total_padsize = _get_pad_size(wav.shape[time_axis], total_sec, sr=r)
    if total_padsize > 0:
        left = total_padsize // 2
        right = total_padsize - left
        wav = torch.nn.functional.pad(wav, [left, right, 0, 0])
        label = torch.nn.functional.pad(label, [left, right], value=10)
    
    return wav, label




def main():
    sr = 16000
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    abs_path = '/root/datasets/ai_challenge/seldnet/seld'
    trans_funcs = np.load('/root/datasets/ai_challenge/seldnet/clean_iris' + '/ir_short_16k.npy').astype(np.float32)
    second = 8
    for p in ['train', 'test', 'validation']:
        path = os.path.join(abs_path, str(second))
        if p in ['train', 'validation']:
            labels = loadmat(os.path.join(os.path.join(abs_path, p), 'metadata_wavs'))
        elif p == 'test':
            labels = loadmat(os.path.join(os.path.join(abs_path, p), 'angle'))['phi'][0]
        
        for idx, w in enumerate(tqdm(sorted(glob(os.path.join(abs_path, f'{p}/*.wav'))), total=len(sorted(glob(os.path.join(abs_path, f'{p}/*.wav')))))):
            save_path = os.path.join(path, w.split('/')[-2])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, os.path.basename(w))
            wav, r = torchaudio.load(w)
            wav.to(device)

            if p == 'test':
                label = labels[idx]
                if label == -1:
                    label = 10
                else:
                    label /= 20
                label = torch.ones(wav.shape[-1], dtype=wav.dtype) * label
            elif p in ['train', 'validation']:
                start = labels['voice_start'][0][idx]
                end = labels['voice_end'][0][idx]
                angle = labels['phi'][0][idx]
                if angle == -1:
                    angle = 10
                else:
                    angle /= 20
                label = torch.cat([10 * torch.ones(start, dtype=wav.dtype, device=wav.device), angle * torch.ones(end-start, dtype=wav.dtype, device=wav.device), 10 * torch.ones(wav.shape[-1] - end, dtype=wav.dtype, device=wav.device)])
                

            if r != sr:
                resample = torchaudio.transforms.Resample(r, sr).to(device)
                wav = resample(wav)
                label = label[::r//sr]  
                wav = wav[:min(wav.shape[-1], label.shape[0])]
                label = label[:min(wav.shape[-1], label.shape[0])]

            output, label = pad_wav(wav, label, p, second, 1)
            
            save_path = os.path.join('/'.join(save_path.split('/')[:-1]), os.path.basename(save_path).split('.')[0]+'.wav')
            
            torchaudio.save(save_path, output.cpu(), 16000)
            joblib.dump(label.cpu().numpy(), open(os.path.join('/'.join(save_path.split('/')[:-1]), os.path.basename(save_path).split('.')[0] + f'_label.joblib'),'wb'))



def parse_label(path):
    data = loadmat(os.path.join(path, 'angle'))
    data
    pdb.set_trace()

def lab():
    sr = 16000
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    abs_path = '/root/datasets/ai_challenge/seldnet/seld'
    trans_funcs = np.load('/root/datasets/ai_challenge/seldnet/clean_iris' + '/ir_short_16k.npy').astype(np.float32)
    second = 10
    for p in ['train', 'test', 'validation']:
        path = os.path.join(abs_path, p)
        labels = parse_label(path)


if __name__ == "__main__":
    main()
