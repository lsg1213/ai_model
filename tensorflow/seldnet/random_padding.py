import torch, os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torchaudio, pdb, joblib
import tensorflow as tf
from glob import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from tqdm import tqdm

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

def pad_wav(wav,
            _label,
            total_sec=None, time_axis=1):
    # wav = (channel, time)
    r = 16000
    if total_sec is None: # if None, double the wav
        total_sec = wav.shape[time_axis] * 2 / r

    total_padsize = int(total_sec * r - wav.shape[time_axis])
    if total_padsize > 0:
        left = torch.randint(high=total_padsize, size=(1,))[0]
        right = total_padsize - left
        label = torch.cat([10 * torch.ones(left, device=wav.device, dtype=wav.dtype), _label * torch.ones(wav.size(1), device=wav.device, dtype=wav.dtype), 10 * torch.ones(right, device=wav.device, dtype=wav.dtype)], -1)
        wav = torch.nn.functional.pad(wav, [left, right, 0, 0])
    return wav, label



def makeAngle(org_wav, trans_funcs, path, device):
    # org_wav = (1, time)
    def apply_trans_func(data):
        '''
        INPUT
        data[0]
        org_wav: single channel wav
        data[1]
        save_path: wav save path
        data[2]
        label: 0 ~ 9
        data[3]
        trans_func: [trans_func_width, output_chan]
        

        OUTPUT
        output: [wav_len] or [wav_len, output_chan]
        '''
        
        org_wav, save_path, label, trans_func = data
        
        org_wav = tf.reshape(org_wav, (1, -1, 1))
        # tf.conv1d is actually cross-correlation
        # reverse trans_func to get true convolution
        trans_func = tf.reverse(trans_func, axis=[0])
        trans_func = tf.expand_dims(trans_func, 1)

        output = tf.nn.conv1d(org_wav, trans_func, 1, 'SAME')
        # output = ()
        
        output = torch.from_numpy(tf.squeeze(output).numpy()).transpose(-1,-2).to(device)
        output, labels = pad_wav(output, label, total_sec=8)

        save_path = os.path.join('/'.join(save_path.split('/')[:-1]), os.path.basename(save_path).split('.')[0] + f'_{label}.wav')
        
        torchaudio.save(save_path, output.cpu(), 16000)
        joblib.dump(labels.cpu().numpy(), open(os.path.join('/'.join(save_path.split('/')[:-1]), os.path.basename(save_path).split('.')[0] + f'_label.joblib'),'wb'))


    # with ThreadPoolExecutor(10) as pool:
    #     pool.map(apply_trans_func, zipping(org_wav, path, trans_funcs))
    for i, j in enumerate(trans_funcs):
        apply_trans_func((org_wav, path, i, j))
    # map(apply_trans_func, zipping(org_wav, path, trans_funcs))



def main():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    abs_path = '/root/datasets/ai_challenge/seldnet/clean_iris'
    trans_funcs = np.load('/root/datasets/ai_challenge/seldnet/clean_iris' + '/ir_short_16k.npy').astype(np.float32)
    second = 8
    for p in ['train', 'test']:
        path = os.path.join(abs_path, str(second))
        for w in tqdm(sorted(glob(os.path.join(abs_path, f'{p}/*.wav')))):
            save_path = os.path.join(path, w.split('/')[-2])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, os.path.basename(w))
            wav, r = torchaudio.load(w)
            wav.to(device)
            if r != 16000:
                resample = torchaudio.transforms.Resample(r, 16000).to(device)
                wav = resample(wav)
                r = 16000

            makeAngle(wav, trans_funcs, save_path, device)
            


if __name__ == "__main__":
    main()

# torchaudio.datasets.LIBRISPEECH('./librispeech','dev-clean',download=True)
# torchaudio.datasets.LIBRISPEECH('./librispeech','train-clean-100',download=True)
# torchaudio.datasets.LIBRISPEECH('./librispeech','test-clean',download=True)