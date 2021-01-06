import os
import pickle
import torch
import torchaudio
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


def from_wav_to_dataset(path, feature='magphase', pad=True, config=None):
    assert feature in ['complex', 'magphase', 'mel']
    files = sorted(os.listdir(path))
    dataset = []
    max_len = 0
    max_wav_len = 0
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    stft = torchaudio.transforms.Spectrogram(config.nfft, power=None).to(device)
    mel = torchaudio.transforms.MelSpectrogram(n_fft=config.nfft, n_mels=80).to(device)

    for f in tqdm(files):
        if not f.endswith('.wav'):
            continue
        data, sample_rate = torchaudio.load(os.path.join(path, f))
        if data.shape[-1] > max_wav_len:
            max_wav_len = data.shape[-1]
        data = data.to(device)
        data = torchaudio.compliance.kaldi.resample_waveform(data,
                                                             sample_rate,
                                                             16000)

        if feature == 'mel':
            data = mel(data)
        else:
            data = stft(data)
            
            if feature == 'magphase':
                data = torchaudio.functional.magphase(data)
            else:
                data = [data[..., 0], data[..., 1]]
            data = torch.cat(data)
        data = data.cpu().numpy().transpose(1, 2, 0) # frequency, time, chan
        data = data.astype('float32')
        dataset.append(data)

        if data.shape[1] > max_len:
            max_len = data.shape[1]
    if pad:
        def pad_spec(x, max_len):
            return np.pad(x, ((0, 0), (0, 375 - x.shape[1]), (0, 0)), 'constant')

        dataset = np.stack(tuple(map(lambda x: pad_spec(x, max_len), dataset))).transpose(0,2,1,3)
        max_wav_len = (375 * 256 - 256) * 3

    return dataset, max_wav_len


def load_dataset(folder_path, 
                 metadata='metadata_wavs.mat',
                 n_fft=512,
                 sample_rate=16000):
    '''
    extracts dataset from the folder
    the folder must have wav files and its metadata in .mat format
    returns continuous spectrograms and corresponding labels

    INPUT:
        folder_path: STR -> location of the folder
        metadata: STR -> the name of metadata

    OUTPUT:
        (spectrogram, labels) -> tuples of 'np.ndarray's
    '''
    assert os.path.isdir(folder_path)
    hop = n_fft // 2

    # specs
    wavs = sorted([wav for wav in os.listdir(folder_path) 
                   if wav.endswith('.wav')])
    specs = []

    # labels
    meta = loadmat(os.path.join(folder_path, metadata))
    start = meta['voice_start'].squeeze()
    end = meta['voice_end'].squeeze()
    phi = meta['phi'].squeeze()
    labels = []

    stft = torchaudio.transforms.Spectrogram(n_fft, power=None)

    for i, wav in tqdm(enumerate(wavs)):
        wav, s_rate = torchaudio.load(os.path.join(folder_path, wav))
        spec = torchaudio.compliance.kaldi.resample_waveform(wav,
                                                             s_rate,
                                                             sample_rate)
        spec = stft(spec)
        spec = torch.cat(torchaudio.functional.magphase(spec))
        spec = spec.numpy().astype(np.float32)
        spec = spec.transpose(2, 1, 0) # time, freq, chan

        label = np.zeros(spec.shape[0], dtype=np.int32) 
        multiplier = sample_rate / s_rate / hop
        label[int(start[i]*multiplier):int(end[i]*multiplier)] = 1
        label = label*phi[i] + (1-label)*-1

        specs.append(spec)
        labels.append(label)

    specs = np.concatenate(specs, axis=0)
    labels = np.concatenate(labels, axis=0)

    return specs, labels


if __name__ == '__main__':
    from_wav_to_dataset('/media/data1/datasets/ai_challenge/interspeech20/test',
                        feature='mel', pickled=True)
