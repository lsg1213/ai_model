"""
Script for util functions
"""
import os, joblib, pdb
import numpy as np
from sklearn import preprocessing
import tensorflow as tf

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)

def spectrogram(audio, sr, maxlen, nfft, winlen, hoplen, max_frames):
    def _spectrogram(audio_input):
        _nb_ch = audio_input.shape[1]
        hann_win = np.repeat(np.hanning(winlen)[np.newaxis].T, _nb_ch, 1)
        nb_bins = nfft // 2
        spectra = np.zeros((max_frames, nb_bins, _nb_ch), dtype=complex)
        for ind in range(max_frames):
            start_ind = ind * hoplen
            aud_frame = audio_input[start_ind + np.arange(0, winlen), :] * hann_win
            spectra[ind] = np.fft.fft(aud_frame, n=nfft, axis=0, norm='ortho')[:nb_bins, :]
        return spectra
    import concurrent.futures as fu
    with fu.ThreadPoolExecutor() as pool:
        res = list(pool.map(_spectrogram, audio))
    return np.array(res)

# def frame_to_window(sample, config):
#     # sample = (batch, channel, frame, feature)
#     sample = tf.transpose(sample, [0,2,3,1]) # (batch, frame, feature, channel)
#     if config.mode == 'sample':
#         return sample

#     def _sample_to_frame(sample):
#         # sample = (frame, feature, channel)
#         res = tf.zeros(sample.shape[:1] + (config.window_size,) + sample.shape[1:], dtype=sample.dtype)
        
#         for i, j in enumerate(sample):
#             start = tf.maximum(i-(config.window_size // 2), 0).numpy()
#             end = tf.minimum(i+(config.window_size // 2), len(sample)).numpy()
#             tf.tensor_scatter_nd_update(res[i][:], tf.range((config.window_size // 2) + (start - i),(config.window_size // 2) + (end - i))[...,tf.newaxis], sample[start:end])

#         return res

#     if config.mode == 'frame':
#         out = tf.map_fn(_sample_to_frame, sample)
#         return out


def preprocess(data_path='/root/datasets/ai_challenge/interspeech20/seld'):
    base_path = data_path
    eps = np.spacing(np.float(1e-16))
    sr = 16000
    maxlen = sr * 7 # sr * seconds
    nfft = 512
    winlen = nfft
    hoplen = nfft // 2
    max_frames = int(np.ceil((maxlen - winlen) / float(hoplen)))

    trainaud_dir = os.path.join(base_path, 'train_x.joblib')
    traindesc_dir = os.path.join(base_path, 'train_y.joblib')
    testaud_dir = os.path.join(base_path, 'test_x.joblib')
    testdesc_dir = os.path.join(base_path, 'test_y.joblib')
    feature_dir = os.path.join(base_path, f'{nfft}_{maxlen}')
    norm_feature_dir = feature_dir + '_norm'
    wts_dir = feature_dir + '_wts'
    create_folder(feature_dir)
    create_folder(norm_feature_dir)

    for path in [trainaud_dir, testaud_dir]:
        audio = joblib.load(path)
        for idx, wa in enumerate(audio):
            wa = np.transpose(wa.astype(np.float32), (1,0)) / 32768.0 + eps
            
            if wa.shape[0] < maxlen:
                zero_pad = np.zeros((maxlen - wa.shape[0], wa.shape[1]),dtype=wa.dtype)
                audio[idx] = np.vstack((wa, zero_pad))
            elif wa.shape[0] > maxlen:
                audio[idx] = wa[:maxlen, :]
        print('spectogram start')

        audio_spec = spectrogram(audio, sr, maxlen, nfft, winlen, hoplen, max_frames)
        audio_spec = audio_spec.reshape(audio_spec.shape[0], max_frames, -1)
        joblib.dump(audio_spec, open(os.path.join(feature_dir, os.path.basename(path)), 'wb'))
        
        print('spectogram end')
        spec_scaler = preprocessing.StandardScaler()
        if 'train_x' in path.split('/')[-1]:
            for i in audio_spec:
                spec_scaler.partial_fit(np.concatenate((np.abs(i), np.angle(i)), axis=1))
            joblib.dump(
                spec_scaler,
                wts_dir
            )
        audio_spec_norm = np.array([spec_scaler.transform(np.concatenate((np.abs(i), np.angle(i)), axis=1)) for i in audio_spec])
        del audio_spec
        joblib.dump(audio_spec_norm, open(os.path.join(norm_feature_dir, path.split('/')[-1]), 'wb'))
        del audio_spec_norm
        del spec_scaler
    print('data preprocessing over')
    # print('label preprocessing start')
    
    # for path in [traindesc_dir, testdesc_dir]:
    #     label = joblib.load(path)
    #     joblib.dump(label, open())

def terminateOnNaN(loss):
    if loss is not None:
        if np.isnan(loss) or np.isinf(loss):
            print('Batch %d: Invalid loss, terminating training' % (batch))
            return True
    return False

if __name__ == "__main__":
    preprocess()