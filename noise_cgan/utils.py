import os, time
import numpy as np
import librosa

FEATURES = ('stft', 'mfcc', 'seq')
audio_path = '/root/datasets/ai_challenge/NOISEX/all/'

def get_data(resample_sr=16000, length=4, feature='stft', audio_path=audio_path):
    dir_ = np.array(os.listdir(audio_path))
    _train_label = [i for i, j in np.char.split(dir_, sep='-')]
    label_list = sorted(list(set(_train_label)))
    for i in range(len(_train_label)):
        _train_label[i] = label_list.index(_train_label[i])

    if not(feature in FEATURES):
        raise ValueError('wrong feature')
    
    resampled_data, sr, train_label = [], [], []
    start_time = time.time()

    step = 0
    for k, i in enumerate(dir_):
        _audio_data = None
        __audio_data, _sr = librosa.load(audio_path+i, sr=None)
        __audio_data = librosa.resample(
            __audio_data, _sr, resample_sr) if _sr != resample_sr else __audio_data
        if __audio_data.shape[0] / resample_sr >= length:
            t = (length * resample_sr -
                (__audio_data.shape[0] % (length * resample_sr)))
            __audio_data = np.pad(__audio_data, (t//2, t - (t//2)),
                                'constant', constant_values=(0)) if t != 0 else __audio_data
            if __audio_data.shape[0] % (length * _sr) != 0:
                raise ValueError('something wrong with doing pad')
            _audio_data = np.split(__audio_data, int(
                __audio_data.shape[0] / (length * _sr)))
            for j in range(int(__audio_data.shape[0] / (length * _sr))):
                train_label.append(_train_label[k])
                sr.append(_sr)
        elif __audio_data.shape[0] / resample_sr < length:
            t = (length * _sr - (__audio_data.shape[0] % (length * _sr)))
            _audio_data = np.pad(__audio_data, (t//2, t - (t//2)),
                                'constant', constant_values=(0))
            train_label.append(_train_label[k])
            sr.append(_sr)
        for j in _audio_data:
            resampled_data.append(j)
        if step % 1 == 0:
            print(f'{step+1}/{dir_.shape[0]}')
        step += 1
    train_label = np.array(train_label)
    return (preprocessing(resampled_data,feature, resample_sr), train_label, label_list)

def preprocessing(data, feature, resample_sr):
    train_data = []
    if feature == 'stft':
        for i in range(len(data)):
            train_data.append(librosa.stft(data[i]))
    elif feature == 'mfcc':
        for i in range(len(data)):
            train_data.append(librosa.feature.mfcc(data[i], sr=resample_sr).T)
    elif feature == 'seq':
        for i in range(len(data)):
            train_data.append(data[i])
    else:
        raise ValueError('wrong feature')
    train_data = np.array(train_data)
    if train_data.ndim == 2:
        train_data = np.expand_dims(train_data,axis=-1)
    print(f'data preprocessing complete, data feature is {feature}')
    return train_data

if __name__ == "__main__":
    a,b,c=get_data()
    print(a.shape, b.shape)