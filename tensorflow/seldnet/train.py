import argparse, pdb
import numpy as np
import os
from params import getparam
import sys
config = getparam(sys.argv[1:])
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
import pickle, joblib
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from cls_feature_class import FeatureClass
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import keras_model
from da_utils import *
from da_transforms import *
AUTOTUNE = tf.data.experimental.AUTOTUNE
'''
LABEL PRE-PROCESSING
'''
def degree_to_class(degrees,
                    resolution=20,
                    min_degree=0,
                    max_degree=180,
                    one_hot=True):
    degrees = np.array(degrees)
    n_classes = int((max_degree-min_degree)/resolution + 2)

    mask = np.logical_and(min_degree <= degrees, degrees <= max_degree)
    classes = mask * (degrees/resolution) + (1-mask) * (n_classes-1)
    classes = classes.astype(np.int32)

    if not one_hot:
        return classes 
    return np.eye(n_classes, dtype=np.float32)[classes]

def samplewise_gaussian_noise(specs, labels, tau=None):
    noise = tf.random.normal(specs.shape) * tf.math.reduce_std(specs) * 0.1
    if tau is not None:
        noise *= tf.math.log(tf.maximum(tf.random.uniform([]), EPSILON)) * tau
    return specs + noise, labels

def to_dataset(x, y, config, train=False):
    '''
    args:
        x: complex spectrograms [batch, freq, time, chan]
        y: degrees [0, 20, 40, ... 180, -1]
    '''

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if train:
        dataset = dataset.repeat().shuffle(len(x))
    dataset = dataset.batch(config.batch, drop_remainder=True)

    return dataset.prefetch(AUTOTUNE)

def pre_batch_augment(specs, labels, time_axis=1, freq_axis=0):
    # specs, labels = samplewise_gaussian_noise(specs, labels, tau=0.5)
    # specs = random_conv(specs)
    specs = mask(specs, axis=time_axis, max_mask_size=30, n_mask=2) # time
    specs = mask(specs, axis=freq_axis, max_mask_size=16, n_mask=2) # freq
    specs, labels = random_magphase_flip(specs, labels)
    return specs, labels

def complex_to_log_minmax_norm_specs(specs, labels=None):
    n_chan = specs.shape[-1] // 2
    mag = tf.math.sqrt(specs[..., :n_chan]**2 + specs[..., n_chan:]**2)
    axis = tuple(range(1, len(specs.shape)))

    mag_max = tf.math.reduce_max(mag, axis=axis, keepdims=True)
    mag_min = tf.math.reduce_min(mag, axis=axis, keepdims=True)

    specs = (mag-mag_min)/tf.maximum(mag_max-mag_min, EPSILON)
    specs = tf.math.log(tf.maximum(specs, EPSILON))

    if labels is not None:
        return specs, labels
    return specs

def label_preprocess(spec, label):
    label = (tf.one_hot(tf.cast(label[0], label[1].dtype), 2), tf.one_hot(label[1], 10), tf.one_hot(label[2], 10))

    return spec, label
def da_to_dataset(x, y, config, train=False):
    '''
    args:
        x: complex spectrograms [batch, freq, time, chan]
        y: degrees [0, 20, 40, ... 180, -1]
    '''
    # y = degree_to_class(y)

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if train:
        dataset = dataset.repeat().shuffle(len(x))
        # dataset = dataset.map(pre_batch_augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(config.batch, drop_remainder=True)
    dataset = dataset.map(complex_to_log_minmax_norm_specs)

    # label processing
    dataset = dataset.map(label_preprocess)
    return dataset.prefetch(AUTOTUNE)

def get_data_sizes(x, y):
    feat_shape = x.shape
    label_shape = [
        y[0].shape,
        y[1].shape,
        y[2].shape
    ]
    return feat_shape, label_shape


def load_data(path, save=True):
    try:
        x = joblib.load(open(path + '_sj_x.joblib','rb'))
    except:
        from da_data_utils import from_wav_to_dataset
        x, max_wav_len = from_wav_to_dataset(path, 'complex', pad=True, config=config)
        if save:
            joblib.dump(x, open(path + '_sj_x.joblib', 'wb'))

    try:
        labels = joblib.load(open(path + '_sj_y.joblib', 'rb'))
    except:
        from scipy.io import loadmat
        sr_over_r = 48000 // 16000
        labels = []
        hop_size = config.nfft // 2
        window_size = config.nfft
        if not os.path.exists(path + '/metadata_wavs.mat'):
            y = loadmat(path + '/angle.mat')['phi'][0]
            y = y[:450]
            for i in range(len(y)):
                angle = y[i]
                if angle == -1:
                    angle = 10
                else:
                    angle /= 20
                tmp = angle * np.ones(((max_wav_len//sr_over_r) // hop_size + 1, window_size))
                labels.append(tmp)
            y = np.stack(labels, 0).astype(np.uint8)
            # vadlabel = (y != 10)
            labels = y.min(-1, keepdims=True)
            vadlabel = labels
            slabel = labels.min(-2)
            labels = (vadlabel, labels, slabel)
        else:
            y = loadmat(path + '/metadata_wavs.mat')
            for i in range(len(y['phi'][0])):
                start = y['voice_start'][0][i]
                end = y['voice_end'][0][i]
                angle = y['phi'][0][i]
                
                if angle == -1:
                    angle = 10
                else:
                    angle /= 20

                tmp = 10 * np.ones(((max_wav_len//sr_over_r) // hop_size + 1, window_size)) 
                tmp_y = np.concatenate([10 * np.ones(start), angle * np.ones(end-start), 10 * np.ones(max_wav_len - end)], 0)
                tmp_y = tmp_y[::sr_over_r] # label resampling
                for k,j in enumerate(range(0, len(tmp_y), hop_size)):
                    _tmp = tmp_y[j:(j+window_size)]
                    tmp[k][:len(_tmp)] = _tmp
                labels.append(tmp)
            
            y = np.stack(labels, 0).astype(np.uint8)
            labels = y.min(-1)
            vadlabel = labels != 10
            slabel = labels.min(-1)
            labels = (vadlabel, labels, slabel)
        if save:
            joblib.dump(labels, open(path + '_sj_y.joblib', 'wb'))
    assert len(x) == len(labels[0])

    # filter -1
    if config.filter:
        tmp_x = []
        tmp_labels = []
        tmp_vad = []
        for idx, data in enumerate(x):
            if labels[-1][idx] != 10:
                tmp_x.append(data)
                tmp_labels.append(labels[1][idx])
                tmp_vad.append(labels[0][idx])
        x = np.stack(tmp_x, 0)
        labels = np.stack(tmp_labels, 0)
        vadlabel = np.stack(tmp_vad, 0)
        slabel = labels.min(-2)
        labels = (vadlabel, labels, slabel)

    return x, labels



if __name__ == "__main__":
    print(config)

    TOTAL_EPOCH = config.epoch
    BATCH_SIZE = config.batch
    NAME = config.name if config.name.endswith('.h5') else config.name + '.h5'
    if config.filter:
        NAME = NAME[:-3] + '_filter.h5'
    N_CLASSES = 11

    def dataload(cl):
        def _dataload(path):
            data = joblib.load(open(path, 'rb'))
            data = data.reshape((data.shape[-2], data.shape[-1] // 2, 2)).view(np.float32)
            return data

        def _labelload(path):
            label = joblib.load(open(path, 'rb'))
            return label

        with ThreadPoolExecutor(20) as pool:
            files = list(pool.map(_dataload, sorted(glob(cl.get_unnormalized_feat_dir()+'/*.joblib'))))
            labels = list(pool.map(_labelload, sorted(glob(cl.f_dir+'/*.joblib'))))
        x = np.stack(files, 0)
        y = np.stack(labels, 0)
        del files, labels
        vadlabel = y != 10
        label = y.min(-1, keepdims=True)
        slabel = label.min(-2)
        return x, (vadlabel, label, slabel)
        

    # cl = FeatureClass(config.nfft, config=config, datasettype='train')
    # x, y = dataload(cl)
    # data_in, data_out = get_data_sizes(x, y)
    # train_data = to_dataset(x, y, config, train=True)
    # cl = FeatureClass(config.nfft, config=config, datasettype='test')
    # x, y = dataload(cl)
    # test_data = to_dataset(x, y, config, train=False)

    """ DATA """
    # TRAIN DATA
    PATH = '/root/datasets/ai_challenge/seldnet/seld/train'
    x, y = load_data(PATH)
    data_in, data_out = get_data_sizes(x, y)
    train_data = da_to_dataset(x, y, config, train=True)

    # VALIDATION DATA
    VAL_PATH = '/root/datasets/ai_challenge/seldnet/seld/validation'
    val_x, val_y = load_data(VAL_PATH)
    validation_data = da_to_dataset(val_x, val_y, config, train=False)

    """ MODEL """
    model = keras_model.da_get_model(data_in=(data_in[1], data_in[2], data_in[3] // 2), data_out=data_out, dropout_rate=config.dropout_rate,
                                  nb_cnn2d_filt=config.nb_cnn2d_filt, pool_size=[int(i) for i in config.pool_size.split(',')],
                                  rnn_size=[int(i) for i in config.rnn_size.split(',')], fnn_size=[int(i) for i in config.fnn_size.split(',')],
                                  classification_mode=config.mode, weights=[int(i) for i in config.loss_weights.split(',')])
    
    if config.resume:
        pdb.set_trace()
        model.load_weights(NAME)
        print('loadded pretrained model')

    """ TRAINING """
    callbacks = [
        # CSVLogger(NAME.replace('.h5', '.log'), append=True),
        ModelCheckpoint(NAME,
                        monitor='val_doa_s_out_mean_absolute_error',
                        mode='min',
                        save_best_only=True),
        TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(log_dir=f'./tensorboard_log/{NAME}'),
        EarlyStopping(monitor='val_loss', patience=config.patience),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, verbose=1)
    ]
    # if not config.pretrain:
    #     callbacks.append(
    #         LearningRateScheduler(
    #             custom_scheduler(4096, TOTAL_EPOCH/12, config.lr_div)))

    # for x,y in train_data.take(2):
    #     yy = model(x)
    #     pdb.set_trace()
    
    model.fit(train_data,
              epochs=TOTAL_EPOCH,
              validation_data=validation_data,
              steps_per_epoch=data_in[0] // config.batch,
              callbacks=callbacks)

    os.system(f'CUDA_VISIBLE_DEVICES={int(config.gpus)} python evaluator.py --name {NAME}.h5')