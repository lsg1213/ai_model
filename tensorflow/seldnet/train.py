import os, argparse, pdb, joblib
import numpy as np
from utils import create_folder
import models
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
import datetime
import cls_feature_class, cls_data_generator
from glob import glob
from tensorflow.keras.optimizers import Adam

AUTOTUNE = tf.data.experimental.AUTOTUNE

def getparam():
    args = argparse.ArgumentParser()
    args.add_argument('--name', type=str, default='test')
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--gpus', type=str, default='-1')
    args.add_argument('--epoch', type=int, default=200)
    args.add_argument('--resume', action='store_true')
    args.add_argument('--skip', type=int, default=1)
    args.add_argument('--decay', type=float, default=1/np.sqrt(2))
    args.add_argument('--db', type=int, default=30)
    args.add_argument('--batch', type=int, default=32)
    args.add_argument('--seq_len', type=int, default=64)
    args.add_argument('--nfft', type=int, default=512)
    
    args.add_argument('--dropout_rate', type=float, default=0.0)
    args.add_argument('--nb_cnn2d_filt', type=int, default=64)
    args.add_argument('--pool_size', type=str, default='8,8,2')
    args.add_argument('--rnn_size', type=str, default='128,128')
    args.add_argument('--fnn_size', type=int, default=128)
    args.add_argument('--loss_weights', type=str, default='1,50')


    return args.parse_args()

def list_element_to_int(li):
    return [int(i) for i in li]
def list_element_to_float(li):
    return [float(i) for i in li]

def angle_to_number(label):
    for i, j in enumerate(label):
        if j == -1:
            label[i] = 10
        else:
            label[i] = int(label[i] / 20)
        if not (label[i] in [0,1,2,3,4,5,6,7,8,9,10]):
            print(label[i])
            raise ValueError('label has a problem')
    return label

def split_in_seqs(data, config):
    if len(data.shape) == 1:
        if data.shape[0] % config.seq_len:
            data = data[:-(data.shape[0] % config.seq_len)]
        data = data.reshape((data.shape[0] // config.seq_len, config.seq_len, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % config.seq_len:
            data = data[:-(data.shape[0] % config.seq_len), :]
        data = data.reshape((data.shape[0] // config.seq_len, config.seq_len, data.shape[1]))
    elif len(data.shape) == 3:
        if data.shape[0] % config.seq_len:
            data = data[:-(data.shape[0] % config.seq_len), :, :]
        data = data.reshape((data.shape[0] // config.seq_len, config.seq_len, data.shape[1], data.shape[2]))
    else:
        print('ERROR: Unknown data dimensions: {}'.format(data.shape))
        exit()
    return data


def get_data(config, train=True):
    feat_cls = cls_feature_class.FeatureClass(config.nfft)
    gen_cls = cls_data_generator.DataGenerator(config, shuffle=train, train=train)
    label_dir = feat_cls.get_label_dir()
    feat_dir = feat_cls.get_unnormalized_feat_dir()

    data = gen_cls.data
    label = np.array(angle_to_number(gen_cls.label))
    for i in range(len(data)):
        data[i] = np.reshape(data[i], (gen_cls.nb_frames_file, gen_cls.feat_len, gen_cls._2_nb_ch))
    data = np.array(data)
    data = np.transpose(data, (0, 3, 1, 2))
    sedlabel = np.array([0 if i == 10 else 1 for i in label]) # sed label
    label = tf.data.Dataset.from_tensor_slices((sedlabel, label))
    # label = [sedlabel, label]
    
    data_in = (config.batch, gen_cls._2_nb_ch, gen_cls.nb_frames_file, gen_cls.feat_len)
    data_out = (config.batch, (1, gen_cls.nb_classes))
    
    _data = tf.data.Dataset.from_tensor_slices(data)
    dataset = tf.data.Dataset.zip((_data, label))
    if train:
        dataset = dataset.repeat(1).shuffle(buffer_size=data.shape[0])
        dataset = dataset.batch(config.batch)
        dataset = dataset.prefetch(AUTOTUNE)
    else:
        dataset = dataset.batch(config.batch, drop_remainder=False)
    
    return dataset, data_in, data_out

def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    model_dir = 'models/'
    create_folder(model_dir)
    trainset, data_in, data_out = get_data(config)
    testset, _, _ = get_data(config, train=False)
    

    model = models.get_model(data_in=data_in, data_out=data_out, dropout_rate=config.dropout_rate,
                            nb_cnn2d_filt=config.nb_cnn2d_filt, pool_size=list_element_to_int(config.pool_size.split(',')),
                            rnn_size=list_element_to_int(config.rnn_size.split(',')), fnn_size=[config.fnn_size],
                            weights=list_element_to_float(config.loss_weights.split(',')))
    # pdb.set_trace()

    callbacks = [
            # ReduceLROnPlateau(monitor='val_loss',
            #                   factor=0.9,
            #                   patience=1, # 1,
            #                   mode='min',
            #                   verbose=1,
            #                   min_lr=1e-5),
            # EarlyStopping(monitor='val_loss',
            #               mode='min',
            #               patience=10), # 3),
            ModelCheckpoint(config.name+'.h5',
                            monitor='doa_out_acc',
                            mode='max',
                            save_best_only=True),
            TerminateOnNaN(),
            TensorBoard(log_dir='tensorboard_log/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')),

        ]
    
    model.compile(optimizer=Adam(learning_rate=config.lr), loss=['binary_crossentropy','sparse_categorical_crossentropy'], loss_weights=list_element_to_float(config.loss_weights.split(',')), metrics=['acc'])

    model.fit(trainset, epochs=config.epoch, validation_data=testset, batch_size=config.batch, callbacks=callbacks)
    
    pdb.set_trace()
    for i,j in trainset:
        print(model(i[0][tf.newaxis,...], training=False))
        print(j[0])
    pdb.set_trace()

if __name__ == "__main__":
    arg = getparam()
    main(arg)
    