import argparse
import os
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import *
from da_utils import *
from da_transforms import *
import keras_model
from train import load_data, complex_to_log_minmax_norm_specs


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--model', type=str, default='EfficientNetB0')



if __name__ == '__main__':
    from params import getparam
    import sys, pdb
    config = getparam(sys.argv[1:])

    N_CLASSES = 11

    
    

    PATH = '/root/datasets/ai_challenge/seldnet/seld/test'
    x, y = load_data(PATH, save=False)
    x = complex_to_log_minmax_norm_specs(x)
    # y = degree_to_class(y, one_hot=False)
    def get_data_sizes(x, y):
        feat_shape = x.shape
        label_shape = [
            y[0].shape,
            y[1].shape,
            y[2].shape
        ]
        return feat_shape, label_shape
    data_in, data_out = get_data_sizes(x,y)
    # 1. Loading a saved model
    model = keras_model.da_get_model(data_in=(data_in[1],data_in[2],data_in[3]), data_out=data_out, dropout_rate=config.dropout_rate,
                                  nb_cnn2d_filt=config.nb_cnn2d_filt, pool_size=[int(i) for i in config.pool_size.split(',')],
                                  rnn_size=[int(i) for i in config.rnn_size.split(',')], fnn_size=[int(i) for i in config.fnn_size.split(',')],
                                  classification_mode=config.mode, weights=[int(i) for i in config.loss_weights.split(',')])
    model.load_weights(config.name)
    # tf.keras.models.save_model(model, f'model_save/best_57.1.hdf5')
    # exit()
    # 3. predict
    y_hat = model.predict(x)[2]
    y_hat_cls = np.argmax(y_hat, axis=-1).astype(np.float32)

    # print("GROUND TRUTH\n", y)
    # print("PREDICTIONS\n", y_hat_cls)
    y = y[2]
    print("Accuracy:", Accuracy()(y, y_hat_cls).numpy())
    print("MAE:", np.mean(tf.abs(y_hat_cls - y.squeeze(-1))))
    print('angle:', np.mean(tf.abs(y_hat_cls - y.squeeze(-1))) * 20)
    print(confusion_matrix(y, y_hat_cls))
    
