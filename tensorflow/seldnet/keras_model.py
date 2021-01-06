#
# The SELDnet architecture
#

from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import pdb
import tensorflow as tf

def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
                                rnn_size, fnn_size, classification_mode, weights):

    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))
    spec_cnn = spec_start
    for i, convCnt in enumerate(pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    # spec_cnn = Permute((2, 1, 3))(spec_cnn)
    
    # spec_rnn = Reshape(([spec_cnn.shape[1].value, spec_cnn.shape[2].value * spec_cnn.shape[3].value]))(spec_cnn)
    spec_rnn = Reshape(([spec_cnn.shape[1], spec_cnn.shape[2] * spec_cnn.shape[3]]))(spec_cnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True), merge_mode='mul')(spec_rnn)

    # SED
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)
    
    doa = sed * doa

    doa = TimeDistributed(Dense(11))(doa)
    doa_s = Bidirectional(GRU(32))(doa)
    doa_s = Dense(11)(doa_s)
    doa_s = Activation('softmax', name='doa_s_out')(doa_s)
    doa = Activation('softmax', name='doa_out')(doa)

    

    model = Model(inputs=spec_start, outputs=[sed, doa, doa_s])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'], loss_weights=weights,
    metrics={'sed_out':'AUC', 'doa_out': tf.keras.metrics.MeanAbsoluteError(), 'doa_s_out': tf.keras.metrics.MeanAbsoluteError()})

    model.summary()
    return model

def da_get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
                                rnn_size, fnn_size, classification_mode, weights):
    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))
    spec_cnn = spec_start
    for i, convCnt in enumerate(pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    # spec_cnn = Permute((2, 1, 3))(spec_cnn)
    
    # spec_rnn = Reshape(([spec_cnn.shape[1].value, spec_cnn.shape[2].value * spec_cnn.shape[3].value]))(spec_cnn)
    spec_rnn = Reshape(([spec_cnn.shape[1], spec_cnn.shape[2] * spec_cnn.shape[3]]))(spec_cnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True), merge_mode='mul')(spec_rnn)

    # SED
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)
    
    # doa = sed * doa

    doa = TimeDistributed(Dense(11))(doa)
    doa_s = Bidirectional(GRU(32))(doa)
    doa_s = Dense(11)(doa_s)
    doa_s = Activation('softmax', name='doa_s_out')(doa_s)
    doa = Activation('softmax', name='doa_out')(doa)

    

    model = Model(inputs=spec_start, outputs=[sed, doa, doa_s])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'], loss_weights=weights,
    metrics={'sed_out':'AUC', 'doa_out': tf.keras.metrics.MeanAbsoluteError(), 'doa_s_out': tf.keras.metrics.MeanAbsoluteError()})

    model.summary()
    return model
