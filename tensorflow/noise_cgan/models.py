import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
import tensorflow.keras.backend as K


def dnn(input_shape, dropout_rate=0.5, **kwargs):
    model_input = Input(shape=input_shape)

    x = Flatten()(model_input)
    for i in range(2):
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    x = Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=model_input, outputs=x)


def build_generator(output_shape=shape, class_num=class_num, stddev=0.2, z_dim=noise_dim, **kwargs):
    noise = Input(shape=(z_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(class_num, z_dim)(label))

    model_input = Concatenate()([noise, label_embedding])

    x = Dense(400, activation='relu')(model_input)
    if tf.rank(x) == 2:
        x = tf.expand_dims(x, axis=1)
    x = LSTM(100, return_sequences=True, kernel_initializer='he_normal')(x)

    if feature == 'seq':
        x = Dense(output_shape[0], activation='tanh')(x)
        x = Flatten()(x)
        output = Reshape(output_shape)(x)
    else:
        x = Dense(output_shape[0]*output_shape[1], activation='tanh')(x)
        output = Reshape(output_shape)(x)

    return Model([noise, label], output)

def build_discriminator(input_shape=shape, class_num=class_num, stddev=0.2):
    noise_input = Input(shape=input_shape)
    reshaped_noise = Flatten()(noise_input)

    noise = Input(shape=input_shape)
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(Embedding(class_num, np.prod(input_shape))(label))
    flat_noise = Flatten()(noise)

    x = Multiply()([flat_noise, label_embedding])
    if tf.rank(x) == 2:
        x = tf.expand_dims(x, axis=-1)
    x = LSTM(100, kernel_initializer='he_normal')(x)
    output = Dense(2, activation='softmax')(x)
    

    return Model([noise, label], output)