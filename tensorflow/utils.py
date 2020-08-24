import tensorflow as tf
import numpy as np
from models import resnet_v1


def pad(x):
    return np.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)))


def mean_average(base_model, obj_model, alpha=0.9):
    base_model.set_weights(
        [w*alpha + o_w*(1-alpha)
        for w, o_w in zip(base_model.get_weights(), obj_model.get_weights())]
    )


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = (x_train / 255.).astype(np.float32)
    x_test = (x_test / 255.).astype(np.float32)

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    return (x_train, y_train), (x_test, y_test)
