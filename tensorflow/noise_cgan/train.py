import argparse
import numpy as np
import os
import pickle
import tensorflow as tf
import time
from functools import partial
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from utils  import get_data
import models
from utils import *


AUTOTUNE = tf.data.experimental.AUTOTUNE


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='bdnn')
args.add_argument('--pretrain', type=str, default='')
args.add_argument('--pad_size', type=int, default=19)
args.add_argument('--step_size', type=int, default=9)
args.add_argument('--model', type=str, default='bdnn')
args.add_argument('--lr', type=float, default=0.1)
args.add_argument('--gpus', type=str, default='2,3')
args.add_argument('--skip', type=int, default=2)
args.add_argument('--aug', type=str, default='min')


def make_callable(value):
    def _callable(inputs):
        return value
    return _callable


class Trainer:
    def __init__(self, 
                 model, 
                 optimizer, 
                 strategy, 
                 batch_size=256):
        self.model = model
        self.optimizer = optimizer
        self.strategy = strategy
        self.batch_size = batch_size

        # Loss
        self.loss = tf.keras.losses.binary_crossentropy

        # Metrics
        self.train_metrics = [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC()]
        self.test_metrics = [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC()]

        self.train_loss_metric = tf.keras.metrics.Sum()
        self.test_loss_metric = tf.keras.metrics.Sum()

    @tf.function
    def distributed_train_epoch(self, dataset):
        total_loss = 0.
        num_train_batches = 0.

        for minibatch in dataset:
            per_replica_loss = self.strategy.experimental_run_v2(
                self.train_step, args=(minibatch,))
            total_loss += strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
            num_train_batches += 1

        return total_loss, num_train_batches

    def train_step(self, inputs):
        image, label = inputs

        with tf.GradientTape() as tape:
            predictions = self.model(image, training=True)

            ''' Loss Calculation (modify here) '''
            loss = tf.reduce_sum(self.loss(label, predictions)) / self.batch_size
            loss += (sum(self.model.losses) / self.strategy.num_replicas_in_sync)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                                           self.model.trainable_variables))

        for metric in self.train_metrics:
            metric(label, predictions)
        self.train_loss_metric(loss)
        return loss

    @tf.function
    def distributed_test_epoch(self, dataset):
        num_test_batches = 0.

        for minibatch in dataset:
            strategy.experimental_run_v2(self.test_step, args=(minibatch,))
            num_test_batches += 1
        return self.test_loss_metric.result(), num_test_batches

    def test_step(self, inputs):
        image, label = inputs
        predictions = self.model(image, training=False)
        unscaled_test_loss = self.loss(label, predictions) \
                             + sum(self.model.losses)

        for metric in self.test_metrics:
            metric(label, predictions)
        self.test_loss_metric(unscaled_test_loss)

    def fit(self, train_set, test_set, epochs, lr_scheduler=0.01):
        if not callable(lr_scheduler):
            lr_scheduler = make_callable(lr_scheduler)

        timer = time.time()
        for epoch in range(epochs):
            self.optimizer.learning_rate = lr_scheduler(epoch)

            train_total_loss, num_train_batches = self.distributed_train_epoch(train_set)
            test_total_loss, num_test_batches = self.distributed_test_epoch(test_set)

            print('Epoch: {}, Train Loss: {}, Train Accuracy: {}, '
                  'Test Loss: {}, Test Accuracy {}, {} seconds'.format(
                      epoch,
                      train_total_loss / num_train_batches,
                      self.train_metrics[0].result(),
                      test_total_loss / num_test_batches,
                      self.test_metrics[0].result(),
                      time.time() - timer))

            timer = time.time()

            if epoch != epochs - 1: # End of Epochs
                for metric in self.train_metrics:
                    metric.reset_states()
                for metric in self.test_metrics:
                    metric.reset_states()

        result = [train_total_loss / num_train_batches]
        for metric in self.train_metrics:
            result.append(metric.result().numpy())
        result.append(test_total_loss / num_test_batches)
        for metric in self.test_metrics:
            result.append(metric.result().numpy())

        return result


def prepare_dataset(dataset, is_train, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    # dataset = dataset.map(_preprocess, num_parallel_calls=AUTOTUNE)
    if is_train:
        dataset = dataset.shuffle(buffer_size=50000)
    return dataset.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)


if __name__ == '__main__':
    config = args.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    strategy = tf.distribute.MirroredStrategy()

    """ HYPER_PARAMETERS """
    BATCH_SIZE = 4096 # per each GPU
    TOTAL_EPOCH = 100

    WINDOW_SIZE = int(2*(config.pad_size-1)/config.step_size + 3)

    """ DATA """
    # 1. IMPORTING TRAINING DATA
    ORG_PATH = '/datasets/ai_challenge/' # inside a docker container
    if not os.path.isdir(ORG_PATH): # outside... 
        ORG_PATH = '/media/data1/datasets/ai_challenge/'

    x, val_x = [], []
    y, val_y = [], []

    # 1.1 training set
    PATH = os.path.join(ORG_PATH, 'TIMIT_sound_norm')
    for snr in ['-15', '-5', '5', '15']:
        x += pickle.load(
            open(os.path.join(PATH, 
                              # f'train/snr{snr}_10_no_noise_aug.pickle'),
                              f'snr{snr}_10_no_noise_aug.pickle'),
                 'rb'))
        y += pickle.load(
            open(os.path.join(PATH, f'label_10.pickle'), 'rb'))

    # 1.2 validation set
    PATH = os.path.join(ORG_PATH, 'TIMIT_noisex_norm')
    for snr in ['-20', '-10', '0', '10', '20']:
        val_x += pickle.load(
            open(os.path.join(PATH, f'test/snr{snr}.pickle'), 'rb'))

        val_y += pickle.load(open(os.path.join(PATH, 'test/phn.pickle'), 'rb'))

    # 1.3 fix mismatch 
    for i in range(len(x)):
        x[i] = x[i][:, :len(y[i])]

    """ MODEL """
    # 2. model definition
    with strategy.scope(): 
        time, freq = WINDOW_SIZE, x[0].shape[0]
        input_shape = (WINDOW_SIZE, freq)

        print(config.model)
        if config.pretrain != '':
            model = tf.keras.models.load_model(config.pretrain, compile=False)
        else:
            model = getattr(models, config.model)(
                input_shape=input_shape,
                kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        model.summary()
        optimizer = SGD(0.1, momentum=0.9)

    """ DATA """
    # 3. DATA PRE-PROCESSING
    x = np.concatenate(
        list(map(preprocess_spec(config, skip=config.skip), x)), axis=0)
    val_x = np.concatenate(
        list(map(preprocess_spec(config), val_x)), axis=0)

    # 3.1 sequence to window
    if model.output_shape[-1] != 1: # win2win
        y = np.concatenate(
            list(map(label_to_window(config, skip=config.skip), y)), axis=0)
        val_y = np.concatenate(
            list(map(label_to_window(config), val_y)), axis=0)
    else: # win2one
        y = np.concatenate(y, axis=0)
        val_y = np.concatenate(val_y, axis=0)
    print("data pre-processing finished")

    # 3.2 shuffling
    perm = np.random.permutation(len(x))
    x = np.take(x, perm, axis=0)
    y = np.take(y, perm, axis=0)
    print("shuffling training data finished")

    # 3.3 CVAL
    cval = getattr(x, config.aug)()
    print(f'CVAL: {cval}')

    with strategy.scope():
        train_set, test_set = tf.keras.datasets.cifar10.load_data()
        train_set = prepare_dataset(train_set, True, BATCH_SIZE)
        test_set = prepare_dataset(test_set, False, BATCH_SIZE)

    trainer = Trainer(model, optimizer, strategy, BATCH_SIZE)
    trainer.fit(train_set, test_set, 5, lr_scheduler=0.1)
    model.save('dist_model.h5')
