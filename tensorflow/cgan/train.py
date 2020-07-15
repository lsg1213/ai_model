import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
import model

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

AUTOTUNE = tf.data.experimental.AUTOTUNE


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='model')
args.add_argument('--model', type=str, default='')
args.add_argument('--lr', type=float, default=0.2)
args.add_argument('--opt', type=str, default='adam')
args.add_argument('--gpus', type=str, default='0,1,2,3')
args.add_argument('--epoch', type=int, default=50)
args.add_argument('--decay', type=float, default=0.95)
args.add_argument('--batch', type=int, default=4096)
args.add_argument('--norm', action='store_true')
args.add_argument('--dataset', type=str, default='')



if __name__ == "__main__":
    config = args.parse_args()
    print(config, '\n')
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus

    devices = ['/device:GPU:{}'.format(i) for i in device_nums]
    strategy = tf.distribute.MirroredStrategy() # devices)
    

    """ HYPER_PARAMETERS """
    BATCH_SIZE = config.batch
    TOTAL_EPOCH = config.epoch
    VAL_SPLIT = 0.15
    
    """ DATA """
    # 1. IMPORTING TRAINING DATA
    PATH = '/datasets/'
    TRAINPATH = None
    if config.dataset == '':
        if config.norm:
            TRAINPATH = os.path.join(PATH, '') 
        else:
            TRAINPATH = os.path.join(PATH, '')
    else:
        raise ValueError('wrong dataset select')
    
    
    x, val_x = [], []
    y, val_y = [], []

    

    # fix mismatch 

    """ MODEL """
    with strategy.scope(): 
        print(config.model)

        model = getattr(models, config.model)(
            input_shape=())
            # kernel_regularizer=tf.keras.regularizers.l2(1e-5))

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                        config.lr,
                                        decay_steps=TOTAL_EPOCH,
                                        decay_rate=config.decay,
                                        staircase=True)
        opt = str(config.opt).upper()
        lr_schedule = config.lr
        if opt == 'SGD':
            opt = SGD(learning_rate=lr_schedule,momentum=0.9)
        elif opt == 'ADAM':
            opt = Adam(learning_rate=lr_schedule)
        elif opt == 'ADAGRAD':
            opt = Adagrad(learning_rate=lr_schedule)
        else:
            raise ValueError('wrong optimizer')

        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'AUC'])
        # model.summary()

    """ DATA """
    # 2. DATA PRE-PROCESSING
    
    print("data pre-processing finished")

    # 2.1. SHUFFLING TRAINING DATA
    # perm = np.random.permutation(len(x))
    # x = np.take(x, perm, axis=0)
    # y = np.take(y, perm, axis=0)
    print("shuffling training data finished")

    # 3. TRAINING
    with strategy.scope(): 
        """ TRAINING """
        train_dataset = 
        val_dataset = 

        callbacks = [
            # ReduceLROnPlateau(monitor='val_AUC',
            #                   factor=0.9,
            #                   patience=1,
            #                   mode='max',
            #                   verbose=1,
            #                   min_lr=1e-5),
            EarlyStopping(monitor='val_AUC',
                          mode='max',
                          patience=3),
            CSVLogger(config.name + '.log',
                      append=True),
            ModelCheckpoint(config.name+'.h5',
                            monitor='val_AUC',
                            mode='max',
                            save_best_only=True),
            TerminateOnNaN(),
            # ModelCheckpoint(checkpoint_path,
            #                 monitor='val_AUC',
            #                 mode='max',
            #                 verbose=0)
        ]

        model.fit(train_dataset,
                  epochs=TOTAL_EPOCH,
                  validation_data=val_dataset,
                  steps_per_epoch=len(x)//BATCH_SIZE,
                  callbacks=callbacks)

