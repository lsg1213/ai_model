import os, argparse, pdb, joblib
import numpy as np
from utils import create_folder, terminateOnNaN
import models
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime
import cls_feature_class, cls_data_generator
from glob import glob
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from tensorboardX import SummaryWriter
from params import getparam
AUTOTUNE = tf.data.experimental.AUTOTUNE



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

def get_data(config, train=True):
    feat_cls = cls_feature_class.FeatureClass(config.nfft)
    gen_cls = cls_data_generator.DataGenerator(config, shuffle=train, train=train)

    data = gen_cls.data
    label = np.array(angle_to_number(gen_cls.label))
    flabel = np.array(gen_cls.flabel)
    for i in range(len(data)):
        data[i] = np.reshape(data[i], (gen_cls.nb_frames_file, gen_cls.feat_len, gen_cls._2_nb_ch))
    data = np.array(data)
    data = np.transpose(data, (0, 3, 1, 2))
    
    label = tf.data.Dataset.from_tensor_slices((label, flabel))
    # label = [sedlabel, label]
    
    data_in = (config.batch, gen_cls._2_nb_ch, gen_cls.nb_frames_file, gen_cls.feat_len)
    data_out = (config.batch, (1, gen_cls.nb_classes))
    
    _data = tf.data.Dataset.from_tensor_slices(data)
    dataset = tf.data.Dataset.zip((_data, label))
    if train:
        dataset = dataset.repeat(1).shuffle(buffer_size=data.shape[0])
        dataset = dataset.batch(config.batch, drop_remainder=True)
        dataset = dataset.prefetch(AUTOTUNE)
    else:
        dataset = dataset.batch(config.batch, drop_remainder=False)
    
    return dataset, data_in, data_out



def unique(data):
    return tf.unique(data)[0]

# @tf.function
def get_1_from_frame(fdata, config):
    if tf.rank(fdata) > 2 and fdata.shape[-1] > 1:
        # fdata = (batch, frame, softmax > 1)
        fdata = tf.reduce_max(fdata,-1)
    tens = tf.ones(fdata.shape[:2], dtype=fdata.dtype) * 10
    
    # fdata = (batch, frame)
    
    data = tf.map_fn(unique, tf.where(fdata > config.th, fdata, tens))
    if tf.rank(data) == 1:
        return tf.cast(tf.ones((config.batch,), dtype=fdata.dtype) * 10, dtype=tf.int64)
    else:
        data = data[:-1]
        return tf.cast(tf.ones((config.batch,), dtype=fdata.dtype) * tf.sort(data)[-1], dtype=tf.int64)


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    model_dir = 'models/'
    create_folder(model_dir)
    trainset, data_in, data_out = get_data(config)
    testset, _, _ = get_data(config, train=False)
    tensorboard_path = 'tensorboard_log/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)

    model = models.get_model(data_in=data_in, data_out=data_out, dropout_rate=config.dropout_rate,
                            nb_cnn2d_filt=config.nb_cnn2d_filt, pool_size=list_element_to_int(config.pool_size.split(',')),
                            rnn_size=list_element_to_int(config.rnn_size.split(',')), fnn_size=[config.fnn_size], config=config)

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
        ]
    
    # model.compile(optimizer=Adam(learning_rate=config.lr), loss=['binary_crossentropy','sparse_categorical_crossentropy'], loss_weights=list_element_to_float(config.loss_weights.split(',')), metrics=['acc'])

    # model.fit(trainset, epochs=config.epoch, validation_data=testset, batch_size=config.batch, callbacks=callbacks)
    optimizer = Adam(learning_rate=config.lr)
    startepoch = 0
    bce = tf.keras.losses.BinaryCrossentropy()
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    weight = list_element_to_float(config.loss_weights.split(','))
    sedacc = tf.keras.metrics.Accuracy()
    doaacc = tf.keras.metrics.Accuracy()
    maxacc = 0.
    for epoch in range(startepoch, config.epoch):
        with tqdm(trainset) as pbar:
            for idx, (x, y) in enumerate(pbar):
                label = tf.cast(y[0], dtype=tf.int64) # (batch, label)
                flabel = tf.cast(y[1], dtype=tf.int64) # (batch, frame, label)

                sedlabel = tf.cast(tf.ones_like(label) * 10 != label, dtype=tf.int64)
                sedflabel = tf.cast(tf.ones_like(flabel) * 10 != flabel, dtype=tf.int64)
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    logits = model(x, training=True) # logits = (sed, doa)
                    sedloss = bce(sedflabel, logits[0]) * weight[0]
                    doaloss = scce(flabel, logits[1]) * weight[1]
                    loss = sedloss + doaloss
                if config.mode == 'frame':
                    sedpred = tf.argmax(logits[0], -1) # (batch, framelabel)
                    doapred = logits[1] # (batch, framelabel)
                sedpred = tf.cast(get_1_from_frame(sedpred, config), dtype=sedlabel.dtype)
                doapred = tf.cast(get_1_from_frame(doapred, config), dtype=label.dtype)
                sedpred = tf.cast(sedpred != 10, dtype=sedpred.dtype)
                sedacc.update_state(sedlabel, sedpred)
                doaacc.update_state(label, doapred)
                
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                pbar.set_postfix(epoch=f'{epoch:3}', loss=f'{loss.numpy():0.4}', doaacc=f'{doaacc.result().numpy():0.4}', sedacc=f'{sedacc.result().numpy():0.4}')
        writer.add_scalar('train/total_loss',loss.numpy(),epoch)
        writer.add_scalar('train/sed_loss',sedloss.numpy(),epoch)
        writer.add_scalar('train/doa_loss',doaloss.numpy(),epoch)
        writer.add_scalar('train/sed_acc',sedacc.result().numpy(),epoch)
        writer.add_scalar('train/doa_acc',doaacc.result().numpy(),epoch)
        sedacc.reset_states()
        doaacc.reset_states()

        with tqdm(trainset) as pbar:
            for idx, (x, y) in enumerate(pbar):
                label = tf.cast(y[0], dtype=tf.int64) # (batch, label)
                flabel = tf.cast(y[1], dtype=tf.int64) # (batch, frame, label)

                sedlabel = tf.cast(tf.ones_like(label) * 10 != label, dtype=tf.int64)
                sedflabel = tf.cast(tf.ones_like(flabel) * 10 != flabel, dtype=tf.int64)
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    logits = model(x, training=False) # logits = (sed, doa)
                    sedloss = bce(sedflabel, logits[0]) * weight[0]
                    doaloss = scce(flabel, logits[1]) * weight[1]
                    loss = sedloss + doaloss
                if config.mode == 'frame':
                    sedpred = tf.argmax(logits[0], -1) # (batch, framelabel)
                    doapred = logits[1] # (batch, framelabel)
                sedpred = tf.cast(get_1_from_frame(sedpred, config), dtype=sedlabel.dtype)
                doapred = tf.cast(get_1_from_frame(doapred, config), dtype=label.dtype)
                sedpred = tf.cast(sedpred != 10, dtype=sedpred.dtype)
                
                pbar.set_postfix(epoch=f'{epoch:3}', val_loss=f'{loss.numpy():0.4}', val_doaacc=f'{doaacc.result().numpy():0.4}', val_sedacc=f'{sedacc.result().numpy():0.4}')
        writer.add_scalar('test/total_loss',loss.numpy(),epoch)
        writer.add_scalar('test/sed_loss',sedloss.numpy(),epoch)
        writer.add_scalar('test/doa_loss',doaloss.numpy(),epoch)
        writer.add_scalar('test/sed_acc',sedacc.result().numpy(),epoch)
        writer.add_scalar('test/doa_acc',doaacc.result().numpy(),epoch)
        sedacc.reset_states()
        doaacc.reset_states()

        

def train(model, dataset):
    pass
    

if __name__ == "__main__":
    import sys
    arg = getparam(sys.argv[1:])
    main(arg)
    