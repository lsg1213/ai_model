#
# A wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_data_generator
import evaluation_metrics
import keras_model
import utils
import time, pdb
from params import getparam
from IPython import embed
from tensorboardX import SummaryWriter
plot.switch_backend('agg')
import tensorflow as tf

class dataGenerator():
    def __init__(self, data, sedlabel, doalabel, doaSlabel):
        self.data = data
        self.sedlabel = sedlabel
        self.doalabel = doalabel
        self.doaSlabel = doaSlabel

    def __call__(self):
        for i in range(self.data.shape[0].value):
            yield self.data[i], [self.sedlabel[i], self.doalabel[i], self.doaSlabel[i]]

def makedataset(gen, config, shuffle=False):
    testlabel = np.stack(gen.label,0).min(-1, keepdims=True)
    testvadlabel = testlabel != 10

    testdata = [data.reshape(gen.nb_frames_file, gen.feat_len, gen.feat_cls.get_nb_channels()).view(np.float32) for data in gen.data]
    testdata = tf.stack(testdata,0)

    testdataset = tf.data.Dataset.from_tensor_slices((testdata, (tf.convert_to_tensor(testvadlabel), tf.convert_to_tensor(testlabel), tf.reduce_min(testlabel, -2))))
    if shuffle:
        testdataset.shuffle(5000)
    testdataset = testdataset.batch(config.batch)
    return testdataset
def main(argv):
    config = getparam(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    """
    Main wrapper for training sound event localization and detection network.
    
    :param argv: expects two optional inputs. 
        first input: job_id - (optional) all the output files will be uniquely represented with this. (default) 1
        second input: task_id - (optional) To chose the system configuration in parameters.py. 
                                (default) uses default parameters
    """
    model_dir = f'model_save/{config.name}'
    utils.create_folder(model_dir)
    unique_name = f'{config.s}'
    unique_name = os.path.join(model_dir, unique_name)
    print("unique_name: {}\n".format(unique_name))
    
    data_gen_train = cls_data_generator.DataGenerator(
        config, shuffle=True, datasettype='train'
    )
    data_gen_test = cls_data_generator.DataGenerator(
        config, shuffle=False, datasettype='test'
    )
    if config.dataset != 'clean_iris':
        data_gen_validation = cls_data_generator.DataGenerator(
            config, shuffle=False, datasettype='validation'
        )
    else:
        data_gen_validation = data_gen_test
    

    

    data_in, data_out = data_gen_train.get_data_sizes()

    traindataset = makedataset(data_gen_train, config, shuffle=True)
    validationset = makedataset(data_gen_validation, config, shuffle=False)
    testdataset = makedataset(data_gen_test, config, shuffle=False)
    # pdb.set_trace()
    # c = 0
    # for i,j in data_gen_train.generate():
    #     pdb.set_trace()
    #     c += 1
    #     print(i,j)
    #     if c == 3:
    #         break
    # for i,j in traindataset.take(2):
    #     pdb.set_trace()
    #     print(i,j)
    # gt = collect_test_labels(data_gen_test, data_out, params['mode'], params['quick_test'])
    # sed_gt = evaluation_metrics.reshape_3Dto2D(gt[0])
    # doa_gt = evaluation_metrics.reshape_3Dto2D(gt[1])

    model = keras_model.get_model(data_in=data_in, data_out=data_out, dropout_rate=config.dropout_rate,
                                  nb_cnn2d_filt=config.nb_cnn2d_filt, pool_size=[int(i) for i in config.pool_size.split(',')],
                                  rnn_size=[int(i) for i in config.rnn_size.split(',')], fnn_size=[int(i) for i in config.fnn_size.split(',')],
                                  classification_mode=config.mode, weights=[int(i) for i in config.loss_weights.split(',')])
    best_metric = 99999
    conf_mat = None
    best_conf_mat = None
    best_epoch = -1
    patience_cnt = 0
    epoch_metric_loss = np.zeros(config.epoch)
    tr_loss = np.zeros(config.epoch)
    val_loss = np.zeros(config.epoch)
    test_loss = np.zeros(config.epoch)
    tr_sed_auc = np.zeros(config.epoch)
    val_sed_auc = np.zeros(config.epoch)
    test_sed_auc = np.zeros(config.epoch)
    tr_doa_abm = np.zeros(config.epoch)
    val_doa_abm = np.zeros(config.epoch)
    test_doa_abm = np.zeros(config.epoch)
    tr_doa_s_abm = np.zeros(config.epoch)
    val_doa_s_abm = np.zeros(config.epoch)
    test_doa_s_abm = np.zeros(config.epoch)
    nb_epoch = config.epoch
    tensorboard_path = 'tensorboard_log' + f'/{config.name}'
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)
    for epoch_cnt in range(nb_epoch):
        hist = model.fit(
            traindataset,
            validation_data=validationset,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.History()
            ]
        )
        tr_loss[epoch_cnt] = hist.history.get('loss')[-1]
        val_loss[epoch_cnt] = hist.history.get('val_loss')[-1]
        tr_sed_auc[epoch_cnt] = hist.history.get('sed_out_auc')[-1]
        val_sed_auc[epoch_cnt] = hist.history.get('val_sed_out_auc')[-1]
        tr_doa_abm[epoch_cnt] = hist.history.get('doa_out_mean_absolute_error')[-1]
        val_doa_abm[epoch_cnt] = hist.history.get('val_doa_out_mean_absolute_error')[-1]
        tr_doa_s_abm[epoch_cnt] = hist.history.get('doa_s_out_mean_absolute_error')[-1]
        val_doa_s_abm[epoch_cnt] = hist.history.get('val_doa_s_out_mean_absolute_error')[-1]
        
        # testgenerator = dataGenerator(testdata, tf.convert_to_tensor(testvadlabel), tf.convert_to_tensor(testlabel), tf.reduce_min(testlabel, -2))
        
        # testdataset = tf.data.Dataset.from_generator(testgenerator, output_types=(tf.float32, (tf.float32, tf.float32, tf.float32)), output_shapes=(tf.TensorShape([data_gen_test.nb_frames_file, data_gen_test.feat_len, data_gen_test.feat_cls.get_nb_channels() * 4]),tf.TensorShape((tf.TensorShape([data_gen_test.nb_frames_file, 1]), tf.TensorShape([data_gen_test.nb_frames_file, 1]), tf.TensorShape([1,])))))
        
        
        testhist = model.evaluate(testdataset)
        test_loss[epoch_cnt] = testhist[0]
        test_sed_auc[epoch_cnt] = testhist[-3]
        test_doa_abm[epoch_cnt] = testhist[-2]
        test_doa_s_abm[epoch_cnt] = testhist[-1]
        
        writer.add_scalar('train/train_loss', tr_loss[epoch_cnt], epoch_cnt)
        writer.add_scalar('train/sed_auc', tr_sed_auc[epoch_cnt], epoch_cnt)
        writer.add_scalar('train/doa_out_mean_absolute_error', tr_doa_abm[epoch_cnt], epoch_cnt)
        writer.add_scalar('train/doa_s_out_mean_absolute_error', tr_doa_s_abm[epoch_cnt], epoch_cnt)
        writer.add_scalar('val/val_loss', val_loss[epoch_cnt], epoch_cnt)
        writer.add_scalar('val/sed_auc', val_sed_auc[epoch_cnt], epoch_cnt)
        writer.add_scalar('val/doa_out_mean_absolute_error', val_doa_abm[epoch_cnt], epoch_cnt)
        writer.add_scalar('val/doa_s_out_mean_absolute_error', val_doa_s_abm[epoch_cnt], epoch_cnt)
        writer.add_scalar('test/test_loss', test_loss[epoch_cnt], epoch_cnt)
        writer.add_scalar('test/sed_auc', test_sed_auc[epoch_cnt], epoch_cnt)
        writer.add_scalar('test/doa_out_mean_absolute_error', test_doa_abm[epoch_cnt], epoch_cnt)
        writer.add_scalar('test/doa_s_out_mean_absolute_error', test_doa_s_abm[epoch_cnt], epoch_cnt)

        patience_cnt += 1
        if test_loss[epoch_cnt] < best_metric:
            best_metric = test_loss[epoch_cnt]
            # best_conf_mat = conf_mat
            best_epoch = epoch_cnt
            model.save(f'{model_dir}/{epoch_cnt}_{best_metric}_model.h5')
            patience_cnt = 0

        if patience_cnt > config.patience:
            print('early stop!')
            break
            
    pdb.set_trace()
    res = model.predict(data_gen_train.generate())

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
