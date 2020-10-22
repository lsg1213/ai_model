import os, joblib, pdb, argparse
import numpy as np
import cls_feature_class
from glob import glob
import tensorflow as tf
from collections import deque

class DataGenerator(object):
    def __init__(self, config, shuffle=True, train=True):
        self.config = config
        self.shuffle = shuffle
        self.feat_cls = cls_feature_class.FeatureClass(config.nfft)
        self.label_dir, self.flabel_dir = self.feat_cls.get_label_dir()
        self.feat_dir = self.feat_cls.get_unnormalized_feat_dir()
        
        if train:
            self.data = [joblib.load(open(i,'rb')) for i in sorted(glob(self.feat_dir + '/*train_x*'))]
            self.label = [joblib.load(open(i,'rb')) for i in sorted(glob(self.label_dir + '/*train_y.*'))]
            self.flabel = [joblib.load(open(i,'rb')) for i in sorted(glob(self.flabel_dir + '/*train_y_window*'))]
        else:
            self.data = [joblib.load(open(i,'rb')) for i in sorted(glob(self.feat_dir + '/*test_x*'))]
            self.label = [joblib.load(open(i,'rb')) for i in sorted(glob(self.label_dir + '/*test_y.*'))]
            self.flabel = [joblib.load(open(i,'rb')) for i in sorted(glob(self.flabel_dir + '/*test_y_window*'))]
        self.data = [x for y in self.data for x in y]
        self.label = [x for y in self.label for x in y]
        self.flabel = [x for y in self.flabel for x in y]
        
        self.filenames_list = tf.range(len(self.data))
        self.nb_frames_file = len(self.data[0])
        self._2_nb_ch = 2 * self.feat_cls.get_nb_channels()
        self.feat_len = self.data[0].shape[1] // self._2_nb_ch
        self.label_len = len(self.label)
        self.classes = self.feat_cls.get_classes()
        self.nb_classes = len(self.classes)
        self.doa_len = (self.label_len - self.nb_classes)//self.nb_classes
        self.default_azi = self.feat_cls.get_default_azi_ele_regr()
        self.perm = tf.range(self.nb_frames_file)

        self.batch_seq_len = config.batch * config.seq_len

        self.nb_total_batches = int(np.floor((len(self.filenames_list) * self.nb_frames_file /
                                                float(self.batch_seq_len))))
        
        print(
            'nb_files: {}, nb_classes:{}\n'
            'nb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                len(self.filenames_list),  self.nb_classes,
                self.nb_frames_file, self.feat_len, self._2_nb_ch, self.label_len
                )
        )

        print(
            'batch_size: {}, seq_len: {}, shuffle: {}\n'
            'label_dir: {}\n '
            'feat_dir: {}\n'.format(
                config.batch, config.seq_len, self.shuffle,
                self.label_dir, self.feat_dir
            )
        )

def getparam():
    args = argparse.ArgumentParser()
    args.add_argument('--name', type=str, default='test')
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--gpus', type=str, default='0,1,2,3')
    args.add_argument('--epoch', type=int, default=50)
    args.add_argument('--resume', action='store_true')
    args.add_argument('--skip', type=int, default=1)
    args.add_argument('--decay', type=float, default=1/np.sqrt(2))
    args.add_argument('--db', type=int, default=30)
    args.add_argument('--batch', type=int, default=32)
    args.add_argument('--seq_len', type=int, default=64)
    args.add_argument('--nfft', type=int, default=512)
    
    return args.parse_args()
if __name__ == "__main__":
    a = DataGenerator(getparam(),train=False)
    a.generate()