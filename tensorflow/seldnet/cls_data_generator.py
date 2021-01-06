import os, joblib, pdb, argparse, random
import numpy as np
import cls_feature_class
from glob import glob
import tensorflow as tf
from collections import deque

class DataGenerator(object):
    def __init__(self, config, datasettype, shuffle=True):
        self.config = config
        self.shuffle = shuffle
        self.feat_cls = cls_feature_class.FeatureClass(config.nfft, config=self.config, datasettype=datasettype)
        self.label_dir = self.feat_cls.f_dir
        self.feat_dir = self.feat_cls.get_unnormalized_feat_dir()
        
        self.data = [joblib.load(open(i,'rb')) for i in sorted(glob(self.feat_dir + '/*.joblib'))]
        self.label = [joblib.load(open(i,'rb')) for i in sorted(glob(self.label_dir + '/*.joblib'))]
        
        self.filenames_list = np.arange(len(self.data))
        self.nb_frames_file = len(self.data)
        self._2_nb_ch = 2 * self.feat_cls.get_nb_channels()
        self.classes = self.feat_cls.get_classes()
        self.nb_classes = len(self.classes)
        self.default_azi = self.feat_cls.get_default_azi_ele_regr()
        self.perm = np.arange(self.nb_frames_file)
        self.batch_seq_len = self.config.batch * self.config.seq_len
        
        self.nb_total_batches = int(np.floor((1000 * self.config.batch)))
        self._get_label_filenames_sizes()
        
        print(
            'nb_files: {}, nb_classes:{}\n'
            'nb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                self.filenames_list.shape[0],  self.nb_classes,
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
    
    def get_data_sizes(self):
        feat_shape = (self.config.batch, self.nb_frames_file, self.feat_len, self._2_nb_ch * 2)
        label_shape = [
            (self.config.batch, self.nb_frames_file, 1),
            (self.config.batch, self.nb_frames_file, self.nb_classes), 
            (self.config.batch, 1)
        ]
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self.nb_total_batches

    def _get_label_filenames_sizes(self):
        self.nb_frames_file = self.data[0].shape[0]
        self.feat_len = self.data[0].shape[-1] // self.feat_cls.get_nb_channels()

        self.label_len = self.label[0].shape[0]
        # self.doa_len = (self._label_len - self.nb_classes)//self.nb_classes
        return

    def generate(self):
        """
        Generates batches of samples
        :return: 
        """

        while 1:
            if self.shuffle:
                random.shuffle(self.filenames_list)

            # Ideally this should have been outside the while loop. But while generating the test data we want the data
            # to be the same exactly for all epoch's hence we keep it here.


            file_cnt = 0

            for i in range(self.nb_total_batches):
                feat = np.zeros((self.batch_seq_len, self.nb_frames_file, self.feat_len, 2 * self._2_nb_ch))
                label = np.zeros((self.batch_seq_len, self.label_len, self.feat_len))
                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while file_cnt < self.batch_seq_len:
                    idx = self.filenames_list[file_cnt]
                    temp_feat = self.data[idx].reshape(self.nb_frames_file, self.feat_len, self.feat_cls.get_nb_channels()).view(np.float32)
                    temp_label = self.label[idx]
                    feat[file_cnt] = temp_feat
                    label[file_cnt] = temp_label
                    file_cnt = file_cnt + 1

                # Split to sequences
                # feat = np.transpose(feat, (0, 3, 1, 2))
                
                label = label.min(-1, keepdims=True)
                label = [
                    label != 10,  # SED labels
                    label,    # DOA Cartesian labels
                    label.min(-2)   # DOA sample wise label
                    ]
                yield feat, label


    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] // num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_list_index(self, azi, ele):
        return self._feat_cls.get_list_index(azi, ele)

    def get_matrix_index(self, ind):
        return np.array(self._feat_cls.get_vector_index(ind))

    def getnb_classes(self):
        return self.nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

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