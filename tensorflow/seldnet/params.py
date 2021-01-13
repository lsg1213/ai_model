import argparse
import numpy as np
def getparam(known=[]):
    args = argparse.ArgumentParser()
    args.add_argument('--name', type=str, default='test')
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--gpus', type=str, default='-1')
    args.add_argument('--epoch', type=int, default=2000)
    args.add_argument('--resume', action='store_true')
    args.add_argument('--skip', type=int, default=128)
    args.add_argument('--decay', type=float, default=0.9)
    args.add_argument('--db', type=int, default=30)
    args.add_argument('--batch', type=int, default=16)
    args.add_argument('--seq_len', type=int, default=64)
    args.add_argument('--nfft', type=int, default=512)
    args.add_argument('--window_size', type=int, default=19)
    args.add_argument('--s', type=int, default=8, help='seconds of data')
    args.add_argument('--patience', type=int, default=10)
    args.add_argument('--dataset', type=str, default='seld', choices=['seld', 'clean_iris'])
    args.add_argument('--filter', action='store_true')
    
    args.add_argument('--dropout_rate', type=float, default=0.0)
    args.add_argument('--nb_cnn2d_filt', type=int, default=64)
    args.add_argument('--pool_size', type=str, default='8,8')
    args.add_argument('--rnn_size', type=str, default='128,128')
    args.add_argument('--fnn_size', type=str, default='512')
    args.add_argument('--loss_weights', type=str, default='1,50,50')
    args.add_argument('--mode', type=str, default='frame', choices=['frame', 'sample'])
    args.add_argument('--thdoa', type=float, default=0.3, help='Threshhold of algorithm for transforming doa frame to sample label')
    args.add_argument('--thsed', type=float, default=0.6, help='Threshhold of algorithm for transforming sed frame to sample label')

    return args.parse_known_args(known)[0]