import argparse
import numpy as np
def get_arg(known=[]):
    args = argparse.ArgumentParser()
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--gpus', type=str, default='0')
    args.add_argument('--name', type=str, default='')
    args.add_argument('--epoch', type=int, default=200)
    args.add_argument('--decay', type=float, default=1/np.sqrt(2))
    args.add_argument('--batch', type=int, default=1024)
    args.add_argument('--len', type=int, default=200)
    args.add_argument('--b', type=int, default=200)
    args.add_argument('--opt', type=str, default='adam')
    args.add_argument('--mode', type=str, default='sj_S')
    args.add_argument('--model', type=str, default='CombineAutoencoder')
    args.add_argument('--resume', action='store_true')
    args.add_argument('--ema', action='store_true')
    args.add_argument('--weight', action='store_true')
    args.add_argument('--relu', action='store_true')
    args.add_argument('--eval', action='store_true')
    args.add_argument('--future', action='store_true')
    args.add_argument('--diff', action='store_true')
    args.add_argument('--subtract', action='store_true')
    args.add_argument('--sr', type=int, default=8192)
    args.add_argument('--latency', type=int, default=5, help='latency frame numuber between accel and data')
    args.add_argument('--feature', type=str, default='wav', choices=['wav', 'mel', 'mfcc'])
    args.add_argument('--nmels', type=int, default=80)
    args.add_argument('--nfft', type=int, default=512)
    args.add_argument('--loss_weight', type=float, default=0.5)
    return args.parse_known_args(known)[0]


if __name__ == "__main__":
    import sys, pdb
    pdb.set_trace()