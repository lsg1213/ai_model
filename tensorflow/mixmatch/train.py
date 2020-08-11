import argparse, os
import tensorflow as tf
from tensorboardX import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=None, help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used for training (e.g. cifar10, cifar100, svhn, svhn+extra)')

    parser.add_argument('--epochs', type=int, default=1024, help='number of epochs, (default: 1024)')
    parser.add_argument('--batch-size',  type=int, default=64, help='examples per batch (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='learning_rate, (default: 0.01)')

    parser.add_argument('--labelled-examples', type=int, default=500, help='number labelled examples (default: 4000')
    parser.add_argument('--validation-examples', type=int, default=5000, help='number validation examples (default: 5000')
    parser.add_argument('--val-iteration', type=int, default=1024, help='number of iterations before validation (default: 1024)')
    parser.add_argument('--T', type=float, default=0.5, help='temperature sharpening ratio (default: 0.5)')
    parser.add_argument('--K', type=int, default=3, help='number of rounds of augmentation (default: 2)')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='param for sampling from Beta distribution (default: 0.75)')
    parser.add_argument('--lambda-u', type=int, default=100, help='multiplier for unlabelled loss (default: 100)')
    parser.add_argument('--rampup-length', type=int, default=16,
                        help='rampup length for unlabelled loss multiplier (default: 16)')
    parser.add_argument('--weight-decay', type=float, default=0.02, help='decay rate for model vars (default: 0.02)')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='ema decay for ema model vars (default: 0.999)')

    parser.add_argument('--config-path', type=str, default=None, help='path to yaml config file, overwrites args')
    parser.add_argument('--tensorboard', action='store_true', help='enable tensorboard visualization')
    parser.add_argument('--resume', action='store_true', help='whether to restore from previous training runs')
    parser.add_argument('--gpus', type=str, default='0', help='set gpu numbers')

    return parser.parse_args()

def main(config):
    os.environ['']

if __name__ == "__main__":
    main(get_args())