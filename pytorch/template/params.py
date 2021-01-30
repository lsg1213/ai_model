from config_manager import get_config
import argparse
from numpy import sqrt

def preprocess_config(config):
    config = vars(config)
    preprocess = {
        'auto_augment': {'threshold': 0.1}
    }
    config['data_preprocess'] = preprocess

    preprocess = {
        'onehot': {}
    }
    config['label_preprocess'] = preprocess
    return argparse.Namespace(**config)
    
def get_param(known=[]):
    args = argparse.ArgumentParser()
    
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--gpus', type=str, default='-1')
    args.add_argument('--name', type=str, default='')
    args.add_argument('--epoch', type=int, default=200)
    args.add_argument('--decay', type=float, default=1/sqrt(2))
    args.add_argument('--batch', type=int, default=128)
    args.add_argument('--resume', action='store_true')    
    args.add_argument('--abspath', type=str, default='/root')
    args.add_argument('--config_mode', type=str, default='')
    args.add_argument('--model', type=str, default='Amodel')
    args.add_argument('--input', type=tuple, default=(1,))
    args.add_argument('--output', type=tuple, default=(1,))


    config = args.parse_known_args(known)[0]
    
    config = preprocess_config(config)
    config = get_config(config.name, config, mode=config.config_mode)
    return config

if __name__ == '__main__':
    import sys
    config = get_param(sys.argv[1:])
    print(config)