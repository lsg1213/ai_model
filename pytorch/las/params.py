import argparse
import numpy as np 
from config_manager import getConfig

def getParam(known=[]):
    args = argparse.ArgumentParser()
    
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--gpus', type=str, default='3')
    args.add_argument('--name', type=str, default='')
    args.add_argument('--epoch', type=int, default=200)
    args.add_argument('--decay', type=float, default=1/np.sqrt(2))
    args.add_argument('--batch', type=int, default=128)
    args.add_argument('--resume', action='store_true')    
    args.add_argument('--dataset', type=str, default='timit', choices=['timit','libri'])
    args.add_argument('--abspath', type=str, default='/root')
    args.add_argument('--model', type=str, default='LAS', choices=['LAS'])
    args.add_argument('--sr', type=int, default=16000)

    # listener argument
    args.add_argument('--hidden', type=int, default=256)

    config = args.parse_known_args(known)[0]
    
    config = getConfig(config.name, config, False)
    return config