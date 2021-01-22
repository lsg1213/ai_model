from params import getParam 
import os, torch, torchaudio, pdb
import models
from data_utils import preprocess
from tensorboardX import SummaryWriter


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    abspath = config.abspath
    datapath = os.path.join(abspath, 'datasets/lsj')
    modelsavepath = './model_save'
    tensorboardpath = './tensorboard_log'
    writer = SummaryWriter(tensorboardpath)
    
    # load model
    model = getattr(models, config.model)(config, device)
    
    # load data
    if config.dataset == 'libri':
        trainset = customDataset(torchaudio.datasets.LIBRISPEECH(datapath, url='dev-clean', download=True), config, device)
        valset = customDataset(torchaudio.datasets.LIBRISPEECH(datapath, url='test-clean', download=True), config, device)

    # train

    # eval

    

if __name__ == "__main__":
    import sys
    main(getParam(sys.argv[1:]))