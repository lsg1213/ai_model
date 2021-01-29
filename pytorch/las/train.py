from params import getParam 
import os, torch, torchaudio, joblib, pdb
import models
from tensorboardX import SummaryWriter
from data_utils import customDataset
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

def loadingData(path, name):
    x, y = joblib.load(open(os.path.join(path, name+'_x.joblib'), 'rb')), joblib.load(open(os.path.join(path, name+'_y.joblib'), 'rb'))
    max_len_x = 0
    max_len_y = 0
    for i,j in zip(x,y):
        max_len_x = max(i.shape[-1], max_len_x)
        max_len_y = max(len(j), max_len_y)
    return x, y, max_len_x, max_len_y

def iterloop(model, dataloader, criterion, config, device, epoch:int, itertype:str, optimizer=None, scheduler=None):
    train = itertype == 'train'
    with tqdm(dataloader) as pbar:
        for idx, (x, y) in enumerate(pbar):
            if train:
                optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)

            if train:
                loss.backward()
                optimizer.step()
                if idx % (len(dataloader) // 20) == 0:
                    scheduler.step()
        
    writer.add_scalar(f'{itertype}/{itertype}_loss', loss, epoch)
        

def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    abspath = config.abspath
    datapath = os.path.join(abspath, 'datasets/lsj')
    modelsavepath = './model_save'
    tensorboardpath = './tensorboard_log'
    writer = SummaryWriter(tensorboardpath)
    
    # load model
    train_x, train_y, max_len_x_1, max_len_y_1 = loadingData(datapath, 'train')
    test_x, test_y, max_len_x_2, max_len_y_2 = loadingData(datapath, 'test')
    max_len_x = max(max_len_x_1, max_len_x_2)
    max_len_y = max(max_len_y_1, max_len_y_2)

    model = getattr(models, config.model)((train_x[0].shape[0], None), config, max_len_y).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98, verbose=True)
    criterion = torch.nn.NLLLoss(ignore_index=0).to(device)

    # load data
    if config.dataset == 'libri':
        trainset = customDataset(train_x, train_y, config, max_len_x, max_len_y)
        trainset, valset = torch.utils.data.random_split(trainset, [int(len(train_x) * 0.9), len(train_x) - int(len(train_x) * 0.9)])
        trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
        valloader = DataLoader(valset, batch_size=config.batch)
        testset = customDataset(test_x, test_y, config, max_len_x, max_len_y)
        testloader = DataLoader(testset, batch_size=config.batch)
    
    
    for epoch in range(config.epoch):
        # train
        iterloop(model, trainloader, criterion, config, device, epoch, 'train', optimizer, scheduler)

        model.eval()
        with torch.no_grad():
            pass
            # val

            # eval

    

if __name__ == "__main__":
    import sys
    main(getParam(sys.argv[1:]))