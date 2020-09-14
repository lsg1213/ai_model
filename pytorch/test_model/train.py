import torch, pickle, os, argparse, pdb
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import models
from utils import *
from torchsummary import summary
from glob import glob
args = argparse.ArgumentParser()
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--gpus', type=str, default='0')
args.add_argument('--name', type=str, default='')
args.add_argument('--epoch', type=int, default=200)
args.add_argument('--decay', type=float, default=1/np.sqrt(2))
args.add_argument('--batch', type=int, default=512)
args.add_argument('--len', type=int, default=40)
args.add_argument('--b', type=int, default=40)
args.add_argument('--opt', type=str, default='adam')
args.add_argument('--mode', type=str, default='sj_S')
args.add_argument('--model', type=str, default='CombineAutoencoder')
args.add_argument('--resume', action='store_true')
args.add_argument('--ema', action='store_true')
args.add_argument('--weight', action='store_true')
args.add_argument('--relu', action='store_true')
args.add_argument('--future', action='store_true')
args.add_argument('--latency', type=int, default=5, help='latency frame numuber between accel and data')
args.add_argument('--feature', type=str, default='wav', choices=['wav', 'mel'])
args.add_argument('--n_mels', type=int, default=160)


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    SR = 8192
    WINDOW_SIZE = 500 # us
    data_length = config.len
    BATCH_SIZE = config.batch
    learning_rate = config.lr
    EARLY_STOP_STEP = 15
    EPOCH = 2000
    K, m = 8, 8
    ls = 128

    ABSpath = '/home/skuser'
    if not os.path.exists(ABSpath):
        ABSpath = '/root'
    if config.name == '':
        name = f'{config.model}_{config.mode}_b{config.b}_d{data_length}_lat{config.latency}_{config.opt}_{config.lr}_decay{config.decay:0.4}_feature{config.feature}'
        if config.ema:
            name += '_ema'
        if config.weight:
            name += '_weight'
        if config.relu:
            name += '_relu'
        if config.future:
            name += '_future'
    else:
        name = config.name
    if not os.path.exists(os.path.join(ABSpath, 'ai_model')):
        raise FileNotFoundError('path is wrong')
    tensorboard_path = os.path.join(ABSpath, 'ai_model/pytorch/test_model/tensorboard_log/' + name)
    modelsave_path = os.path.join(ABSpath, 'ai_model/pytorch/test_model/model_save/' + name)
    if not os.path.exists(modelsave_path):
        os.makedirs(modelsave_path)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)
    print(name)
    data_path = os.path.join(ABSpath,'data')
    if not os.path.exists(data_path):
        data_path = os.path.join(ABSpath, 'datasets/hyundai')
    accel_raw_data = pickle.load(open(os.path.join(data_path,'stationary_accel_data.pickle'),'rb'))
    sound_raw_data = pickle.load(open(os.path.join(data_path,'stationary_sound_data.pickle'),'rb'))
    transfer_f = np.array(pickle.load(open(os.path.join(data_path,'transfer_f.pickle'),'rb')))
    transfer_f = torch.from_numpy(transfer_f).to(device)
    transfer_f.requires_grad = False

    # accel_data = dataSplit(accel_raw_data, takebeforetime=config.b, data_length=data_length, expand=True)
    # sound_data = dataSplit(sound_raw_data, takebeforetime=config.b, data_length=data_length, expand=False)
    # model = Model(accel_data.shape[1] * accel_data.shape[2], sound_data.shape[1] * sound_data.shape[2]).to(device)
    dataset = makeDataset(accel_raw_data, sound_raw_data, config)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])
    model = getattr(models, config.model)(dataset[0][0].shape[1], dataset[0][1].shape[0], dataset[0][0].shape[0], dataset[0][1].shape[1], config).to(device)
    print(config.model)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False)

    criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    if config.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif config.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f'optimzier must be sgd or adam, current is {config.opt}')
    # lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.decay)
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.decay, patience=1, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    startepoch = 0
    min_loss = 10000000000.0
    earlystep = 0
    if config.resume:
        if len(glob(modelsave_path+'/*')) != 0:
            resume = torch.load(sorted(glob(modelsave_path+'/*.pt'), key=lambda x: float(x.split('/')[-1].split('_')[0]))[-1])
            optimizer.load_state_dict(resume['optimizer'])
            model.load_state_dict(resume['model'])
            startepoch = resume['epoch'] + 1
            min_loss = resume['min_loss']
            lr_schedule.load_state_dict(resume['lr_schedule'])
            earlystep = resume['earlystep']
        else:
            print('resume fail')


        
    
    model.to(device)
    for epoch in range(startepoch, EPOCH):
        train_loss = []
        model.train()
        
        with tqdm(train_loader) as pbar:
            for index, (accel, sound) in enumerate(pbar):
                accel = accel.to(device).type(torch.float64)
                sound = sound.to(device).type(torch.float64)
                optimizer.zero_grad()
                y = model(accel)
                if config.mode == 'ts_S':
                    y_p = Conv_S(y, transfer_f, device)
                elif config.mode == 'sj_S':
                    y_p = conv_with_S(y, transfer_f, config)
                else:
                    y_p = y

                loss = criterion(y_p.type(sound.dtype), sound)
                loss.backward()
                optimizer.step()
                # _, preds = torch.max(y_p, 1)
                train_loss.append(loss.item())
                # train_acc += torch.sum(preds == sound.data)
                pbar.set_postfix(epoch=f'{epoch}', train_loss=f'{np.mean(train_loss):0.4}')
            train_loss = np.mean(train_loss)

        val_loss = []
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader) as pbar:
                for index, (accel, sound) in enumerate(pbar):
                    accel = accel.to(device)
                    sound = sound.to(device)
                    optimizer.zero_grad()
                    y = model(accel)
                    if config.mode == 'ts_S':
                        y_p = Conv_S(y, transfer_f, device)
                    elif config.mode == 'sj_S':
                        y_p = conv_with_S(y, transfer_f, config)
                    else:
                        y_p = y

                    loss = criterion(y_p, sound)
                    # _, preds = torch.max(y_p, 1)
                    val_loss.append(loss.item())
                    pbar.set_postfix(epoch=f'{epoch}', val_loss=f'{np.mean(val_loss):0.4}')
                val_loss = np.mean(val_loss)
        writer.add_scalar('train/train_loss', train_loss, epoch)
        writer.add_scalar('val/val_loss', val_loss, epoch)
        lr_schedule.step(val_loss)
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'earlystep': earlystep,
            'min_loss': min_loss,
            'lr_schedule': lr_schedule.state_dict()
        }, os.path.join(modelsave_path,f'{epoch}_{val_loss:0.4}' + '.pt'))

        if np.isnan(train_loss) or np.isnan(val_loss):
            print('loss is divergence!')
            break
        if min_loss > val_loss:
            earlystep = 0
            min_loss = val_loss
        else:
            earlystep += 1
            if earlystep == 5:
                print('Early stop!')
                break
    print(name)

            

if __name__ == "__main__":
    config = args.parse_args()
    main(config)
    