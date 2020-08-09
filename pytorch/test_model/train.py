import torch, pickle, os, argparse, pdb
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from model import Model
from utils import dataSplit, makeDataset, Conv_S, conv_with_S
from torchsummary import summary
args = argparse.ArgumentParser()
args.add_argument('--lr', type=float, default=0.01)
args.add_argument('--gpus', type=str, default='0')
args.add_argument('--epoch', type=int, default=200)
args.add_argument('--decay', type=float, default=0.99)
args.add_argument('--batch', type=int, default=64)
args.add_argument('--len', type=int, default=40)
args.add_argument('--b', type=int, default=40)
args.add_argument('--opt', type=str, default='adam')
args.add_argument('--mode', type=str, default='no_S')


def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    SR = 8192
    WINDOW_SIZE = 500 # us
    data_length = config.len
    BATCH_SIZE = config.batch
    learning_rate = config.lr
    EARLY_STOP_STEP = 15
    EPOCH = 200
    K, m = 8, 8
    ls = 128

    ABSpath = '/home/skuser'
    tensorboard_path = os.path.join(ABSpath, 'ai_model/pytorch/test_model/tensorboard_log')
    name = config.mode + f'_{config.b}_{data_length}_{config.opt}_{config.lr}'
    modelsave_path = os.path.join(ABSpath, 'ai_model/pytorch/test_model/modelsave/' + name)
    if not os.path.exists(modelsave_path):
        os.mkdir(modelsave_path)
    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)

    data_path = os.path.join(ABSpath,'data')
    accel_raw_data = pickle.load(open(os.path.join(data_path,'stationary_accel_data.pickle'),'rb'))
    sound_raw_data = pickle.load(open(os.path.join(data_path,'stationary_sound_data.pickle'),'rb'))
    transfer_f = np.array(pickle.load(open(os.path.join(data_path,'transfer_f.pickle'),'rb')))

    accel_data = dataSplit(accel_raw_data, takebeforetime=config.b, data_length=data_length)
    sound_data = dataSplit(sound_raw_data, takebeforetime=config.b, data_length=data_length)
    model = Model(accel_data.shape[1] * accel_data.shape[2], sound_data.shape[1] * sound_data.shape[2]).to(device)
    dataset = makeDataset(accel_data, sound_data)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False)

    criterion = nn.MSELoss()
    if config.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif config.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f'optimzier must be sgd or adam, current is {config.opt}')

    min_loss = 10000000000.0
    earlystep = 0
    model.to(device)
    for epoch in range(EPOCH):
        train_loss = []
        model.train()
        
        with tqdm(train_loader) as pbar:
            for index, (accel, sound) in enumerate(pbar):
                accel = accel.transpose(1,2)
                accel = accel.to(device)
                sound = sound.to(device)
                optimizer.zero_grad()
                y = model(accel)
                if config.mode == 'ts_S':
                    y_p = Conv_S(y, transfer_f, device)
                elif config.mode == 'sj_S':
                    y_p = conv_with_S(y, transfer_f, device)
                else:
                    y_p = y

                loss = criterion(y_p, sound)
                loss.backward()
                optimizer.step()
                # _, preds = torch.max(y_p, 1)
                train_loss.append(loss.item())
                # train_acc += torch.sum(preds == sound.data)
                pbar.set_postfix(epoch=f'{epoch}', train_loss=f'{np.mean(train_loss):0.4}')
            train_loss = np.mean(train_loss)
        writer.add_scalar('train/train_loss', train_loss, epoch)

        val_loss = []
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader) as pbar:
                for index, (accel, sound) in enumerate(pbar):
                    accel = accel.transpose(1,2)
                    accel = accel.to(device)
                    sound = sound.to(device)
                    optimizer.zero_grad()
                    y = model(accel)
                    if config.mode == 'ts_S':
                        y_p = Conv_S(y, transfer_f, device)
                    elif config.mode == 'sj_S':
                        y_p = conv_with_S(y, transfer_f, device)
                    else:
                        y_p = y

                    loss = criterion(y_p, sound)
                    # _, preds = torch.max(y_p, 1)
                    val_loss.append(loss.item())
                    pbar.set_postfix(epoch=f'{epoch}', val_loss=f'{np.mean(val_loss):0.4}')
                val_loss = np.mean(val_loss)
        writer.add_scalar('val/val_loss', val_loss, epoch)

        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }, os.path.join(modelsave_path,f'{epoch}_{val_loss:0.4}'))

        if np.isnan(train_loss) or np.isnan(val_loss):
            print('loss is divergence!')
            break
        if min_loss < val_loss:
            earlystep = 0
            min_loss = 0
        else:
            earlystep += 1
            if earlystep == 15:
                print('Early stop!')
                break

            

if __name__ == "__main__":
    config = args.parse_args()
    main(config)
    