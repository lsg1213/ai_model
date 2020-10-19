import torch, pickle, os, pdb, librosa, joblib
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
from pytorch_model_summary import summary
import concurrent.futures as fu
from params import get_arg



def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    SR = config.sr
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
        name = f'{config.model}_{config.mode}'
        name += f'_b{config.b}_d{data_length}' if config.feature == 'wav' else ''
        name += f'_lat{config.latency}_{config.opt}_{config.lr}_decay{config.decay:0.4}'
        name += f'_feature{config.feature}'
        if config.feature == 'mel':
            name += f'_nfft{config.nfft}'
        if config.ema:
            name += '_ema'
        if config.weight:
            name += '_weight'
        if config.relu:
            name += '_relu'
        if config.future:
            name += '_future'
        if config.diff:
            name += f'_diff_weight{config.loss_weight}'
        if config.subtract:
            name += f'_subtract'
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
    transfer_f = np.array(pickle.load(open(os.path.join(data_path,'transfer_f.pickle'),'rb')))
    transfer_f = torch.from_numpy(transfer_f).to(device)
    transfer_f.requires_grad = False
    if config.feature in ['wav', 'mel']:
        accel_raw_data = pickle.load(open(os.path.join(data_path,'stationary_accel_data.pickle'),'rb'))
        sound_raw_data = pickle.load(open(os.path.join(data_path,'stationary_sound_data.pickle'),'rb'))
    elif config.feature == 'mel':
        data_path = os.path.join(data_path, f'{config.feature}_{config.nfft}_{config.nmels}')
        if not os.path.exists(data_path):
            raise ValueError('directory is wrong for to get data')
        accel_raw_data = torch.cat([joblib.load(open(i, 'rb')) for i in sorted(glob(data_path+'/*accel*.joblib'))]).unsqueeze(1) # (frames, 1, nmels, 12)
        sound_raw_data = torch.cat([joblib.load(open(i, 'rb')) for i in sorted(glob(data_path+'/*sound*.joblib'))]) # (frames, windowsize, 8)

        if accel_raw_data.shape[0] != sound_raw_data.shape[0]:
            raise ValueError(f'length of accel and sound data is not matched, {accel_raw_data.shape}, {sound_raw_data.shape}')

    

    # accel_data = dataSplit(accel_raw_data, takebeforetime=config.b, data_length=data_length, expand=True)
    # sound_data = dataSplit(sound_raw_data, takebeforetime=config.b, data_length=data_length, expand=False)
    # model = Model(accel_data.shape[1] * accel_data.shape[2], sound_data.shape[1] * sound_data.shape[2]).to(device)
    dataset = makeDataset(accel_raw_data, sound_raw_data, config, device)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])
    

    # mel: inputs=(n_mels, 12), outputs=(window_size, 8), inch=(3), outch=(frames)
    if config.feature == 'wav':
        model = getattr(models, config.model)(dataset[0][0].shape[1:], dataset[0][1].shape[1:], dataset[0][0].shape[0], dataset[0][1].shape[0], config).to(device)
    elif config.feature == 'mel':
        model = getattr(models, config.model)((config.nmels, 12), (config.len,), (config.len + config.b) // (config.nfft // 2) + 1, 8, config).to(device)
    print(config.model)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False)
    

    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    if config.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif config.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f'optimzier must be sgd or adam, current is {config.opt}')
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.decay)
    # lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.decay, patience=1, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
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


        
    transfer_f = torch.tensor(transfer_f.transpose(0,1).cpu().numpy()[:,::-1,:].copy(),device=device)
    model.to(device)
    for epoch in range(startepoch, EPOCH):
        train_loss = 0.
        model.train()
        if config.feature == 'mel':
            melspectrogram = torchaudio.transforms.MelSpectrogram(8192, n_fft=config.nfft, n_mels=config.nmels).to(device)
        with tqdm(train_loader) as pbar:
            for index, (accel, sound) in enumerate(pbar):
        # for index, (accel, sound) in enumerate(train_loader):
                accel = accel.to(device)
                if config.feature == 'mel':
                    accel = melspectrogram(accel.type(torch.float32)).transpose(1,3)
                accel = accel.type(torch.float64)
                sound = sound.to(device).type(torch.float64)
                optimizer.zero_grad()
                sound = sound.to(device)
                y = model(accel)
                # if config.feature == 'mel':
                #     y = meltowav(y, config)
                if config.mode == 'sj_S':
                    y_p = conv_with_S(y, transfer_f, config)
                else:
                    y_p = y
                loss = criterion(sound - y_p.type(sound.dtype), torch.zeros_like(y_p))
                if config.diff:
                    if y_p.size(1) <= 1:
                        raise ValueError('Cannot use difference value for loss')
                    diff = sound[:,1:,:] - sound[:,:-1,:]
                    diff_loss = criterion((diff) - ((y_p[:,1:,:] - y_p[:,:-1,:]).type(sound.dtype)), torch.zeros_like(diff))
                    total_loss = config.loss_weight * loss + diff_loss
                else:
                    total_loss = loss
                    
                total_loss.backward()
                optimizer.step()
                # _, preds = torch.max(y_p, 1)
                train_loss += total_loss.item()
                # train_acc += torch.sum(preds == sound.data)
                pbar.set_postfix(epoch=f'{epoch}', train_loss=f'{train_loss / (index + 1):0.4}')
                # pbar.set_postfix(epoch=f'{epoch}', train_loss=f'{train_loss / (index + 1):0.4}', value=f'{y_p[0][0][0]}, {sound[0][0][0]}')
            train_loss /= len(train_loader)
        print(f'{epoch}, loss: {train_loss}\nvalue')
        print(f'{y_p[0][y_p.shape[1] // 2]},\n{sound[0][sound.shape[1] // 2]}')

        val_loss = 0.
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader) as pbar:
                for index, (accel, sound) in enumerate(pbar):
                    accel = accel.to(device)
                    sound = sound.type(torch.float64)
                    
                    # if config.feature == 'mel':
                    #     sound = wavtomel(sound, config)
                    sound = sound.to(device)
                    optimizer.zero_grad()
                    y = model(accel)
                    # if config.feature == 'mel':
                    #     y = meltowav(y, config)
                    if config.mode == 'sj_S':
                        y_p = conv_with_S(y, transfer_f, config)
                    else:
                        y_p = y

                    loss = criterion(sound - y_p, torch.zeros_like(y_p))
                    # diff_loss = criterion((y_p[:,1:,:] - y_p[:,:-1,:]).type(sound.dtype), sound[:,1:,:] - sound[:,:-1,:])
                    # _, preds = torch.max(y_p, 1)
                    total_loss = loss.item()
                    val_loss += total_loss
                    pbar.set_postfix(epoch=f'{epoch}', val_loss=f'{val_loss / (index + 1):0.4}')
                val_loss /= len(val_loader)
        writer.add_scalar('train/train_loss', train_loss, epoch)
        writer.add_scalar('val/val_loss', val_loss, epoch)
        # lr_schedule.step(val_loss)
        lr_schedule.step()
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
            if earlystep == 3:
                print('Early stop!')
                break
    print(name)

            

if __name__ == "__main__":
    import sys
    config = get_arg(sys.argv[1:])
    if config.feature == 'mel' and config.nfft > config.len + config.b:
        config.nfft = config.len + config.b
        print(f'nfft is too big to use, change nfft to {config.len + config.b}')
    main(config)
    