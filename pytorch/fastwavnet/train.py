import torch, pickle, os, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from wavenet import *
from utils import dataSplit, makeDataset
from torchsummary import summary
args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='model')
args.add_argument('--model', type=str, default='st_attention')
args.add_argument('--lr', type=float, default=0.2)
args.add_argument('--opt', type=str, default='adam')
args.add_argument('--gpus', type=str, default='-1')
args.add_argument('--epoch', type=int, default=50)
args.add_argument('--feature', type=str, default='mel')
args.add_argument('--skip', type=int, default=1)
args.add_argument('--decay', type=float, default=0.99)
args.add_argument('--batch', type=int, default=64)





# y = n x k, s = 128 x k x m, y' = n x m
# pad_y = np.concatenate([np.zeros((S.shape[0] - 1, y.shape[1])),
#                     y,
#                     np.zeros((S.shape[0] - 1, y.shape[1]))])


def Shift(y_buffer, k_idx, value):
        y_buffer[1:, k_idx] = y_buffer[:-1, k_idx]
        y_buffer[0, k_idx] = value
        return y_buffer

def Conv_S(signal, device='cpu'):
    #Process S filter to waveform data
    #the shape of signal should be (batch, time, 8)
    batch_size = signal.size(0)
    time_len = signal.size(1)
    y_pred = torch.zeros((batch_size, time_len, 8), device=device).type(torch.float64)
    S_filter = torch.reshape(torch.arange(3*2*2),(3,2,2)).type(torch.float64).to(device) #(Ls, K, M)
    Y_buffer = torch.zeros((3, 2), device=device).type(torch.float64)
    for batch in range(batch_size):
        for n in range(time_len):
            for k in range(2):
                for m in range(2):
                    y_pred[:]= torch.sum(torch.mul(Y_buffer[:, k], S_filter[:, k, m]))
                    Y_buffer = Shift(Y_buffer, k, signal[batch, n, k])
        
    return y_pred


def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tensorboard_path = '/root/ai_model/pytorch/fastwavnet/tensorboard_log'
    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)
    
    SR = 8192
    WINDOW_SIZE = 500 # us
    data_length = SR * WINDOW_SIZE // 1000000
    z_dims = 20
    BATCH_SIZE = config.batch
    learning_rate = 0.0002
    EPOCH = 5
    K, m = 8, 8
    ls = 128

    data_path = '/root/ai_model/pytorch/fastwavnet'
    accel_raw_data = np.array(pickle.load(open(os.path.join(data_path,'accel_data.pickle'),'rb')))
    sound_raw_data = np.array(pickle.load(open(os.path.join(data_path,'sound_data.pickle'),'rb')))
    transfer_f = np.array(pickle.load(open(os.path.join(data_path,'transfer_f.pickle'),'rb')))
    model = FastWaveNet(input_len=data_length, audio_channels=12).to(device)

    transform = transforms.Compose([
        mu_encode(rescale_factor=20)
    ])
    
    accel_data = dataSplit(accel_raw_data, data_length=data_length, device=device)
    sound_data = dataSplit(sound_raw_data, data_length=data_length, device=device)
    dataset = makeDataset(accel_data, sound_data,transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), int(len(dataset) - 0.9 * len(dataset))])
    # accel_loader = torch.utils.data.DataLoader(accel_dataset, batch_size=BATCH_SIZE, num_workers=2, drop_last=True)
    # sound_loader = torch.utils.data.DataLoader(sound_dataset, batch_size=BATCH_SIZE, num_workers=2, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_arr =[]
    model.to(device)
    for epoch in range(EPOCH):
        train_loss, train_acc = 0.0, 0.0
        model.train()
        
        with tqdm(train_loader) as pbar:
            for index, (accel, sound) in enumerate(pbar):
                accel = accel.transpose(1,2)
                accel = accel.to(device)
                sound = sound.to(device)
                optimizer.zero_grad()
                y = model(accel)
                y_p = Conv_S(y)

                loss = criterion(y_p, sound)
                loss.backward()
                optimizer.step()
                _, preds = torch.max(y_p, 1)
                train_loss += loss
                train_acc += torch.sum(preds == sound.data)
                pbar.set_postfix(epoch=f'{epoch}', loss=f'{loss / (index + 1):0.4}',accuracy=f'{train_acc / (index + 1) / BATCH_SIZE}')
            train_loss /= len(train_loader)
            train_acc /= len(train_loader) * BATCH_SIZE
        writer.add_scalar('train/train_loss', train_loss, epoch)
        writer.add_scalar('train/train_acc', train_acc, epoch)

        val_loss, val_acc = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader) as pbar:
                for index, (accel, sound) in enumerate(pbar):
                    accel = accel.to(device)
                    sound = sound.to(device)
                    optimizer.zero_grad()
                    y = model(accel)
                    y_p = Conv_S(y)
                    _, preds = torch.max(y_p, 1)

                    loss = criterion(y_p, sound)
                    val_acc += torch.sum(preds == sound.data)
                    val_loss += loss
                    pbar.set_postfix(epoch=f'{epoch}', loss=f'{loss / (index + 1):0.4}',accuracy=f'{val_acc / (index + 1) / BATCH_SIZE}')
                val_loss /= len(val_loader)
                val_acc /= len(val_loader) * BATCH_SIZE
        writer.add_scalar('val/val_loss', val_loss, epoch)
        writer.add_scalar('val/val_acc', val_acc, epoch)


            

if __name__ == "__main__":
    config = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main(config)
    
    # BATCH_SIZE = 2
   
    # _y_p = Conv_S(_y)
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)

    # transfer_f = torch.reshape(torch.arange(3*2*2),(3,2,2)).type(torch.float64)
    # _y = torch.reshape(torch.arange(BATCH_SIZE*3*2),(BATCH_SIZE,3,2)).type(torch.float64)
    # y = torch.cat((torch.zeros((BATCH_SIZE,2,2)).type(torch.float64),_y),dim=1).type(torch.float64)
    # _transfer_f = transfer_f.permute((2,1,0))
    # _y = y.permute((0,2,1))
    # y_p = F.conv1d(_y, _transfer_f).transpose(1,2)
    # import pdb; pdb.set_trace()
    # print(y_p.shape,y_p)
    # print(_y_p.shape,_y_p)