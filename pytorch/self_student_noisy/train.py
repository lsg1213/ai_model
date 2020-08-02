import torchvision, torch, os
import torchvision.transforms as transforms
import torchaudio, pickle, pdb
import torch.nn as nn
import torch.optim as optim
from glob import glob
from utils import *
from model import st_attention
from torchsummary import summary
from sklearn.metrics import auc, roc_curve
from tensorboardX import SummaryWriter
import argparse
from tqdm import tqdm
from time import time
args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='model')
args.add_argument('--pad_size', type=int, default=19)
args.add_argument('--step_size', type=int, default=9)
args.add_argument('--model', type=str, default='st_attention')
args.add_argument('--lr', type=float, default=0.2)
args.add_argument('--opt', type=str, default='adam')
args.add_argument('--gpus', type=str, default='0,1,2,3')
args.add_argument('--epoch', type=int, default=50)
args.add_argument('--feature', type=str, default='mel')
args.add_argument('--noise_aug', action='store_true')
args.add_argument('--voice_aug', action='store_true')
args.add_argument('--aug', action='store_true')
args.add_argument('--skip', type=int, default=1)
args.add_argument('--decay', type=float, default=0.95)
args.add_argument('--batch', type=int, default=4096)
args.add_argument('--norm', action='store_true')
args.add_argument('--dataset', type=str, default='noisex')
config = args.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
tensorboard_path = './tensorboard_log'
if not os.path.exists(tensorboard_path):
    os.mkdir(tensorboard_path)
writer = SummaryWriter(tensorboard_path)

PATH = '/root/datasets/ai_challenge/ST_attention_dataset'
x = pickle.load(open(PATH+'/timit_noisex_x_mel.pickle', 'rb'))
y = pickle.load(open(PATH+'/timit_noisex_y_mel.pickle', 'rb'))
val_x = pickle.load(open(PATH+'/libri_aurora_val_x_mel.pickle', 'rb'))
val_y = pickle.load(open(PATH+'/libri_aurora_val_y_mel.pickle', 'rb'))
for i in range(len(x)):
    x[i] = x[i][:, :len(y[i])]
for i in range(len(val_x)):
    val_x[i] = val_x[i][:, :len(val_y[i])]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 512
EPOCHS = 200
LR = 0.01
EARLY_STOP_STEP = 10
regularization_weight = 0.1
train_times = 9
val_times = len(val_x) // 10000
transform = torchvision.transforms.Compose([transforms.ToTensor()])
trainloader = Dataloader_generator(x, y, transform, config=config,device=device, n_data_per_epoch=20000,divide=train_times)
valloader = Dataloader_generator(val_x, val_y, transform, config=config, device=device, n_data_per_epoch=len(val_x), divide=val_times)

model = st_attention(device=device).to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.999)
max_auc = 0.0
for epoch in range(EPOCHS):
    start_time = time()
    running_loss, running_correct, running_auc = 0.0, 0.0, 0.0
    loader_len = 0
    for times in range(train_times):
        train_loader = next(iter(trainloader.next_loader(times)))
        model.train()
        with tqdm(train_loader) as pbar:
            for idx, (data, label) in enumerate(pbar):
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                pipe_score, multi_score, post_score = model(data)
                # _, preds = torch.max(post_loss, 1)
                preds = torch.round(post_score).clone()
                pipe_loss = criterion(pipe_score, label)
                multi_loss = criterion(multi_score, label)
                post_loss = criterion(post_score, label)
                loss = pipe_loss + multi_loss + regularization_weight * post_loss
                loss.backward()
                
                optimizer.step()
                running_loss += loss.item()
                running_correct += torch.sum(preds == label.data)
                fpr, tpr, thresholds = roc_curve(np.reshape(label.cpu().numpy(),(-1)), np.reshape(preds.cpu().detach().numpy(),(-1)))
                running_auc += auc(fpr, tpr)
                pbar.set_postfix(accuracy=f'loss: {running_loss / ((idx+1) + loader_len):0.4}, auc: {running_auc / ((idx+1) + loader_len):0.4}, acc: {running_correct / ((idx+1) + loader_len) / 7 / BATCH_SIZE:0.4}')
        loader_len += len(train_loader)
        train_loader = None
        torch.cuda.empty_cache()
    running_loss /= loader_len
    running_correct /= loader_len * 7 * BATCH_SIZE
    running_auc /= loader_len
    writer.add_scalar('loss/train_loss',running_loss, epoch)
    writer.add_scalar('auc/train_acc',running_correct, epoch)
    writer.add_scalar('acc/train_auc',running_auc, epoch)
    train_loader = None
    torch.cuda.empty_cache()
    model.eval()
    val_loss, val_correct, val_auc = 0.0, 0.0, 0.0
    loader_len = 0
    with torch.no_grad():
        for times in range(val_times):
            val_loader = next(iter(valloader.next_loader(times)))
            with tqdm(val_loader) as pbar:
                for idx, (data, label) in enumerate(pbar):
                    data = data.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()
                    pipe_score, multi_score, post_score = model(data)
                    pipe_loss = criterion(pipe_score, label)
                    multi_loss = criterion(multi_score, label)
                    post_loss = criterion(post_score, label)
                    loss =  pipe_loss + multi_loss + regularization_weight * post_loss
                    # _, preds = torch.max(post_loss, 1)
                    preds = torch.round(post_score).clone()
                    optimizer.step()
                    val_loss += loss.item()
                    val_correct += torch.sum(preds == label.data)
                    fpr, tpr, thresholds = roc_curve(np.reshape(label.cpu().numpy(),(-1)), np.reshape(preds.cpu().detach().numpy(),(-1)))
                    val_auc += auc(fpr, tpr)
                    pbar.set_postfix(accuracy=f'val_loss: {val_loss / ((idx+1) + loader_len):0.4}, val_auc: {val_auc / ((idx+1) + loader_len):0.4}, val_acc: {val_correct / ((idx+1) + loader_len) / 7 / BATCH_SIZE:0.4}')
            loader_len += len(val_loader)
            val_loader = None
            torch.cuda.empty_cache()
        val_loss /= loader_len
        val_correct /= loader_len * 7 * BATCH_SIZE
        val_auc /= loader_len
        writer.add_scalar('loss/val_loss',val_loss, epoch)
        writer.add_scalar('acc/val_acc',val_correct, epoch)
        writer.add_scalar('auc/val_auc',val_auc, epoch)
    val_loader = None

    print(f'epoch: {epoch} loss: {running_loss:0.4}, acc: {running_correct:0.4}, auc: {running_auc:0.4}, val_loss: {val_loss:0.4}, val_acc: {val_correct:0.4}, val_auc: {val_auc:0.4}, time: {time() - start_time:0.4}')
    torch.save(model.state_dict(), f'./model_save/{epoch}_acc{val_auc:0.4}.pt')
    lr_schedule.step()
    torch.cuda.empty_cache()
    if val_auc > max_auc:
        epochs_no_improve = 0
        max_auc = val_auc
    else:
        epochs_no_improve += 1
    if epoch > 5 and epochs_no_improve == EARLY_STOP_STEP:
        print('Early stopping!' )
        break
    else:
        continue
