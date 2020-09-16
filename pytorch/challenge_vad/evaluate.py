import torchvision, torch, os
import torchvision.transforms as transforms
import torchaudio, pickle, pdb, joblib
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--pad_size', type=int, default=19)
args.add_argument('--step_size', type=int, default=9)
args.add_argument('--model', type=str, default='st_attention')
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--opt', type=str, default='adam')
args.add_argument('--gpus', type=str, default='0,1,2,3')
args.add_argument('--epoch', type=int, default=50)
args.add_argument('--feature', type=str, default='mel')
args.add_argument('--noise_aug', action='store_true')
args.add_argument('--voice_aug', action='store_true')
args.add_argument('--aug', action='store_true')
args.add_argument('--resume', action='store_true')
args.add_argument('--skip', type=int, default=1)
args.add_argument('--decay', type=float, default=1/np.sqrt(2))
args.add_argument('--batch', type=int, default=512)
args.add_argument('--norm', action='store_true')
args.add_argument('--dataset', type=str, default='tedrium', choices=['tredrium', 'libri'], help='tedrium, libri is available')
config = args.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

name = config.name

PATH = '/root/datasets/ai_challenge'
if config.dataset == 'tedrium':
    datapath = PATH + '/TEDLIUM-3/TEDLIUM_release-3/data'
    wavpath = datapath + '/mel/tedrium_nfft1024_win25_hop10_nmel80'
    labelpath = label + '/label'
    eval_x = [joblib.load(open(i, 'rb')) for i in sorted(glob(wavpath + '/*.joblib'))]
    eval_x = torch.cat([torch.from_numpy(i).permute(2,1,0).squeeze(-1) for i in eval_x])
    eval_y = pickle.load(open(PATH+'/libri_aurora_val_y_mel.pickle', 'rb'))
elif config.dataset == 'libri':
    eval_x = pickle.load(open(PATH+'/ST_attention_dataset/libri_aurora_val_x_mel.pickle', 'rb'))
    eval_y = pickle.load(open(PATH+'/ST_attention_dataset/libri_aurora_val_y_mel.pickle', 'rb'))
    for i in range(len(eval_x)):
        eval_x[i] = eval_x[i][:, :len(eval_y[i])]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
regularization_weight = 0.1
eval_times = 1
win = WindowUtils(config.pad_size, config.step_size, device)
BATCH_SIZE = config.batch
transform = torchvision.transforms.Compose([transforms.ToTensor()])
evalloader = Dataloader_generator(eval_x, eval_y, transform, config=config, device=device, n_data_per_epoch=len(eval_x), divide=eval_times, batch_size=BATCH_SIZE)
model = st_attention(device=device)
model.to(device)
model_save_path = './model_save/'+name
# res = sorted(glob(model_save_path + '/*'), key=lambda x: float(x.split('auc')[-1].split('.pt')[0]), reverse=True)[0]
# print(res)
res = sorted(glob(model_save_path + '/*'), key=lambda x: int(x.split('/')[-1].split('_')[0]), reverse=True)[0]
print(res)
model.load_state_dict(torch.load(res)['model'])
model.eval()
eval_loss, eval_correct, eval_auc = 0.0, 0.0, 0.0
criterion = nn.BCELoss()
start_time = time()
with torch.no_grad():
    loader_len = 0
    for times in range(eval_times):
        eval_loader = next(iter(evalloader.next_loader(times)))
        with tqdm(eval_loader) as pbar:
            for idx, (data, label) in enumerate(pbar):
                data = data.to(device)
                label = label.to(device)
                pipe_score, multi_score, post_score = model(data)
                pipe_loss = criterion(pipe_score, label)
                multi_loss = criterion(multi_score, label)
                post_loss = criterion(post_score, label)
                loss = pipe_loss + multi_loss + regularization_weight * post_loss
                # _, preds = torch.max(post_loss, 1)
                preds = torch.round(post_score).clone()
                eval_loss += loss.item()
                eval_correct += torch.sum(preds == label.data).cpu()

                label_seq = win.windows_to_sequence(label.cpu(),config.pad_size,config.step_size)
                preds_seq = win.windows_to_sequence(preds.cpu(),config.pad_size,config.step_size)
                fpr, tpr, thresholds = roc_curve(label_seq.type(torch.int).numpy(), preds_seq.numpy(), pos_label=1)
                _eval_auc = auc(fpr, tpr)
                eval_auc += _eval_auc
                pbar.set_postfix(accuracy=f'eval_loss: {eval_loss / ((idx+1) + loader_len):0.4}, eval_auc: {eval_auc / ((idx+1) + loader_len):0.4}, eval_acc: {eval_correct / ((idx+1) + loader_len) / 7 / BATCH_SIZE:0.4}')
            loader_len += len(pbar)
        eval_loader = None
        torch.cuda.empty_cache()
    eval_loss /= loader_len
    eval_correct /= loader_len * 7 * BATCH_SIZE
    eval_auc /= loader_len
print(f'eval_loss: {eval_loss:0.4}, eval_acc: {eval_correct:0.4}, eval_auc: {eval_auc:0.4}, time: {time() - start_time:0.4}')