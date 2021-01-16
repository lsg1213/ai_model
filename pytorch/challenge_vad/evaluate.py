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
import concurrent.futures, multiprocessing

args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--pad_size', type=int, default=19)
args.add_argument('--step_size', type=int, default=9)
args.add_argument('--model', type=str, default='st_attention')
args.add_argument('--gpus', type=str, default='0,1,2,3')
args.add_argument('--feature', type=str, default='mel')
args.add_argument('--batch', type=int, default=512)
args.add_argument('--norm', type=str, default='paper', choices=['paper', 'timit'])
args.add_argument('--dataset', type=str, default='tedrium', choices=['tedrium', 'libri'], help='tedrium, libri is available')
config = args.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

name = config.name + f'_eval_{config.dataset}'
tensorboard_path = './tensorboard_log/'+name
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
writer = SummaryWriter(tensorboard_path)

model_path = './model_save' + f'/{config.name}'

# model_path = sorted(glob(model_path + '/*.pt'), key=lambda x: float(x.split('/')[-1].split('valloss')[-1].split('.pt')[0]))[0]
model_path = sorted(glob(model_path + '/*.pt'), key=lambda x: int(os.path.basename(x).split('_')[0]), reverse=True)[:1]
# print(model_path.split('/')[-1])
PATH = '/root/datasets/ai_challenge'
if config.dataset == 'tedrium':
    datapath = PATH + '/TEDLIUM-3/TEDLIUM_release-3/data/mel'
    preprocessing_name = '/tedrium_nfft1024_win25_hop10_nmel80/'
    wavpath = datapath + preprocessing_name + '/mel'
    labelpath = datapath + preprocessing_name + '/label'
    def loading(path):
        return joblib.load(open(path, 'rb'))
    print('get x')
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() // 2) as pool:
        eval_x = list(pool.map(loading, sorted(glob(wavpath + '/*.joblib'))))
    print('get y')
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() // 2) as pool:
        eval_y = list(pool.map(loading, sorted(glob(labelpath + '/*.joblib'))))
elif config.dataset == 'libri':
	datapath = PATH + '/ST_attention_Libri_aurora/mel'
	preprocessing_name = '/libriaurora_nfft1024_win25_hop10_nmel80'
	wavpath = datapath + preprocessing_name + '/data'
	labelpath = datapath + preprocessing_name + '/label'
	def loading(path):
		return joblib.load(open(path, 'rb'))

	with concurrent.futures.ThreadPoolExecutor() as pool:
		eval_x = list(pool.map(loading, sorted(glob(wavpath + '/*.joblib')[:100])))
	with concurrent.futures.ThreadPoolExecutor() as pool:
		eval_y = list(pool.map(loading, sorted(glob(labelpath + '/*.joblib')[:100])))
    # eval_x = pickle.load(open(PATH+'/ST_attention_dataset/libri_aurora_val_x_mel.pickle', 'rb'))
    # eval_y = pickle.load(open(PATH+'/ST_attention_dataset/libri_aurora_val_y_mel.pickle', 'rb'))
for i in range(len(eval_x)):
	eval_x[i] = eval_x[i][:, :len(eval_y[i])]
	if config.dataset == 'libri':
		eval_x[i] = eval_x[i].transpose(1,0)

print('data load success')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
regularization_weight = 0.1
eval_times = 30
win = WindowUtils(config.pad_size, config.step_size, device)
BATCH_SIZE = config.batch
transform = None
evalloader = Dataloader_generator(eval_x, eval_y, transform, config=config, device=device, n_data_per_epoch=len(eval_x), divide=eval_times, batch_size=BATCH_SIZE)
eval_x = None
eval_y = None
model = st_attention(device=device)
# model.load_state_dict(torch.load(model_path)['model'])
# model.to(device)

# res = sorted(glob(model_save_path + '/*'), key=lambda x: float(x.split('auc')[-1].split('.pt')[0]), reverse=True)[0]
# print(res)

model.eval()

criterion = nn.BCELoss()
for index, path in enumerate(model_path):
	eval_loss, eval_correct, eval_auc = 0.0, 0.0, 0.0
	start_time = time()
	model.load_state_dict(torch.load(path)['model'])
	model.to(device)
	with torch.no_grad():
	    loader_len = 0
	    auc_count = 0
	    for times in range(eval_times):
	        eval_loader = next(iter(evalloader.next_loader(times)))
	        with tqdm(eval_loader) as pbar:
	            for idx, (data, label) in enumerate(pbar):
	                data = data.to(device)
	                label = label.type(data.dtype).to(device)
	                pipe_score, multi_score, post_score = model(data)
	                pipe_loss = criterion(pipe_score, label)
	                multi_loss = criterion(multi_score, label)
	                post_loss = criterion(post_score, label)
	                loss = pipe_loss + multi_loss + regularization_weight * post_loss
	                # _, preds = torch.max(post_loss, 1)
	                preds = post_score.clone()
	                eval_loss += loss.item()
	                eval_correct += torch.sum(torch.round(preds) == label.data).cpu()
	                # label_seq = win.windows_to_sequence(label.cpu(),config.pad_size,config.step_size)
	                # preds_seq = win.windows_to_sequence(preds.cpu(),config.pad_size,config.step_size)
	                label_seq = label.cpu().reshape(-1)
					pdb.set_trace()
	                if len(label_seq.unique()) == 2:
	                    eval_auc += getAUC(preds.cpu().reshape(-1), label_seq)
	                else:
	                    auc_count += 1
	                pbar.set_postfix(accuracy=f'eval_loss: {eval_loss / ((idx+1) + loader_len):0.4}, eval_auc: {eval_auc / (((idx+1) + loader_len) - auc_count):0.4}, eval_acc: {eval_correct / ((idx+1) + loader_len) / 7 / BATCH_SIZE:0.4}')
	            loader_len += len(pbar)
	        eval_loader = None
	        torch.cuda.empty_cache()
	    eval_loss /= loader_len
	    eval_correct /= loader_len * 7 * BATCH_SIZE
	    eval_auc /= loader_len - auc_count
	    writer.add_scalar('loss/eval_loss',eval_loss, len(model_path) - index - 1)
	    writer.add_scalar('acc/eval_acc',eval_correct, len(model_path) - index - 1)
	    writer.add_scalar('auc/eval_auc',eval_auc, len(model_path) - index - 1)
	print(f'eval_loss: {eval_loss:0.4}, eval_acc: {eval_correct:0.4}, eval_auc: {eval_auc:0.4}, time: {time() - start_time:0.4}')
