import torch, pickle, p
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from model import Model
from utils import *
args.add_argument('--gpus', type=str, default='0')
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus

tensorboard_path = './tensorboard_log'
model_save_path = './modelsave'
if not os.path.exists(tensorboard_path):
    os.mkdir(tensorboard_path)
writer = SummaryWriter(tensorboard_path)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
PATH = '/home/skuser/data'
accel = pickle.load(open(PATH + '/stationary_accel_data.pickle', 'rb'))
sound = pickle.load(open(PATH + '/stationary_sound_data.pickle', 'rb'))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

splited_accel = dataSplit(accel,data_length=40)
splited_sound = dataSplit(sound,data_length=40)
assert len(splited_accel) == len(splited_sound), 'wrong datasplit'

BATCH_SIZE = 512
EPOCHS = 200
LR = 0.01
EARLY_STOP_STEP = 10
regularization_weight = 0.1
train_times = 3

model = Model()

optimizer = optim.SGD(model.parameter(),lr=LR,momentum=0.9)