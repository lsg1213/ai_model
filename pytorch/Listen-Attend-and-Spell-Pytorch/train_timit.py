import yaml
from util.timit_dataset import load_dataset,create_dataloader
from model.las_model import Listener,Speller
from util.functions import batch_iterator
import numpy as np
from torch.autograd import Variable
import torch
import sys
from tensorboardX import SummaryWriter
import argparse

# Load config file for experiment
parser = argparse.ArgumentParser(description='Training script for LAS on TIMIT .')

parser.add_argument('config_path', metavar='config_path', type=str,
                     help='Path to config file for training.')

paras = parser.parse_args()

config_path = paras.config_path

conf = yaml.load(open(config_path,'r'))


# Parameters loading
torch.manual_seed(conf['training_parameter']['seed'])
total_steps = conf['training_parameter']['total_steps']
use_pretrained = conf['training_parameter']['use_pretrained']
verbose_step = conf['training_parameter']['verbose_step']
valid_step  = conf['training_parameter']['valid_step']
tf_rate_upperbound = conf['training_parameter']['tf_rate_upperbound']
tf_rate_lowerbound = conf['training_parameter']['tf_rate_lowerbound']

# Load preprocessed TIMIT Dataset ( using testing set directly here, replace them with validation set your self)
# X : Padding to shape [num of sample, max_timestep, feature_dim]
# Y : Squeeze repeated label and apply one-hot encoding (preserve 0 for <sos> and 1 for <eos>)
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(**conf['meta_variable'])
train_set = create_dataloader(X_train, y_train, **conf['model_parameter'], **conf['training_parameter'], shuffle=True)
valid_set = create_dataloader(X_val, y_val, **conf['model_parameter'], **conf['training_parameter'], shuffle=False)
#test_set = create_dataloader(X_test, y_test, **conf['model_parameter'], **conf['training_parameter'], shuffle=False)

# Construct LAS Model or load pretrained LAS model
log_writer = SummaryWriter(conf['meta_variable']['training_log_dir']+conf['meta_variable']['experiment_name'])

if not use_pretrained:
    listener = Listener(**conf['model_parameter'])
    speller = Speller(**conf['model_parameter'])
else:
    listener = torch.load(conf['training_parameter']['pretrained_listener_path'])
    speller = torch.load(conf['training_parameter']['pretrained_speller_path'])
optimizer = torch.optim.Adam([{'params':listener.parameters()}, {'params':speller.parameters()}],
                             lr=conf['training_parameter']['learning_rate'])
listener_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.listener'
speller_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.speller'

# save checkpoint with the best ler
best_ler = 1.0
global_step = 0

while global_step < total_steps:

    # Teacher forcing rate linearly decay
    tf_rate = tf_rate_upperbound - (tf_rate_upperbound-tf_rate_lowerbound)*(global_step/total_steps)
    
    # Training
    for batch_index,(batch_data,batch_label) in enumerate(train_set):
        batch_loss, batch_ler = batch_iterator(batch_data, batch_label, listener, speller, optimizer, 
                                               tf_rate, is_training=True, **conf['model_parameter'])
        global_step += 1

        if global_step % verbose_step == 0:
            log_writer.add_scalars('loss',{'train':batch_loss}, global_step)
            log_writer.add_scalars('cer',{'train':np.array([np.array(batch_ler).mean()])}, global_step)
        if global_step % valid_step == 0:
            break
    
    # Validation
    dev_loss = []
    dev_ler = []
    for _,(batch_data,batch_label) in enumerate(valid_set):
        batch_loss, batch_ler = batch_iterator(batch_data, batch_label, listener, speller, optimizer, 
                                               tf_rate, is_training=False, **conf['model_parameter'])
        dev_loss.append(batch_loss)
        dev_ler.extend(batch_ler)

    log_writer.add_scalars('loss',{'dev':np.array([sum(dev_loss)/len(dev_loss)])}, global_step)
    log_writer.add_scalars('cer',{'dev':np.array([np.array(dev_ler).mean()])}, global_step)

    # Generate Attention map
    if conf['model_parameter']['bucketing']:
        feature = listener(Variable(batch_data.float()).squeeze(0).cuda())
        batch_label = batch_label.squeeze(0)
    else:
        feature = listener(Variable(batch_data.float()).cuda())
    pred_seq, attention_score = speller(feature)
    
    pred_seq = [char.cpu() for char in pred_seq]
    for t in range(len(attention_score)):
        for h in range(len(attention_score[t])):
            attention_score[t][h] = attention_score[t][h].cpu()
    del feature
    
        
    att_map = {i:[] for i in range(conf['training_parameter']['batch_size'])}
    num_head = len(attention_score[0])
    for i in range(conf['training_parameter']['batch_size']):
        for j in range(num_head):
            att_map[i].append([])
    for t,head_score in enumerate(attention_score):
        for h,att_score in enumerate(head_score):
            for idx,att in enumerate(att_score.data.numpy()):
                att_map[idx][h].append(att)
    for i in range(conf['training_parameter']['batch_size']-1):
        for j in range(num_head):
            m = np.repeat(np.expand_dims(np.array(att_map[i][j]),0),3,axis=0)
            log_writer.add_image('attention_'+str(i)+'_head_'+str(j),
                                 torch.FloatTensor(m[:,:,:]), global_step)



    # Checkpoint
    if best_ler >= sum(dev_ler)/len(dev_ler):
        best_ler = sum(dev_ler)/len(dev_ler)
        torch.save(listener, listener_model_path)
        torch.save(speller, speller_model_path)
