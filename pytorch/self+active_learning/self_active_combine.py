import torch
import torchvision
import torchvision.transforms as transforms
import os, pdb
import numpy as np
import torch.optim as optim
import torch.nn as nn
from models import DenseNet201
from tqdm import tqdm
from advanced_active_learning import *
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from utils import matplotlib_imshow, plot_classes_preds
import time
from self_training import self_training
os.environ['CUDA_VISIBLE_DEVICES']='0'
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
PAIRED = int(0.1 * len(trainset))
UNLABELED = int(0.8 * len(trainset))
VAL = int(0.1 * len(trainset))
PRETRAIN_EPOCH = 5
ACTIVE_EPOCH = 500
BATCH_SIZE = 16
LR = 0.001
SAMPLE_NUM = 100
SAMPLE_LIMIT = 10000
EARLY_STOP_STEP = 15
verbose = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
d1, d2, val = torch.utils.data.random_split(trainset, [PAIRED,UNLABELED,VAL])
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=2)
d1_loader = torch.utils.data.DataLoader(d1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
d2_loader = torch.utils.data.DataLoader(d2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

model = DenseNet201().to(device)
import pdb; pdb.set_trace()
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR,
                      momentum=0.9, weight_decay=5e-4)
min_val_loss = 10000000000.0
''' pretraining with paired data '''
for epoch in range(PRETRAIN_EPOCH):
    start_time = time.time()
    running_loss, running_correct = 0.0, 0.0
    for i, data in enumerate(d1_loader):
        model.train()
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_correct += torch.sum(preds == labels.data)
    running_loss /= len(d1)
    running_correct /= len(d1)

    val_loss, val_correct = 0.0, 0.0
    model.eval()
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        val_loss += criterion(outputs, labels)
        val_correct += torch.sum(preds == labels.data)

    val_loss /= len(testset)
    val_correct = val_correct / len(testset)

    print(f'epoch: {epoch} loss: {running_loss:0.4}, acc: {running_correct:0.4}, val_loss: {val_loss:0.4}, val_acc: {val_correct:0.4}, time: {time.time() - start_time:0.4}')
print(f'---------pretraining is over {PRETRAIN_EPOCH:03} epochs-------------')

active = AdvancedActiveLearning()
uncert_sampling = UncertaintySampling(verbose)    
diversity_samp = DiversitySampling(verbose)      
adv_samping = AdvancedActiveLearning(verbose)
board_writer = SummaryWriter(f'./tensorboard_log/combine_{SAMPLE_NUM}')
teacher_model = DenseNet201()
teacher_model.load_state_dict(torch.load('./model_save/baseline_all/34_acc86.41.pt'))
teacher_model.eval()
teacher_model.to(device)
self_training = self_training(teacher_model)

print('Active + Self learning start')
for epoch in range(ACTIVE_EPOCH):
    start_time = time.time()
    if len(d2) != 0:
        # active learning 3 type
        d2, sampled_labeled_data = uncert_sampling.get_samples(model, d2, uncert_sampling.least_confidence, number=SAMPLE_NUM, device=device)
        d1 = torch.utils.data.ConcatDataset([d1, sampled_labeled_data])
        
        d2, sampled_labeled_data = uncert_sampling.get_samples(model, d2, uncert_sampling.margin_confidence, number=SAMPLE_NUM, device=device)
        d1 = torch.utils.data.ConcatDataset([d1, sampled_labeled_data])

        d2, sampled_labeled_data = uncert_sampling.get_samples(model, d2, uncert_sampling.entropy_based, number=SAMPLE_NUM, device=device)
        d1 = torch.utils.data.ConcatDataset([d1, sampled_labeled_data])
        uncert_sampling.logprobs = []

        # self-training
        d2, sampled_labeled_data = self_training.get_samples(d2, number=SAMPLE_NUM * 3,device=device)
        d1 = torch.utils.data.ConcatDataset([d1, sampled_labeled_data])
        self_training.prob = []

        d1_loader = torch.utils.data.DataLoader(d1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        d2_loader = torch.utils.data.DataLoader(d2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    else:
        d1_loader = torch.utils.data.DataLoader(d1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # tensorboard logging
    dataiter = iter(d1_loader)
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid)

    board_writer.add_image('cifar 10 images', img_grid)
    running_loss, running_correct = 0.0, 0.0
    model.train()
    for i, data in enumerate(d1_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_correct += torch.sum(preds == labels.data)
    running_loss /= len(d1)
    running_correct = running_correct / len(d1_loader.dataset)


    val_loss, val_correct = 0.0, 0.0
    model.eval()
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        val_loss += criterion(outputs, labels)
        val_correct += torch.sum(preds == labels.data)

    val_loss /= len(testset)
    val_correct = val_correct / len(test_loader.dataset)

    
    # ...학습 중 손실(running loss)을 기록하고
    board_writer.add_scalar('training loss',
                    running_loss,
                    epoch)
    board_writer.add_scalar('val loss',
                    val_loss,
                    epoch)
    board_writer.add_scalar('training acc',
                    running_correct,
                    epoch)
    board_writer.add_scalar('val acc',
                    val_correct,
                    epoch)

    # ...무작위 미니배치(mini-batch)에 대한 모델의 예측 결과를 보여주도록
    # Matplotlib Figure를 기록합니다
    model.eval()
    for i, data in enumerate(val_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        board_writer.add_figure('predictions vs. actuals',
                    plot_classes_preds(model, inputs, labels, classes),
                    global_step=epoch)
        break

    
    torch.save(model.state_dict(), f'./model_save/active_confidence/{epoch}_acc{val_correct:0.4}.pt')
    print(f'epoch: {epoch} loss: {running_loss:0.4}, acc: {running_correct:0.4}, val_loss: {val_loss:0.4}, val_acc: {val_correct:0.4}, time: {time.time() - start_time:0.4} seconds, used dataset length {len(d1)}')


    if val_loss < min_val_loss:
        epochs_no_improve = 0
        min_val_loss = val_loss
    else:
        epochs_no_improve += 1

    if epoch > 5 and epochs_no_improve == EARLY_STOP_STEP:
        print('Early stopping!' )
        break
    else:
        continue
print(f'd1 dataset size: {len(d1)}')
