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

os.environ['CUDA_VISIBLE_DEVICES']='0'
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
PAIRED = int(0.1 * len(trainset))
UNLABELED = int(0.8 * len(trainset))
VAL = int(0.1 * len(trainset))
PRETRAIN_EPOCH = 1000
ACTIVE_EPOCH = 300
BATCH_SIZE = 8
LR = 0.01
SAMPLE_NUM = 50
SAMPLE_LIMIT = 50
EARLY_STOP_STEP = 10
verbose = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
d1, d2, val = torch.utils.data.random_split(trainset, [PAIRED,UNLABELED,VAL])

d1_loader = torch.utils.data.DataLoader(d1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(d1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=2)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
model = DenseNet201().to(device)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR,
                      momentum=0.9, weight_decay=5e-4)
board_writer = SummaryWriter('./tensorboard_log')
min_val_loss = 10000000000.0
''' pretraining with paired data '''
for epoch in range(PRETRAIN_EPOCH):
    running_loss, running_correct = 0.0, 0.0
    for i, data in enumerate(train_loader):
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
    running_loss /= len(train_loader.dataset)
    running_correct = running_correct / len(train_loader.dataset)

    # tensorboard logging
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid)

    board_writer.add_image('cifar 10 images', img_grid)


    val_loss, val_correct = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_loss += criterion(outputs, labels)
            val_correct += torch.sum(preds == labels.data)
    val_loss /= len(test_loader.dataset)
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

    
    torch.save(model.state_dict(), f'./model_save/baseline_all/{epoch}_acc{val_correct:0.4}.pt')
    print(f'epoch: {epoch} loss: {running_loss:0.4}, acc: {running_correct:0.4}, val_loss: {val_loss:0.4}, val_acc: {val_correct:0.4}')

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
