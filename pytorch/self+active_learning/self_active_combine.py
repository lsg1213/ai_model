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


os.environ['CUDA_VISIBLE_DEVICES']='-1'
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
PAIRED = int(0.1 * len(trainset))
UNLABELED = int(0.8 * len(trainset))
VAL = int(0.1 * len(trainset))
PRETRAIN_EPOCH = 5
ACTIVE_EPOCH = 100
BATCH_SIZE = 8
LR = 0.1
SAMPLE_NUM = 100
verbose = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
d1, d2, val = torch.utils.data.random_split(trainset, [PAIRED,UNLABELED,VAL])
d1_loader = torch.utils.data.DataLoader(d1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
d2_loader = torch.utils.data.DataLoader(d1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(d1, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
model = DenseNet201().to(device)
testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR,
                      momentum=0.9, weight_decay=5e-4)

''' pretraining with paired data '''
# for epoch in range(PRETRAIN_EPOCH):
#     running_loss, running_correct = 0.0, 0
#     for i, data in enumerate(tqdm(d1_loader)):
#         model.train()
#         inputs, labels = data
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         running_correct += torch.sum(preds == labels.data)
#     print(f'loss: {running_loss}, acc: {running_correct.type(torch.float32)/len(d1)}')

active = AdvancedActiveLearning()
uncert_sampling = UncertaintySampling(verbose)    
diversity_samp = DiversitySampling(verbose)      
adv_samping = AdvancedActiveLearning(verbose)

sampled_data = uncert_sampling.get_samples(model, d2, uncert_sampling.least_confidence, 
                                                        number=SAMPLE_NUM, device=device)

for epoch in range(ACTIVE_EPOCH):
    pass