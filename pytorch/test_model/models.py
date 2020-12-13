import torch, pickle, torchaudio, pdb
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
from torchsummary import summary



def cal_outputs_conv(inputs, layer):
    if len(layer.kernel_size) == 1:
        if type(inputs) != int:
            inputs = inputs[0]
        return (inputs + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
    elif len(layer.kernel_size) == 2:
        
        return ((inputs[0] + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1,
                (inputs[1] + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1)



class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv1d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm1d(D)
        self.conv_conv = nn.Conv1d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm1d(D)
        self.conv_expand = nn.Conv1d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm1d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class ResNext(nn.Module):
    def __init__(self, inputs, outputs, inch, outch, config):
    # def __init__(self, cardinality, depth, nlabels, base_width, widen_factor=4):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNext, self).__init__()
        self.cardinality = config.cardinality
        self.depth = config.depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = config.base_width
        self.widen_factor = config.widen_factor
        self.nlabels = config.nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv1d(12, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm1d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)

        self.classifier_1 = nn.Linear(self.stages[3] * ((config.len + config.b) // 4 - 7), self.nlabels)
        self.classifier_2 = nn.Linear(self.stages[3] * ((config.len + config.b) // 4 - 7), self.nlabels)
        self.classifier_3 = nn.Linear(self.stages[3] * ((config.len + config.b) // 4 - 7), self.nlabels)
        self.classifier_4 = nn.Linear(self.stages[3] * ((config.len + config.b) // 4 - 7), self.nlabels)
        self.classifier_5 = nn.Linear(self.stages[3] * ((config.len + config.b) // 4 - 7), self.nlabels)
        self.classifier_6 = nn.Linear(self.stages[3] * ((config.len + config.b) // 4 - 7), self.nlabels)
        self.classifier_7 = nn.Linear(self.stages[3] * ((config.len + config.b) // 4 - 7), self.nlabels)
        self.classifier_8 = nn.Linear(self.stages[3] * ((config.len + config.b) // 4 - 7), self.nlabels)
        nn.init.kaiming_normal(self.classifier_1.weight)
        nn.init.kaiming_normal(self.classifier_2.weight)
        nn.init.kaiming_normal(self.classifier_3.weight)
        nn.init.kaiming_normal(self.classifier_4.weight)
        nn.init.kaiming_normal(self.classifier_5.weight)
        nn.init.kaiming_normal(self.classifier_6.weight)
        nn.init.kaiming_normal(self.classifier_7.weight)
        nn.init.kaiming_normal(self.classifier_8.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    nn.init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        # x = (batch, self.tages[3], (config.len + config.b) // 4)
        x = F.avg_pool1d(x, 8, 1)
        x = x.reshape((x.size(0), -1))

        x1 = self.classifier_1(x)
        x2 = self.classifier_2(x)
        x3 = self.classifier_3(x)
        x4 = self.classifier_4(x)
        x5 = self.classifier_5(x)
        x6 = self.classifier_6(x)
        x7 = self.classifier_7(x)
        x8 = self.classifier_8(x)
        
        return torch.softmax(torch.stack([x1,x2,x3,x4,x5,x6,x7,x8], 1), -1)

class CombineAutoencoder(nn.Module):
    def __init__(self, inputs, outputs, inch, outch, config):
        super(CombineAutoencoder, self).__init__()
        self.config = config
        # self.conv1 = nn.Conv1d(inch, 128, 3, stride=2, padding=1)
        # self.batchnorm1 = nn.BatchNorm1d(128)
        # self.dropout1 = nn.Dropout(p=0.2)
        # self.conv2 = nn.Conv1d(128, 8, 1)
        # self.batchnorm2 = nn.BatchNorm1d(8)
        # self.dropout2 = nn.Dropout(p=0.2)
        if config.feature == 'mel':
            self.conv1 = nn.Conv2d(inch, 64, 3, padding=1)
            self.batchnorm1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.batchnorm2 = nn.BatchNorm2d(128)
        elif config.feature == 'wav':
            self.conv1 = nn.Conv1d(inch, 64, 3, padding=1)
            self.batchnorm1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
            self.batchnorm2 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        _inputs = cal_outputs_conv(cal_outputs_conv((inputs), self.conv1), self.conv2)

        if config.feature == 'wav':
            self.back = FCAutoencoder(_inputs, config.len, self.conv2.out_channels, outputs[0], config)
        elif config.feature == 'mel':
            # mel: inputs=(frames, 12), outputs=(out_frames), inch=(self.conv2 filter number), outch=(8)
            self.back = FCAutoencoder(_inputs, config.len, self.conv2.out_channels, 8, config)

    
    def forward(self, x):
        x = self.conv1(x.type(torch.float32))
        x = F.relu(self.dropout1(self.batchnorm1(x)))
        # x = F.relu(self.dropout1(x))
        x = self.conv2(x)
        x = F.relu(self.dropout2(self.batchnorm2(x)))
        # x = F.relu(self.dropout2(x))

        # (batch, self.conv2.output_channels, n_mels, 12)
        x = self.back(x)
        if self.config.norm:
            x = torch.tanh(x)
        return x

class CNN(nn.Module):
    def __init__(self, inputs, outputs, inch, outch, config):
        super(CNN, self).__init__()
        self.config = config
        self.inch = inch
        # self.conv1 = nn.Conv1d(inch, 128, 3, padding=1).double()
        # self.bn1 = nn.BatchNorm1d(16).double()
        self.do1 = nn.Dropout(0.1)
        self.do2 = nn.Dropout(0.1)
        self.do3 = nn.Dropout(0.1)
        self.do4 = nn.Dropout(0.1)
        # self.conv2 = nn.Conv1d(128, 8, 3, padding=1).double()
        # self.bn2 = nn.BatchNorm1d(8).double()
        # self.li1 = nn.Linear(inputs,100,bias=True).double()
        # self.li2 = nn.Linear(100,outputs,bias=False).double()
        self.conv1 = nn.Conv1d(12, 32, kernel_size=129, padding=64).double()
        self.conv2 = nn.Conv1d(12, 32, kernel_size=65, padding=32).double()
        self.conv3 = nn.Conv1d(12, 32, kernel_size=257, padding=128).double()
        self.conv4 = nn.Conv1d(12, 32, kernel_size=33, padding=16).double()
        self.conv5 = nn.Conv1d(128, 8, kernel_size=129, padding=64).double()

    def forward(self, x):
        x1 = x2 = x3 = x4 = x
        # x1 = self.do1(x)
        # x2 = self.do2(x)
        # x3 = self.do3(x)
        # x4 = self.do4(x)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4) 
        x = torch.cat([x1,x2,x3,x4], dim=1)
        x = torch.tanh(self.conv5(x))
        return x.transpose(1,2)

class FCAutoencoder(nn.Module):
    def __init__(self, inputs, outputs, inch, outch, config):
        super(FCAutoencoder, self).__init__()
        self.config = config
        if type(inputs) != int:
            if len(inputs) > 1:
                _in = 1
                for i in inputs:
                    _in *= i
                inputs = _in
        # mel: inputs=(frames, 12), outputs=(out_frames), inch=(self.conv2 filter number), outch=(8)
        # mel: (batch, 128, n_mels, 12)
        self.linear1 = nn.Linear(inputs * inch, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 30)
        self.linear5 = nn.Linear(30, 64)
        self.linear6 = nn.Linear(64, 128)
        self.linear7 = nn.Linear(128, 256)
        if config.feature == 'wav':
            self.linear8 = nn.Linear(256, outputs * outch)
        elif config.feature == 'mel':
            self.linear8 = nn.Linear(256, outputs * outch)
            # self.linear8 = nn.Linear(256, self.config.nmels * 8 * (self.config.len // (self.config.nfft // 2) + 1))

    def forward(self, x):
        x = torch.reshape(x, (x.size(0), -1))
        if self.config.relu:
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            x = F.relu(self.linear3(x))
            x = F.relu(self.linear4(x))
            x = F.relu(self.linear5(x))
            x = F.relu(self.linear6(x))
            x = F.relu(self.linear7(x))
        else:
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            x = self.linear4(x)
            x = self.linear5(x)
            x = self.linear6(x)
            x = self.linear7(x)
        # out = self.linear8(x).transpose(1,2)
        out = self.linear8(x)
        if self.config.feature == 'wav':
            out = torch.reshape(out, (out.size(0), -1, 8))
        elif self.config.feature == 'mel':
            # make mel output
            # out = torch.reshape(out, (out.size(0), self.config.nmels, 8, -1))
            out = torch.reshape(out, (out.size(0), self.config.len, 8))
        return out.type(torch.double)

if __name__ == "__main__":
    import sys
    from params import get_arg
    config = get_arg(sys.argv[1:])
    config.b = 0
    config.len = 2048
    device = torch.device('cuda:0')
    model = CNN(config.len + config.b, config.len, 12, 8, config).to(device).float()
    summary(model, (12, config.b+config.len))