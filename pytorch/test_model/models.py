import torch, pickle, torchaudio, pdb
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
from torchsummary import summary
from concurrent.futures import ThreadPoolExecutor

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
        if config.feature == ['mel','stft']:
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

        if config.weight:
            with torch.no_grad():
                self.conv1.weight = torch.nn.Parameter(torch.zeros_like(self.conv1.weight) + 1e-5)
                self.conv2.weight = torch.nn.Parameter(torch.zeros_like(self.conv2.weight) + 1e-5)

    
    def forward(self, x):
        x = self.conv1(x.type(torch.float32))
        x = F.relu(self.dropout1(self.batchnorm1(x)))
        # x = F.relu(self.dropout1(x))
        x = self.conv2(x)
        x = F.relu(self.dropout2(self.batchnorm2(x)))
        # x = F.relu(self.dropout2(x))

        # (batch, self.conv2.output_channels, n_mels, 12)
        x = self.back(x)
        return x

class CNN(nn.Module):
    def __init__(self, inputs, outputs, inch, outch, config):
        super(CNN, self).__init__()
        self.config = config
        self.conv1 = nn.Conv1d(inch, 16, 3, padding=1).double()
        self.bn1 = nn.BatchNorm1d(16).double()
        self.do1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(16, 8, 3, padding=1).double()
        self.bn2 = nn.BatchNorm1d(8).double()
        self.do2 = nn.Dropout(0.1)
        self.li1 = nn.Linear(inputs,100,bias=True).double()
        self.li2 = nn.Linear(100,outputs,bias=False).double()

    def forward(self, x):
        x = self.conv1(x)
        x = self.do1(self.bn1(x))
        x = self.conv2(x)
        x = self.do2(self.bn2(x))

        x = torch.relu(self.li1(x))
        # x = torch.tanh(self.li2(x))
        x = self.li2(x)

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

        if config.weight:
            layers = dict(self._modules)
            for layer in layers.keys():
                la = getattr(self, layer).weight
                la = torch.nn.Parameter(torch.zeros_like(la) + 1e-5)

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



import torch
import torch.nn as nn
import math
from efficient_module import *
from efficient_util import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    Swish,
    MemoryEfficientSwish,
)



class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * \
            self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))

        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(
                self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate,
                                 training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 24  # rgb
        # number of output channels
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for i in range(len(self._blocks_args)):
            # Update block input and output filters based on depth multiplier.
            self._blocks_args[i] = self._blocks_args[i]._replace(
                input_filters=round_filters(
                    self._blocks_args[i].input_filters, self._global_params),
                output_filters=round_filters(
                    self._blocks_args[i].output_filters, self._global_params),
                num_repeat=round_repeats(
                    self._blocks_args[i].num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(
                self._blocks_args[i], self._global_params))
            if self._blocks_args[i].num_repeat > 1:
                self._blocks_args[i] = self._blocks_args[i]._replace(
                    input_filters=self._blocks_args[i].output_filters, stride=1)
            for _ in range(self._blocks_args[i].num_repeat - 1):
                self._blocks.append(MBConvBlock(
                    self._blocks_args[i], self._global_params))

        # Head'efficientdet-d0': 'efficientnet-b0',
        # output of final block
        in_channels = self._blocks_args[len(
            self._blocks_args)-1].output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        P = []
        index = 0
        num_repeat = 0
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            num_repeat = num_repeat + 1
            if(num_repeat == self._blocks_args[index].num_repeat):
                num_repeat = 0
                index = index + 1
                P.append(x)
        return P

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        # Convolution layers
        P = self.extract_features(inputs)
        return P

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(
            model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={
                              'num_classes': num_classes})

        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(
                image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = cls.from_name(model_name, override_params={
                              'num_classes': num_classes})
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet-b'+str(i) for i in range(num_models)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' +
                             ', '.join(valid_models))

    def get_list_features(self):
        list_feature = []
        for idx in range(len(self._blocks_args)):
            list_feature.append(self._blocks_args[idx].output_filters)

        return list_feature

class efficientnet(nn.Module):
    def __init__(self, inputs, outputs, inch, outch, config):
        super(efficientnet, self).__init__()
        self.frontmodel = EfficientNet.from_pretrained('efficientnet-b0')
        self.config = config
        self.output = int(np.ceil((config.len) / (config.nfft // 2 + 1)) + 1)

        self.conv0 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv1 = nn.Conv2d(24, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(40, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(80, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(112, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 16, 3, padding=1)
        self.conv6 = nn.Conv2d(320, 16, 3, padding=1)
                    
        self.fc = nn.Linear(7 * self.output, self.output)


    def forward(self, x):
        x = self.frontmodel(x)
        res = torch.ones((x[0].shape[0],16, self.config.nfft // 2 + 1, self.output * len(x)), dtype=x[0].dtype, device=x[0].device)
        for i, feat in enumerate(x):
            _x = getattr(self, f'conv{i}')(feat)
            _x = F.interpolate(_x, size=(self.config.nfft // 2 + 1, self.output)) # (batch, channel, freq, time)
            res[:,:,:,i*self.output : (i+1)*self.output] = _x
        x = self.fc(res)
        return x


if __name__ == '__main__':
    from params import get_arg
    config = get_arg([])
    model = EfficientNet.from_pretrained('efficientnet-b0')
    config.len = 1000
    config.b = 1000
    config.nfft = 2048
    inputs = torch.randn(4, 12, config.nfft // 2 + 1, config.len + config.b)
    P = efficientnet((config.nfft // 2 + 1, config.len + config.b), (config.nfft // 2 + 1, config.len), 12, 8, config)(inputs)


    for idx, p in enumerate(P):
        print('P{}: {}'.format(idx, p.size()))
    # print('model: ', model)