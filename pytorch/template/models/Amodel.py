import torch.nn as nn

class Amodel(nn.Module):
    def __init__(self, config):
        super(Amodel, self).__init__()
        self.input_shape = config.input[-1]
        self.output_shape = config.output[-1]
        self.linear = nn.Linear(self.input_shape, self.output_shape)

    def forward(self, x):
        return self.linear(x)