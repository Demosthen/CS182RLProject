from itertools import product
import torch
from torch import tensor
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filter_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, filter_size, stride=stride, kernel_size=(3, 3), padding=1)
        self.batchnorm = nn.BatchNorm2d(filter_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv(input)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class FCBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        print("hi", out_size, in_size)
        self.fc = nn.Linear(in_size, out_size)
        self.batchnorm = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.fc(input)
        #x = self.batchnorm(x)
        x = self.relu(x)
        return x

class PPONetwork(nn.Module):
    def __init__(self, input_dims, num_actions, conv_layer_sizes, fc_layer_sizes, strides): 
        super().__init__()
        self.input_dims = input_dims
        self.width, self.height, self.num_channels = input_dims #64 x 64 x 9
        self.num_actions = num_actions
        self.output_dims = num_actions + 1
        self.convs = []
        self.convs.append(ConvBlock(self.num_channels, conv_layer_sizes[0], strides[0]))
        for i in range(1, len(conv_layer_sizes)):
            self.convs.append(ConvBlock(conv_layer_sizes[i-1], conv_layer_sizes[i], strides[i]))
        for i in range(len(self.convs)):
            self._modules["conv" + str(i)] = self.convs[i]
        
        power = np.prod([int(d) for d in strides])
        in_size = int(conv_layer_sizes[-1] * self.width * self.height / (power**2))
        self.fcs = [nn.Linear(in_size, fc_layer_sizes[0])]
        for i in range(1, len(fc_layer_sizes)):
            self.fcs.append(FCBlock(fc_layer_sizes[i-1], fc_layer_sizes[i]))
        self.fcs.append(FCBlock(fc_layer_sizes[-1], self.output_dims))
        for i in range(len(self.fcs)):
            self._modules["fc" + str(i)] = self.fcs[i]
        
    def forward(self, x : Tensor):
        skip = torch.zeros([1])
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            if i % 2 == 0:
                if x.shape != skip.shape:
                    pass
                    # print("You screwed up the skip connections and filter sizes and downsampling whyyyyyyyyyyyyyyyyyyyyyyy")
                #x += skip
                skip = x
        # print(x.shape)
        x = x.flatten(start_dim = 1)
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
        return x
            
    