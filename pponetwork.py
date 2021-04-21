from itertools import product
import torch
from torch import tensor
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
    
    def forward(self, input):
        out = self.relu(input)
        out = self.conv0(out)
        out = self.relu(out)
        out = self.conv1(out)
        return out + input

class ConvSequence(nn.Module):
    def __init__(self, in_channels, depth):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, depth, 3, padding=1)
        self.max0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res0 = ResidBlock(depth)
        self.res1 = ResidBlock(depth)
    
    def forward(self, input):
        out = self.conv0(input)
        out = self.max0(out)
        out = self.res0(out)
        out = self.res1(out)
        return out
        

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filter_size, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, filter_size, stride=stride, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2)
        self.batchnorm = nn.BatchNorm2d(filter_size)
        with torch.no_grad():
            self.conv.weight *= torch.sqrt(torch.tensor(2))
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv(input)
        #x = self.batchnorm(x)
        x = self.relu(x)
        return x

class FCBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        print("hi", out_size, in_size)
        self.fc = nn.Linear(in_size, out_size)
        with torch.no_grad():
            self.fc.weight *= torch.sqrt(torch.tensor(2))
        self.batchnorm = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.fc(input)
        #x = self.batchnorm(x)
        x = self.relu(x)
        return x

class PPONetwork(nn.Module):
    def __init__(self, input_dims, num_actions, conv_layer_sizes, fc_layer_sizes, strides, filter_sizes): 
        super().__init__()
        self.input_dims = input_dims
        self.width, self.height, self.num_channels = input_dims #64 x 64 x 9
        self.num_actions = num_actions
        self.output_dims = num_actions + 1
        self.convs = []
        self.filter_sizes = filter_sizes
        self.convs.append(ConvBlock(self.num_channels, filter_sizes[0], conv_layer_sizes[0], strides[0]))
        for i in range(1, len(conv_layer_sizes)):
            self.convs.append(ConvBlock(conv_layer_sizes[i-1], filter_sizes[i], conv_layer_sizes[i], strides[i]))
        for i in range(len(self.convs)):
            self._modules["conv" + str(i)] = self.convs[i]
        
        power = np.prod([int(d) for d in strides])
        #in_size = int(conv_layer_sizes[-1] * self.width * self.height / (power**2))
        in_size = 324
        self.fcs = [nn.Linear(in_size, fc_layer_sizes[0])]
        for i in range(1, len(fc_layer_sizes)):
            self.fcs.append(FCBlock(fc_layer_sizes[i-1], fc_layer_sizes[i]))
        self.fcs.append(nn.Linear(fc_layer_sizes[-1], self.output_dims))
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

def compute_size(width, filter_size, stride, padding):
    return (width - filter_size + 2 * padding) / stride + 1

class IMPALA_CNN(nn.Module):
    def __init__(self, in_channels, depths, out_dim):
        super().__init__()
        self.convs = [ConvSequence(in_channels, depths[0])]
        
        for i in range(1, len(depths)):
            self.convs.append(ConvSequence(depths[i-1], depths[i]))
        for i in range(len(self.convs)):
            self._modules["conv" + str(i)] = self.convs[i]
        size = 64 // (2 ** len(depths))
        size = size ** 2
        size = size * depths[-1]
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(size, 256)
        self.fc1 = nn.Linear(256, out_dim)
        self.val_fc0 = nn.Linear(size, 256)
        self.val_fc1 = nn.Linear(256, 1)
        self.out_dim = out_dim
    
    def forward(self, input):
        out = input
        for i in range(len(self.convs)):
            out = self.convs[i](out)
        
        out = torch.flatten(out, start_dim=1)
        out = self.relu(out)
        val_out = self.val_fc0(out)
        val_out = self.relu(val_out)
        val_out = self.val_fc1(val_out)
        out = self.fc0(out)
        out = self.relu(out)
        out = self.fc1(out)
        output = torch.cat([out, val_out], dim=-1)
        return output

        
    

