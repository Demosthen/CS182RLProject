import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filter_size, downsample=False):
        self.conv = nn.Conv2d(in_channels, filter_size, stride=int(downsample) + 1)
        self.batchnorm = nn.BatchNorm2d(filter_size)
        self.relu = nn.ReLU()

    def forward(self, in):
        x = self.conv(in)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class FCBlock(nn.Module):
    def __init__(self, in_size, out_size):
        self.fc = nn.Linear(in_size, out_size)
        self.batchnorm = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()

    def forward(self, in):
        x = self.fc(in)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class PPONetwork(nn.Module):
    def __init__(self, input_dims, num_actions, conv_layer_sizes, fc_layer_sizes, downsampling): 
        self.input_dims = input_dims
        self.width, self.height, self.num_channels = input_dims #64 x 64 x 9
        self.num_actions = num_actions
        self.output_dims = num_actions + 1
        self.convs = []
        self.convs.append(ConvBlock(self.num_channels, conv_layer_sizes[0], downsampling[0]))
        for i in range(1, len(conv_layer_sizes)):
            self.convs.append(ConvBlock(conv_layer_sizes[i-1], conv_layer_sizes[i], downsampling[i]))
        power = sum([int(d) for d in downsampling])
        in_size = conv_layer_sizes[-1] * self.width * self.height / (4 ** power)
        self.fcs = [nn.Linear(in_size, fc_layer_sizes[0])]
        for i in range(1, len(fc_layer_sizes)):
            self.fcs.append(FCBlock(fc_layer_sizes[i-1], fc_layer_sizes[i]))
        
    def forward(self, x):
        skip = 0
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            if i % 2 == 0:
                if x.shape != skip.shape:
                    print("You screwed up the skip connections and filter sizes and downsampling whyyyyyyyyyyyyyyyyyyyyyyy")
                x += skip
                skip = x
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
        return x
            
    