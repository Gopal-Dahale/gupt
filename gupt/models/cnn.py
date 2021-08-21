""" CNN Model"""
import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, input_dims, mapping):
        super().__init__()
        conv_dim = 64
        layer_size = 128

        self.conv1 = nn.Conv2d(in_channels=input_dims[0],
                               out_channels=conv_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=conv_dim,
                               out_channels=conv_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.dropout = nn.Dropout(0.25)
        self.max_pool_layer = nn.MaxPool2d(2)
        conv_out_size = 28 // 2
        input_dim = conv_out_size * conv_out_size * conv_dim
        self.linear1 = nn.Linear(input_dim, layer_size)
        self.linear2 = nn.Linear(layer_size, len(mapping))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool_layer(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        return x