""" CNN Model"""
import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Convolutional Neural Network with 3x3 kernel and padding 1.

    Args:
        nn (Module): NN module
    """

    def __init__(self, input_dims, mapping):
        super().__init__()
        conv_dim = 64
        layer_size = 128
        output_size = len(mapping)
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
        self.conv3 = nn.Conv2d(in_channels=conv_dim,
                               out_channels=conv_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(conv_dim)
        self.bn2 = nn.BatchNorm2d(conv_dim)
        self.bn3 = nn.BatchNorm2d(conv_dim)
        self.dropout = nn.Dropout(0.25)
        self.max_pool_layer = nn.MaxPool2d(2)
        conv_out_size = 28 // 2
        input_dim = conv_out_size * conv_out_size * conv_dim
        self.linear1 = nn.Linear(input_dim, layer_size)
        self.linear2 = nn.Linear(layer_size, output_size)

    def forward(self, x):
        """Forward Propagation

        Args:
            x (tensor): Input

        Returns:
            out (tensor): Output
        """
        residual = x  # used for identity

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.max_pool_layer(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
