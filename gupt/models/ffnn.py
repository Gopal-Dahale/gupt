""" Feed Forward Neural Network Model"""
import torch
from torch import nn
import torch.nn.functional as F


class FeedForwardNN(nn.Module):
    """Feed Forward Neural Network class for recognizing digits from 0 to 9

    Args:
        nn (Module): torch nn module
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.linear.append(
                nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))

    def forward(self, x):
        """Forward Propagation

        Args:
            x (tensor): Input

        Returns:
            out (tensor): Output
        """
        out = torch.flatten(x, 1)

        # Apply layers & activation functions
        for i in range(len(self.linear) - 1):
            out = self.linear[i](out)
            out = F.relu(out)
            out = self.dropout(out)

        # Output layer
        out = self.linear[-1](out)
        return out
