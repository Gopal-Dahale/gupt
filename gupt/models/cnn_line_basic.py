""" CNNLine Basic Model"""
import math
from torch import nn
from gupt.models.cnn import CNN


class CNNLineBasic(nn.Module):
    """Convolutional Neural Network for line recognition with 3x3 kernel and padding 1.
    Each window is sent through CNN separately.

    Args:
        nn (Module): NN module
    """

    def __init__(self, input_dims, output_dims, mapping):
        super().__init__()
        self.num_mapping = len(mapping)  # Number of output classes i.e. 83
        self.limit = output_dims[0]  # Length of string to output
        self.window_width = 28  # Width of window
        self.window_stride = 28  # Stride by which the window is shifted

        self.cnn = CNN(input_dims, mapping)

    def forward(self, x):
        """Forward Propagation

        Args:
            x (tensor): Input

        Returns:
            out (tensor): Output
        """
        batch_size, num_channels, line_height, line_width = x.shape
        num_windows = math.floor(
            (line_width - self.window_width) / (self.window_stride)) + 1

        for i in num_windows:
            start = i * self.window_stride
            end = start + self.window_width
            window = x[:, :, :, start:end]
