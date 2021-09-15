""" CNNLine Basic Model"""
import math
import torch
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

        # type_as(x) is required for setting the device (cuda or cpu)
        # https://forums.pytorchlightning.ai/t/training-fails-but-found-at-least-two-devices-cuda-0-and-cpu/694
        preds = torch.zeros(
            (batch_size, self.num_mapping, num_windows)).type_as(x)

        for i in range(num_windows):
            start = i * self.window_stride  # Start of a window
            end = start + self.window_width  # End of a window
            window = x[:, :, :, start:end]
            preds[:, :, i] = self.cnn(window)  # Predictions

        return preds
