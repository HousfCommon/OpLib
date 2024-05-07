#--just practice--
import torch.nn as nn
import torch.nn.functional as F
import torch


class Conv2DTesting(nn.Module):
    def __init__(self, input_channels, mid_channels, output_channels):
        super(Conv2DTesting, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=5, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(mid_channels, output_channels, kernel_size=5)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, input_data):
        x = self.conv1(input_data)
        print("conv1", x.shape)
        y = self.conv2(x)
        print("conv2", y.shape)
        return y