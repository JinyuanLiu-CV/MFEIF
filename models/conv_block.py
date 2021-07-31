import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 3, s: int = 1, p: int = 0, d: int = 1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, (k, k), (s, s), (p, p), (d, d))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
