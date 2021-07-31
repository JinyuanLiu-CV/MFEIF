import torch
import torch.nn as nn
from torch import Tensor

from models.conv_block import ConvBlock


class Constructor(nn.Module):
    def __init__(self):
        super(Constructor, self).__init__()

        self.conv_1 = ConvBlock(64, 16, p=1)
        self.conv_2 = ConvBlock(64, 32, p=1)
        self.conv_3 = ConvBlock(64, 16, p=1)
        self.conv_4 = ConvBlock(16, 1, p=1)

    def forward(self, x, b_1, b_2) -> Tensor:
        x = self.conv_1(x)
        x = torch.cat([x, b_1], dim=1)
        x = self.conv_2(x)
        x = torch.cat([x, b_2], dim=1)
        x = self.conv_3(x)
        x = self.conv_4(x)

        return x
