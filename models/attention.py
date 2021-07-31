import torch.nn as nn

from functions.edge_detect import EdgeDetect


class ConvConv(nn.Module):
    def __init__(self, a_channels, b_channels, c_channels):
        super(ConvConv, self).__init__()

        self.conv_1 = nn.Conv2d(a_channels, b_channels, (3, 3), padding=(1, 1))
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(b_channels, c_channels, (3, 3), padding=(2, 2), dilation=(2, 2))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        return x


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.conv_1 = ConvConv(1, 32, 32)
        self.conv_2 = ConvConv(32, 64, 128)
        self.conv_3 = ConvConv(128, 64, 32)
        self.conv_4 = nn.Conv2d(32, 1, (1, 1))

        self.ed = EdgeDetect()

    def forward(self, x):
        # edge detect
        e = self.ed(x)
        x = x + e

        # attention
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)

        return x
