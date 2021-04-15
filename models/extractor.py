import torch.nn as nn

from models.conv_block import ConvBlock


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        # group S
        self.conv_1 = ConvBlock(1, 16, p=1)

        # group A
        self.conv_a1 = ConvBlock(16, 32, p=1)
        self.conv_a2 = ConvBlock(32, 48, p=1)
        self.conv_a3 = ConvBlock(48, 64, p=1)

        # group B
        self.conv_b1 = ConvBlock(16, 32, p=2, d=2)
        self.conv_b2 = ConvBlock(32, 48, p=1)
        self.conv_b3 = ConvBlock(48, 64, p=1)

        # group C
        self.conv_c1 = ConvBlock(16, 32, p=3, d=3)
        self.conv_c2 = ConvBlock(32, 48, p=1)
        self.conv_c3 = ConvBlock(48, 64, p=1)

    def forward(self, x):
        # group S
        x = self.conv_1(x)

        # group A
        a1 = self.conv_a1(x)
        a2 = self.conv_a2(a1)
        a3 = self.conv_a3(a2)

        # group B
        b1 = self.conv_b1(x)
        b2 = self.conv_b2(b1)
        b3 = self.conv_b3(b2)

        # group C
        c1 = self.conv_c1(x)
        c2 = self.conv_c2(c1)
        c3 = self.conv_c3(c2)

        # final feathers
        w_tp = [0.1, 0.1, 1]
        f = w_tp[0] * a3 + w_tp[1] * b3 + w_tp[2] * c3

        # transform block
        b_2 = a1 + b1 + c1
        b_1 = a2 + b2 + c2

        # pass
        return f, b_1, b_2
