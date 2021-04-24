import torch
import torch.nn as nn
from torch import Tensor


class FeatherFuse(nn.Module):
    def __init__(self):
        super(FeatherFuse, self).__init__()

    @staticmethod
    def forward(ir_b: [Tensor], vi_b: [Tensor], mode='min-mean') -> [Tensor]:
        b_1 = torch.min(ir_b[0], vi_b[0])
        b_2 = torch.min(ir_b[1], vi_b[1])
        b_3 = (ir_b[0] + vi_b[0] + b_1) / 3
        b_4 = (ir_b[1] + vi_b[1] + b_2) / 3
        return (b_1, b_2) if mode == 'min' else (b_3, b_4)
