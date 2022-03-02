"""
    Paper: HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions
    Link: https://arxiv.org/abs/2207.14284
"""

import torch
import torch.nn as nn

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

class gnconv(nn.Module):
    def 