
"""
Paper: Visual Attention Network
Link: https://arxiv.org/abs/2202.09741
"""

import torch
import torch.nn as nn

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()