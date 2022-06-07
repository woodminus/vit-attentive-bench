"""
    Paper: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    Link: https://arxiv.org/abs/2103.14030
"""

import torch
import torch.nn as nn
from utils import window_reverse, window_partition
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_

class ShiftedWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with re