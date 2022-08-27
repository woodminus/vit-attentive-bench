"""
    Paper: Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
    Link: https://arxiv.org/abs/2102.12122
"""

import torch
import torch.nn as nn
from utils import conv_flops

class SRAttention(nn.Module):
    """
    Spatial Reduction Attention

    Paper: Pyramid Vision Transforme