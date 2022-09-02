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

    Paper: Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
    Link: https://arxiv.org/abs/2102.12122
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
 