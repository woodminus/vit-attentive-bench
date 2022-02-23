
"""
    Paper: Focal Self-attention for Local-Global Interactions in Vision Transformers
    Link: https://arxiv.org/abs/2107.00641
"""

import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_
import math
from utils import window_reverse, window_partition, window_partition_noreshape, get_relative_position_index


class FocalWindowAttention(nn.Module):
    r""" Focal Attention

    Args:
        dim (int): Number of input channels.
        expand_size (int): The expand size at focal level 1.
        window_size (tuple[int]): The height and width of the window.
        focal_window (int): Focal region size.
        focal_level (int): Focal attention level.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pool_method (str): window pooling method. Default: none
    """

    def __init__(self, dim, expand_size=3, window_size=7, focal_window=3, focal_level=2, num_heads=12,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pool_method="none", shift_size=0):

        super().__init__()


        self.dim = dim
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.pool_method = pool_method
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.focal_level = focal_level
        self.focal_window = focal_window

        # from block-wise setting
        self.shift_size = shift_size
        self.window_size_glo = self.window_size
        self.pool_layers = nn.ModuleList()
        if self.pool_method != "none":
            for k in range(self.focal_level - 1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                if self.pool_method == "fc":