"""
    Paper: CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped
    Link: https://arxiv.org/abs/2107.00652
"""

import torch
import torch.nn as nn
import numpy as np

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
  