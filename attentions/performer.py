"""
    Paper: Rethinking Attention with Performers
    Link: https://arxiv.org/abs/2009.14794

    mainly modified from https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
"""

import math
from scipy.stats import ortho_group
import torch
from torch import nn
from eino