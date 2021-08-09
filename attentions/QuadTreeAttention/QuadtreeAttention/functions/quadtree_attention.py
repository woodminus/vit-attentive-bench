import torch
from torch.autograd import Function
import score_computation_cuda
import value_aggregation_cuda
from einops.einops import rearrange


class Sc