import torch
from torch.autograd import Function
import score_computation_cuda
import value_aggregation_cuda
from einops.einops import rearrange


class ScoreComputation(Function):
    @staticmethod
    def forward(ctx, query, key, index):
        x = score_computation_cuda.score_forward(query, key, index)
        ctx.save_for_backward(que