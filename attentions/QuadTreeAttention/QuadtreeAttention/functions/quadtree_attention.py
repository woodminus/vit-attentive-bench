import torch
from torch.autograd import Function
import score_computation_cuda
import value_aggregation_cuda
from einops.einops import rearrange


class ScoreComputation(Function):
    @staticmethod
    def forward(ctx, query, key, index):
        x = score_computation_cuda.score_forward(query, key, index)
        ctx.save_for_backward(query, key, index)
        return x[0]

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, index = ctx.saved_tensors
        grad_output = grad_output.contiguous