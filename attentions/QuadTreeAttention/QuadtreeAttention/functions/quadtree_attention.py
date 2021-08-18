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
        grad_output = grad_output.contiguous()
        x = score_computation_cuda.score_backward(grad_output, input1, input2, index)
        return x[0], x[1], None


score_computation_op = ScoreComputation.apply


class value_aggregation(Function):
    @staticmethod
    def forward(ctx, score, value, index):
        ctx.save_for_backward(score, value, index)
        f = score.shape[2]
        score = rearrange(score, "b n f K h -> b (n f) K h")  # [b, N, 4, 4K, H] -> [b, 4N, 4K, H]
        index = rearrange(index, "b n f K h -> b (n f) K h")  # [b, N, 4, 4K, H] -> [b, 4N, 4K, H]
        b, N, _, H = score.shape
        D = value.shape[-1]
        # value [b, M, H, D]
        output = score.new_zeros([b, N, H, D]).contiguous()  # b, 4N, H, D
        value_aggregation_cuda.value_aggregation_forward(score, val