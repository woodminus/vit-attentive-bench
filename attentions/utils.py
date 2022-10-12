import torch
import time
from fvcore.nn import FlopCountAnalysis



def measure_flops_params(model, x):
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops = FlopCountAnalysis(model,  x)
    converted = flops.total() / 1e6
    print(f'Number of Params: {round(n_parameters/ 1e6, 2)} M')
    print(f'FLOPs = {round(converted, 2)} M')


def measure_throughput_gpu(model):
    H = W = 1