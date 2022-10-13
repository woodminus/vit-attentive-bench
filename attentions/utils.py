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
    H = W = 14
    B = 64
    x = torch.randn(B, H*W, model.dim).cuda()

    model = model.cuda()
    print(f"throughput averaged with 30 times")
    torch.cuda.synchronize()
    tic1 = time.time()
    for i in range(30):
        model(x)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {B} throughput on GPU {int(30 * B / (tic2 - tic1))}")


def measure_thro