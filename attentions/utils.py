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


def measure_throughput_cpu(model):
    H = W = 14
    B = 64
    x = torch.randn(B, H*W, model.dim)
    print(f"throughput averaged with 30 times")
    tic1 = time.time()
    for i in range(30):
        model(x)
    tic2 = time.time()
    print(f"batch_size {B} throughput on CPU {int(30 * B / (tic2 - tic1))}")



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def get_relative_position_index(q_windows, k_windows):
    """
    Args:
        q_windows: tuple (query_window_height, query_window_width)
        k_windows: tuple (key_window_height, key_window_width)
    Returns:
        relative_position_index: query_window_height*query_window_width, key_window_height*key_window_width
    """
    # get pair-wise relative position index for each token inside the window
    coords_h_q = torch.arange(q_windows[0])
    coords_w_q = torch.arange(q_windows[1])
    coords_q = torch.stack(torch.meshgrid([coords_h_q, coords_w_q]))  # 2, Wh_q, Ww_q

    coords_h_k = torch.arange(k_windows[0])
    coords_w_k = torch.arange(k_windows[1])
    coords_k = torch.stack(torch.meshgrid([coords_h_k, coords_w_k]))  # 2, Wh, Ww

    coords_flatten_q = torch.flatten(coords_q, 1)  