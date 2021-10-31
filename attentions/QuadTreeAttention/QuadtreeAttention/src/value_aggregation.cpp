#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "value_aggregation.h"
extern THCState *state;
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void value_aggregation_cuda_forward(
                    at::Tensor score, // B, N, K, H
                    at::Tensor value, // B, M, H, D
                    at::Tensor index, // B, N, K, H
                    at::Tensor output)// B, N, H, D
{
    CHECK_INPUT(score);
    CHECK_INPUT(value);
    CHECK_INPUT(index);
    auto score_size = score.sizes();
    auto value_size = value.sizes();
    int B = score_size[0];
    int N = score_size[1];
    int K = score_size[2];
    int H = score_size[3];
    int M = value