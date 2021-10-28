#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "value_aggregation.h"
extern THCState *state;
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(