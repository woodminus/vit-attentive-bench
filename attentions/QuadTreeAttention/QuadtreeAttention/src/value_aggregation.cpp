#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "value_aggregation.h"
extern THCState *state;
#define C