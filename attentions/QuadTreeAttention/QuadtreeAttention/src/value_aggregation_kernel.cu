#include <vector>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <vector>
#include "value_aggregation.h"
#include "THC/THCAtomics.cuh"
#include <stdio.h>
#include "utils.h"

#define ROUND_OFF 50000

#define CUDA_NUM_THREADS 1024
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define GET_BLOCKS(n, t) (n+t-1) / t

__global__ void ValueAggregationForwardFunc(float* score, float* value, long* index, float* output, int B, int N, int K, int H, int M, int D) {
  ///*
  long LENGTH = B*N*H*D;
  CUDA_KERNEL_LOOP(cur_idx, LENGTH){
      long d_idx = cur_idx % D; 
      long h_idx = (cur_idx - d_idx) / D % H; 
      long n_idx = (cur_idx - d_idx - h_idx * D) / D / H % N;
      long b_idx = (cur_idx - d_idx - h_idx * D - n_idx * H * D) / D / H / N;
      if (cur_idx < LENGTH) {
        long score_start_idx = b_idx * N * K * H + n_idx * K * H + h_idx;
        long value_start_idx = b_idx * M * H * D + h_idx * D + d_idx;
        
        float out_val = 0;
        for(int k_idx = 0; k_idx < K; k_idx++){
            int score_idx = score_start_idx + k_idx * H;
            int value_idx = value_start_idx + index[score_idx] * H * D;
            out_val += score[score_idx] * value[value_idx];
        }
        output[cur_idx] = out_val;
      }
  }
}


void value_aggregation_forward_kernel(float* score, float* value, long* index, float* ouput, int B, int N, int K, int H, int M, int D