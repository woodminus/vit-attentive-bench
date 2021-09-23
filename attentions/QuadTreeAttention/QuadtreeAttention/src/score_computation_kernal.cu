#include <vector>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <vector>
#include "score_computation.h"
#include <stdio.h>

#define ROUND_OFF 50000

#define CUDA_NUM_THREADS 1024
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32
#define MAX_H 8

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define GET_BLOCKS(n, t) (n+t-1) / t


template <typename scalar_t>
__global__ void ScoreData(
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> query, // B, N1, 4, H, dim
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> key, //B, N2, H, dim
  torch::PackedTensorAccessor32<long,4,torch::RestrictPtrTraits> index, //B, N1, K*4, H
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> output //B, N1, 4, K*4, H
  ){
  extern __shared__ char patch_data_char[];
  
  scalar_t *feat1_data = (scalar_t *)patch_data_char;


  int b = blockIdx.x;
  int n1 = blockIdx.y;
  int f = blockIdx.z;
  
  int ch_off = threadIdx.x;
  
  int D=query.size(4);
  int HD=query.size(3)*D;
  int K=index.size(2);
  for(int ch = ch_off; ch < HD; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
    feat1_data[ch] = query[b][n1][f][ch/D][ch%D];
  }
  __syncthreads();
  
  __shared__ scalar_t score[THREADS_PER_WARP*MAX_H];
  for(int k = ch_off; k < K; k += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
      
      for(int h=0;h<query.size(3);h++){
          int score_idx=ch_off*query.size(3)+h;
          score[score_idx]=0;
          int idx=index[b][n1][k][h];
          for(int d=0;d<query.size(4);d++){
              score[score_idx]+=feat1_data[h*D+d]*key[b][idx][h][d];
          }
          output[b][n1][f][k][h]=score[score_idx];
      }
  }


}


std::vector<torch::Tensor> ScoreData_ongpu(torch::Tensor query, // B, N1, 4, H, dim
  torch::Tensor key, // B, N2, H, dim
  torch::Tensor index) // B, N1, K, 4, H
{

    const auto B = query.size(0);
    const auto N1 = query.size(1);
    const auto H = query.size(3);
    const auto D = query.size(4);
    const auto K = index.size(-2);


    auto output = torch::zeros({B, N1, 4, K, H},torch::device(torch::kCUDA));
    
    int shared_memory_per_block = H*D;
    
    dim3 totalBlocks(B, N1, 4);
    dim3 threadsPerBlock(THREADS_PER_WARP);
    AT_DISPATCH_FLOATING_TYPES(query.type(), "ScoreData_ongpu", ([&] {
      ScoreData<scalar_t><<<totalBlocks, threadsPerBlock, shared_memory_per_block * sizeof(scalar_t)>>>(
          query.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>