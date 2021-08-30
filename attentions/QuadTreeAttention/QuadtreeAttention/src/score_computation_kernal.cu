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

#define CUDA_KERNEL_LOOP(i, n) fo