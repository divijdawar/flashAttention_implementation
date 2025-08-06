#include <cuda_runtime.h>
#include <cuda_fp16.h> 

template<typename T>
__global__ void initialize(
    T* __restrict__ out, 
    float* __restrict__ l, 
    float* __restrict__ m, 
    int N, 
    int d, 
    int ldo
){ 
    const int row = blockIdx.y * blockDim,y + threadidx.y; 
    if (row >= N) return;

    
}