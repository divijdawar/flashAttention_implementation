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
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int total_elems = N * d; 
    
    for (int i = idx; i < total_elems; i += blockIdx.x * gridDim.x) { 
        out[i] = static_cast<T>(0.0f); 
    }

    for (int i = idx; i < N; i += blockDim.x * gridDim.x){
        l[i] = 0.0f; 
        m[i] = -CUDART_INF_F;
    }
}
