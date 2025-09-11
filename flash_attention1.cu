// the following is the implementation of flash attention as described in the paper :  https://arxiv.org/abs/2205.14135

#include <cuda_runtime.h> 
#include <cuda_fp16.h> 
#include <stdint.h> 
#include <cooperative_groups.h>
#include <vector> 
#include <cuda_pipeline.h> 

namespace cg = cooperative_groups; 

#define WARP_SIZE 32 
#define M // to-be defined 

#define CUDA_CHECK(call) { 
do { 
    cudaError_t err = (call); 
    if (err != cudaSuccess) { 
        fprintf(stderr, "CUDA error at %s:%d: %s\n in call: %s \n",
          __FILE__ __LINE__, cudaGetErrorString(err), #call);
        std::abort(); 
    }
} while (0)

inline constexpr int num_elems = M / sizeof(__half); 

template<int HEAD_DIM> 
inline constexpr int Bc() { return (num_elems + 4 * HEAD_DIM - 1) / (4 * HEAD_DIM);}  // block columns -> the number of key/value vectors processed per iteration

template<int HEAD_DIM> 
inline constexpr int Br(){ return std::min(Bc<HEAD_DIM>(), HEAD_DIM); }  // block rows -> the number of rows of query vectors processed per tile interation 

template<int HEAD_DIM> 
__global__ void flash_attention1 { 
    const __half* __restrict__ Q, 
    const __half* __restrict__ K, 
    const __half* __restrict__ V, 
          __half* __restrict__ O, 
    float*        __restrict__ l, 
    float*        __restrict__ m, 
    int N, 
    float scale
} { 
    static_assert(HEAD_DIM % 8 == 0, "HEAD_DIM must be a multiple of 8 for vectorized loads")
    constexpr int bc = Bc<HEAD_DIM>(); 
    constexpr int br = Br<HEAD_DIM>(); 
    constexpr int tr = (N + br - 1) / br; // divides Q into blocks of br * d  
    constexpr int tc = (N + bc -1)  / bc; // divides k,v into blocks of bc * d 
    
    size_t o_size = sizeof(__half) * N * HEAD_DIM; 
    size_t l_size = sizeof(__half) * N; 
    size_t m_size = sizeof(__half) * N; 

    CUDA_CHECK(cudaStreamCreate(&stream1)); 
    CUDA_CHECK(cudaStreamCreate(&stream2)); 

    cudaMallocAsync((void**)&O, o_size, stream1); 
    cudaMallocAsync((void**)&l, l_size, stream1); 
    cudaMallocAsync((void**)&m, m_size, stream1); 

    CUDA_CHECK(cudaMemsetAsync(O, 0, o_size, stream1)); 
    CUDA_CHECK(cudaMemsetAsync(l, 0, l_size, stream1)); 
    
    std::vector<float> m_host (N, -std::numeric_limits<float>::infinity());
    CUDA_CHECK(cudaMemsetAsync(m, m_host.data(), m_size, cudaMemcpyHostToDevice, stream1));

    __shared__ __half K_j[2][bc][HEAD_DIM];
    __shared__ __half V_j[2][bc][HEAD_DIM];
    __shared__ __half V_i[2][br][HEAD_DIM]; 

    for (int j = 0; j < tc; j++) {
        
        for (int i = 0; i < tr; i++) { 
            
        }
    }
}

void main() { 
    
}

