// the following is the implementation of flash attention as described in the paper :  https://arxiv.org/abs/2205.14135

#include <cuda_runtime.h> 
#include <cuda_fp16.h> 
#include <stdint.h> 
#include <cooperative_groups.h> 

namespace cg = cooperative_groups; 

#define WARP_SIZE 32 
#define M // to-be defined 

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
    
    const int num_bytes = 

}

