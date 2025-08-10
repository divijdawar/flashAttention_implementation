//  he following is the implementation of flash attention as described in the paper: https://arxiv.org/abs/2205.14135

#include <cuda_runtime.h> 
#include <cuda_fp16.h> 
#include <stdint.h>
#include <type_traits> 

namespace cg = cooperative_groups; 

#define M 96 * 1024
#define WARP_SIZE 32

inline constexpr int num_elems = M / sizeof(__half); 

template<int HEAD_DIM> 
inline constexpr int Bc() {return (num_elems + 4 * HEAD_DIM - 1) / (4 * HEAD_DIM);} // block columns -> the number of key/value vectors processed per tile iteration

template<int HEAD_DIM> 
inline constexpr int Br() { return std::min(Bc<HEAD_DIM>(), HEAD_DIM);} // block rows -> the number of rows of query vectors processed per tile iteration

// 16b zero stores 
// todo: update v8 to v4
__device__ __forceinline__ void store16b( void* p) { 
    asm volatile("st.global.v8.u16 [%0], {0,0,0,0,0,0,0,0};" :: "l"(p) : "memory");
}

template<typename T> 
__device__ void initialze(
    T* __restrict__ out, 
    float* __restrict__ l,   // [N]
    float* __restrict__ m,   // [N]
    int N,                   // rows
    int d,                   // active columns per row 
) { 
    const float neg_inf = -CUDART_INF_F;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    const int total_elems = N * d; 
    
    for (int i = tid * 8; i < total_elems; i += 8 * gridDim.x * blockDim.x) { 
        store16b(reinterpret_cast<void*>(0+i)); 
    }

    const uint32_t zero = __float_as_uint(0.0f); 
    const uint32_t inf = __float_as_uint(-CUDART_INF_F); 

    for (int i = tid; i < N; i += 4 * gridDim.x * blockDim.x){
        store16b<zero>(reinterpret_cast<void*>(l + i));
        store16b<inf>(reinterpret_cast<void*>(m + i));
    }
} 

template <int HEAD_DIM> 
void main() {
    constexpr int Br = Br<HEAD_DIM>();
    constexpr int Bc = Bc<HEAD_DIM>(); 

    const int Tr = (N+Br-1)/ Br; 
    const int Tc = (N+Bc-1)/ Bc; 
    
}
