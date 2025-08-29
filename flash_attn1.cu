//  he following is the implementation of flash attention as described in the paper: https://arxiv.org/abs/2205.14135

#include <cuda_runtime.h> 
#include <cuda_fp16.h> 
#include <stdint.h>
#include <type_traits> 
#include <cooperative_groups.h> 

namespace cg = cooperative_groups; 

#define M 96 * 1024
#define WARP_SIZE 32

inline constexpr int num_elems = M / sizeof(__half); 

#define CUDA_CHECK(call)                                               \
do{                                                                    \
    cudaError_t err = call;                                            \
    if(err != cudaSuccess) {                                           \
        printf("CUDA error at %s %d: %s: \n ", __FILE__, __LINE__,     \
                cudaGetErrorString(err));                              \
        return;                                                        \
    }                                                                   \
} while (0)

template<int HEAD_DIM> 
inline constexpr int Bc() {return (num_elems + 4 * HEAD_DIM - 1) / (4 * HEAD_DIM);} // block columns -> the number of key/value vectors processed per tile iteration

template<int HEAD_DIM> 
inline constexpr int Br() { return std::min(Bc<HEAD_DIM>(), HEAD_DIM);} // block rows -> the number of rows of query vectors processed per tile iteration

// 16b zero stores 
__device__ __forceinline__ void store16b( void* p) { 
    asm volatile("st.global.v8.u16 [%0], {0,0,0,0,0,0,0,0};" :: "l"(p) : "memory");
}

__device__ __forceinline__ void store_q(void *dest, const void* src) { 
    uint32_t r0, r1, r2, r3; 
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
    : = "r"(r0)
}

struct __align__(8) half4 {half w,x, y,z};
__device__ half4 make_half4(half x, half y, half z, half z) { half r={w,x,y,z}; return r;}

struct __align__(16) half8 {half x, y, z,w, a, b, c, d;};
__device__ half8 make_half8(half x, half y, half z, half w, half a, half b, half c, half d) { half8 r={x, y, z, w, a, b, c, d}; return r; }

__device__ void __ldmatrix_a(half8 *regs, half *smem){ 
    uint32_t reg0, reg1, reg3, reg3; 
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3)
        : "l"(__cvta_generic_to_shared(smem))
    );
    uint32_t *addr = reinterpret_cast<uint32_t*>(regs); 
    addr[0] = reg0; 
    addr[1] = reg1; 
    addr[2] = reg2;
    addr[3] = reg3; 
}

__device__ void __ldmatrix_b(half4 *regs_lo, half4 *regs_hi, half* smem) { 
    uint32_t reg0, reg1, reg2, reg3; 
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3)
        : "l"(__cvta_generic_to_shared(smem))
    );
    uint32_t *addr_lo = reinterpret_cast<uint32_t*>(regs_lo);
    uint32_t *addr_hi = reinterpret_cast<uint32_t*>(regs_hi);
    addr_lo[0] = reg0; 
    addr_lo[1] = reg1; 
    addr_hi[0] = reg2; 
    addr_hi[1] = reg3; 
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

template <int HEAD_DIM, int Br, int Bc> 
__global__ void flash_attn1(
    const __half* __restrict__ Q, 
    const __half* __restrict__ K, 
    const __half* __restrict__ V, 
          __half* __restrict__ O, 
    float*        __restrict__ l, 
    float*        __restrict__ m, 
    int N, 
    float scale
) { 
    static_assert(HEAD_DIM % 8 == 0, "HEAD_DIM must be a multiple of 8 for vectorized loads")
    constexpr int bc = Bc<HEAD_DIM>(); 
    constexpr int br = Br<HEAD_DIM>(); 
    const     int tc = (N + bc -1) / bc;
    const     int tr = (N + br -1) / br;

    const int tile_y = blockIdx.y;
    const int tile_x = blockIdx.x; 
    const int idx = threadIdx.x; 
    const int lane = idx & ( WARP_SIZE - 1);
    const int warp_id = threadIdx.x / WARP_SIZE; 
    const int lane_id = threadIdx.x % WARP_SIZE; 

    extern __shared__ char shared[]; 
    __half* sq = reinterpret_cast<__half*>(shared); // br * d 
    __half* sk = reinterpret_cast<__half*>(sq + br * HEAD_DIM); // bc * d
    __half* sv = reinterpret_cast<__half*>(sk + bc * HEAD_DIM); // bc * d 

    for (int j = 0; j < tc; j++){

        const int k_offset = j * bc * HEAD_DIM;
        const int v_offset = j * bc * HEAD_DIM;

        for (int i = 0; i < tr; i++) {
            // load q, o, l , m from HBM to on-chip SRAM 
        }
    }

}

void main() {
    
}
