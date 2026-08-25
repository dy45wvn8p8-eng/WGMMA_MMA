#ifndef HYBRID_KERNEL_CUH
#define HYBRID_KERNEL_CUH

#include "common.h"

// PTX Inline Wrapper Interface
__device__ __forceinline__ uint32_t cvt_generic_to_shared(const void* ptr) {
    uint32_t smem_ptr;
    asm volatile(
        "{\n\t"
        "  .reg .u64 p64;\n\t"
        "  cvta.to.shared.u64 p64, %1;\n\t"
        "  cvt.u32.u64 %0, p64;\n\t"
        "}"
        : "=r"(smem_ptr)
        : "l"(ptr)
    );
    return smem_ptr;
}

void launch_hybrid_kernel(float* d_out, const half* d_in, int size, cudaStream_t stream = 0);

#endif // HYBRID_KERNEL_CUH