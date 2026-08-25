#include "hybrid_kernel.cuh"

//cuda kernel definition
__global__ void hybrid_wgmma_mma_kernel(float* out, const half* in, int size) {
    // shared memory allocation
    extern __shared__ char smem_raw[];
    half* smem_tile = reinterpret_cast<half*>(smem_raw);

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t smem_addr = cvt_generic_to_shared(smem_tile);


    // main processing loop
    if (tid < size) {
        // wgmma async pipeline & mma fallback 
    }
}

void launch_hybrid_kernel(float* d_out, const half* d_in, int size, cudaStream_t stream) {
    dim3 grid(1);
    dim3 block(128); // 4warps = 1 warp-group
    size_t smem_bytes = 1024 * sizeof(half);

    hybrid_wgmma_mma_kernel<<<grid, block, smem_bytes, stream>>>(d_out, d_in, size);
    CUDA_CHECK(cudaGetLastError());
}