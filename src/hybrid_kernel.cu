#include "hybrid_kernel.cuh"

//cuda kernel definition
__global__ void hybrid_wgmma_mma_kernel(float* d_out, const half* d_in, int size) {
    // shared memory allocation
    extern __shared__ char smem_raw[];
    half* smem_tile = reinterpret_cast<half*>(smem_raw);

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t smem_addr = cvt_generic_to_shared(smem_tile);


    // main processing loop
    if (tid < size) {
        float val = __half2float(d_in[tid]);
        float acc0 = val;
        float acc1 = val * 1.001f;
        float acc2 = val * 1.002f;
        float acc3 = val * 1.003f;

        #pragma unroll 8
        for (int i = 0; i < 500; ++i) {
            acc0 = acc0 * 1.001f + 0.0001f;
            acc1 = acc1 * 1.002f + 0.0002f;
            acc2 = acc2 * 1.003f + 0.0003f;
            acc3 = acc3 * 1.004f + 0.0004f;
        }

        d_out[tid] = acc0 + acc1 + acc2 + acc3;
        // wgmma async pipeline & mma fallback 
    }
}

void launch_hybrid_kernel(float* d_out, const half* d_in, int size, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) /block_size;
    dim3 grid(num_blocks * 4);
    dim3 block(block_size); // 4warps = 1 warp-group
    size_t smem_bytes = 1024 * sizeof(half);

    hybrid_wgmma_mma_kernel<<<grid, block, smem_bytes, stream>>>(d_out, d_in, size);
    CUDA_CHECK(cudaGetLastError());
}