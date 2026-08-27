#include "common.h"
#include "hybrid_kernel.cuh"

int main() {
    std::cout << "=== CUDA / PTX Custom Kernel Pipeline ===" << std::endl;

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) return 1;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    constexpr int N = 1024 * 1024 * 16;
    float *d_out = nullptr;
    half *d_in = nullptr;

    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(half)));

    launch_hybrid_kernel(d_out, d_in, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_in));
    return 0;
}