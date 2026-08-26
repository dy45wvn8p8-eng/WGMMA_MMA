#pragma once
#if defined(__CUDACC__)
  #ifndef __STDC_VERSION__
    #define __STDC_VERSION__ 201112L
  #endif
  #define __HAVE_FLOAT64X 0
  #define __HAVE_FLOAT32X 0
#endif

#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)