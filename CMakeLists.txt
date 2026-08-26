cmake_minimum_required(VERSION 3.25)
project(wgmma_mma LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
set(CMAKE_CUDA_ARCHITECTURES 120) 

include_directories(include)

add_executable(hybrid_main
    src/main.cu
    src/hybrid_kernel.cu
)

set_target_properties(hybrid_main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)