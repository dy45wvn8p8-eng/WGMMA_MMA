cmake_minimum_required(VERSION 3.18)
project(wgmmma_mma LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURE 100a)

include_directories(include)

add_executable(hybrid_main
src/main.cu
src/hybrid_kernel.cu
)

set_target_properties(hybrid_main PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
)
