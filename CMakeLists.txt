cmake_minimum_required(VERSION 3.18)
project(cuda_event_processor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# Set CUDA flags for H100 (compute capability 9.0)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")

# Include directories
include_directories(include)

# Library source files
set(LIB_SOURCES
    src/cuda_event_processor.cpp
)

# Create the library
add_library(cuda_event_processor STATIC ${LIB_SOURCES})

# Link CUDA libraries
target_link_libraries(cuda_event_processor CUDA::cuda_driver CUDA::cudart)

# Test executable with CUDA kernels
add_executable(test_processor 
    tests/test_basic.cpp
    tests/cuda_kernels.cu
)
target_link_libraries(test_processor cuda_event_processor)

# Set properties
set_target_properties(cuda_event_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
) 