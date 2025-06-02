cmake_minimum_required(VERSION 3.18)
project(ebpf_gpu_processor LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find CUDA
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# Detect GPU architecture automatically
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Try to detect GPU architecture
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
        OUTPUT_VARIABLE GPU_COMPUTE_CAPS
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(GPU_COMPUTE_CAPS)
        string(REPLACE "\n" ";" GPU_COMPUTE_CAPS_LIST ${GPU_COMPUTE_CAPS})
        list(GET GPU_COMPUTE_CAPS_LIST 0 FIRST_GPU_CAP)
        string(REPLACE "." "" CUDA_ARCH ${FIRST_GPU_CAP})
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
        message(STATUS "Detected GPU compute capability: ${FIRST_GPU_CAP}, using architecture: ${CUDA_ARCH}")
    else()
        # Fallback to common architectures
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
        message(STATUS "Could not detect GPU, using fallback architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

# Include directories
include_directories(include)

# Library source files
set(LIB_SOURCES
    src/ebpf_gpu_processor.cpp
    src/gpu_device_manager.cpp
    src/kernel_loader.cpp
)

# Create the library
add_library(ebpf_gpu_processor STATIC ${LIB_SOURCES})

# Set target properties
set_target_properties(ebpf_gpu_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link CUDA libraries
target_link_libraries(ebpf_gpu_processor 
    CUDA::cuda_driver 
    CUDA::cudart
)

# Build the examples
option(BUILD_EXAMPLES "Build example applications" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

# Test framework
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS ebpf_gpu_processor
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
) 