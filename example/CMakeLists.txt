cmake_minimum_required(VERSION 3.18)
project(ebpf_gpu_examples LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Get backend configuration from parent project
if(USE_CUDA_BACKEND OR USE_OPENCL_BACKEND)
    # Add the multi-backend example
    add_executable(multi_backend_example multi_backend_example.cpp)
    target_link_libraries(multi_backend_example ebpf_gpu_processor)

    # Copy kernel files to build directory
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/opencl_kernel.cl")
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/opencl_kernel.cl
                      ${CMAKE_CURRENT_BINARY_DIR}/opencl_kernel.cl COPYONLY)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernel.cu")
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernel.cu
                      ${CMAKE_CURRENT_BINARY_DIR}/cuda_kernel.cu COPYONLY)
    endif()

    # Installation
    install(TARGETS multi_backend_example
        RUNTIME DESTINATION bin
    )
else()
    message(STATUS "No GPU backends available. Examples will not be built.")
endif()

# Configuration summary
message(STATUS "eBPF GPU Examples configuration:")
message(STATUS "  CUDA Backend: ${USE_CUDA_BACKEND}")
message(STATUS "  OpenCL Backend: ${USE_OPENCL_BACKEND}") 