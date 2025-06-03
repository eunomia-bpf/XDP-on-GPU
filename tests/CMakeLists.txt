# Test framework for eBPF GPU Processor

cmake_minimum_required(VERSION 3.18)
project(ebpf_gpu_tests LANGUAGES CXX)

# Find Catch2
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add test kernels
add_library(test_kernels INTERFACE)
target_sources(test_kernels INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/opencl_kernels.cl
)

# Set up CUDA PTX generation
if(USE_CUDA_BACKEND)
    # Generate PTX from CUDA kernel
    add_custom_target(generate_ptx
        COMMAND ${CMAKE_CUDA_COMPILER} -ptx
                -o ${CMAKE_CURRENT_BINARY_DIR}/ptx/cuda_kernels.ptx
                ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda_kernels.cu
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda_kernels.cu
        COMMENT "Generating PTX from CUDA kernels for architecture sm_${CMAKE_CUDA_ARCHITECTURES}"
    )
    
    # Create directory for PTX files
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ptx)
endif()

# Copy OpenCL kernels to build directory
if(USE_OPENCL_BACKEND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/opencl_kernels.cl
        ${CMAKE_CURRENT_BINARY_DIR}/cl/opencl_kernels.cl
        COPYONLY
    )
    
    # Create directory for CL files
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cl)
endif()

# Main test executable - excludes performance tests
add_executable(ebpf_gpu_tests
    test_main.cpp
    test_event_processor.cpp
    test_gpu_backend.cpp
    test_device_manager.cpp
    test_kernel_loader.cpp
    test_integration.cpp
)

# Benchmark test executable - separate for performance tests
add_executable(ebpf_gpu_benchmarks
    test_main.cpp
    test_performance.cpp
    test_performance_cpu.cpp
)

# Set test dependencies
target_link_libraries(ebpf_gpu_tests
    PRIVATE
    Catch2::Catch2WithMain
    ebpf_gpu_processor
)

target_link_libraries(ebpf_gpu_benchmarks PRIVATE
    Catch2::Catch2WithMain
    ebpf_gpu_processor
)

# Add PTX generation dependency if CUDA is enabled
if(USE_CUDA_BACKEND)
    add_dependencies(ebpf_gpu_tests generate_ptx)
    add_dependencies(ebpf_gpu_benchmarks generate_ptx)
endif()

# Set test working directory to find kernel files
set_target_properties(ebpf_gpu_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Performance test (optional)
option(BUILD_PERFORMANCE_TESTS "Build performance tests" OFF)
if(BUILD_PERFORMANCE_TESTS AND USE_CUDA_BACKEND)
    add_executable(test_performance
        performance_test.cpp
    )
    
    target_link_libraries(test_performance 
        PRIVATE
        Catch2::Catch2WithMain
        ebpf_gpu_processor
    )
    
    add_dependencies(test_performance generate_ptx)
    
    target_compile_definitions(test_performance PRIVATE 
        PTX_FILE_PATH="${CMAKE_CURRENT_BINARY_DIR}/ptx/cuda_kernels.ptx"
        CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES}"
    )
    
    include(CTest)
    include(Catch)
    catch_discover_tests(test_performance
        TEST_SPEC "[benchmark]"
    )
endif()

# Register tests with CTest
include(CTest)
include(Catch)
catch_discover_tests(ebpf_gpu_tests) 