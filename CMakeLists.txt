cmake_minimum_required(VERSION 3.18)
project(ebpf_gpu_processor LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Backend selection option
option(AUTO_DETECT_BACKEND "Auto-detect the best available backend" ON)
option(USE_CUDA "Use CUDA backend (default)" OFF)
option(USE_OPENCL "Use OpenCL backend" OFF)

# Auto-detect the backend if requested
if(AUTO_DETECT_BACKEND)
    # First, try to find CUDA
    include(FindCUDA OPTIONAL)
    find_package(CUDAToolkit QUIET)
    if(CUDA_FOUND OR CUDAToolkit_FOUND)
        set(USE_CUDA ON)
        message(STATUS "Auto-detected CUDA, enabling CUDA backend")
    else()
        # If CUDA not found, try OpenCL
        find_package(OpenCL QUIET)
        if(OpenCL_FOUND)
            set(USE_OPENCL ON)
            message(STATUS "Auto-detected OpenCL, enabling OpenCL backend")
        else()
            message(FATAL_ERROR "No GPU backends detected. Please install CUDA or OpenCL development packages.")
        endif()
    endif()
endif()

# Validate backend choice
if(USE_CUDA AND USE_OPENCL)
    message(FATAL_ERROR "Cannot use both CUDA and OpenCL backends at the same time. Choose one.")
endif()

if(NOT USE_CUDA AND NOT USE_OPENCL)
    message(STATUS "No backend explicitly selected. Defaulting to CUDA.")
    set(USE_CUDA ON)
endif()

# Setup backend-specific configuration
if(USE_CUDA)
    # Configure CUDA backend
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Find CUDA
    find_package(CUDAToolkit REQUIRED)
    
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
    
    # Use CUDA implementation
    set(LIB_SOURCES
        src/cuda/ebpf_gpu_processor.cpp
        src/cuda/gpu_device_manager.cpp
        src/cuda/kernel_loader.cpp
    )
    
    # Add CUDA-specific dependencies
    set(BACKEND_LIBS CUDA::cuda_driver CUDA::cudart)
    
    # Define CUDA backend flag - use add_compile_definitions for modern CMake
    add_compile_definitions(USE_CUDA_BACKEND)
    
elseif(USE_OPENCL)
    # Configure OpenCL backend
    find_package(OpenCL REQUIRED)
    
    # Use OpenCL implementation
    set(LIB_SOURCES
        src/opencl/ebpf_gpu_processor.cpp
        src/opencl/gpu_device_manager.cpp
        src/opencl/kernel_loader.cpp
    )
    
    # Add OpenCL-specific dependencies
    set(BACKEND_LIBS OpenCL::OpenCL)
    
    # Define OpenCL backend flag
    add_compile_definitions(USE_OPENCL_BACKEND)
endif()

if(USE_CUDA)
    message(STATUS "Using backend: CUDA")
else()
    message(STATUS "Using backend: OpenCL")
endif()

# Include directories
include_directories(include)

# Create the library
add_library(ebpf_gpu_processor STATIC ${LIB_SOURCES})

# Set target properties
set_target_properties(ebpf_gpu_processor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add CUDA-specific properties if using CUDA
if(USE_CUDA)
    set_target_properties(ebpf_gpu_processor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Link backend-specific libraries
target_link_libraries(ebpf_gpu_processor ${BACKEND_LIBS})

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