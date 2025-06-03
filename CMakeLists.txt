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

# Backend options
option(USE_CUDA_BACKEND "Enable CUDA backend" ON)
option(USE_OPENCL_BACKEND "Enable OpenCL backend" OFF)
option(AUTO_DETECT_BACKENDS "Auto-detect available backends" ON)

# Auto-detection of backends if enabled
if(AUTO_DETECT_BACKENDS)
    # Try to find CUDA
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
        set(CUDA_AVAILABLE TRUE)
    else()
        message(STATUS "CUDA compiler not found, disabling CUDA backend")
        set(CUDA_AVAILABLE FALSE)
        set(USE_CUDA_BACKEND OFF)
    endif()

    # Try to find OpenCL
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL found: ${OpenCL_LIBRARIES}")
        set(OPENCL_AVAILABLE TRUE)
    else()
        message(STATUS "OpenCL not found, disabling OpenCL backend")
        set(OPENCL_AVAILABLE FALSE)
        set(USE_OPENCL_BACKEND OFF)
    endif()

    # Enable at least one backend if available
    if(NOT USE_CUDA_BACKEND AND OPENCL_AVAILABLE)
        message(STATUS "CUDA disabled, enabling OpenCL backend automatically")
        set(USE_OPENCL_BACKEND ON)
    endif()
    
    if(NOT USE_OPENCL_BACKEND AND CUDA_AVAILABLE)
        message(STATUS "OpenCL disabled, enabling CUDA backend automatically")
        set(USE_CUDA_BACKEND ON)
    endif()
endif()

# Check if at least one backend is enabled
if(NOT USE_CUDA_BACKEND AND NOT USE_OPENCL_BACKEND)
    message(WARNING "No GPU backends available. The library will be built with limited functionality.")
    # Still continue the build without backends
endif()

# Common include directories
include_directories(include)

# Setup CUDA backend if enabled
if(USE_CUDA_BACKEND)
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
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
            message(STATUS "Could not detect GPU, using fallback architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    endif()

    # CUDA compilation flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
    endif()
    
    add_definitions(-DUSE_CUDA_BACKEND)
endif()

# Setup OpenCL backend if enabled
if(USE_OPENCL_BACKEND)
    find_package(OpenCL REQUIRED)
    add_definitions(-DUSE_OPENCL_BACKEND)
    message(STATUS "OpenCL include dirs: ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "OpenCL libraries: ${OpenCL_LIBRARIES}")
    include_directories(${OpenCL_INCLUDE_DIRS})
endif()

# Library source files
set(LIB_SOURCES
    src/ebpf_gpu_processor.cpp
    src/gpu_device_manager.cpp
    src/kernel_loader.cpp
    src/backend_factory.cpp
)

# Add backend-specific source files
if(USE_CUDA_BACKEND)
    list(APPEND LIB_SOURCES 
        src/cuda_backend.cpp
    )
endif()

if(USE_OPENCL_BACKEND)
    list(APPEND LIB_SOURCES 
        src/opencl_backend.cpp
    )
endif()

# Create the library
add_library(ebpf_gpu_processor STATIC ${LIB_SOURCES})

# Set target properties
set_target_properties(ebpf_gpu_processor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link backend-specific libraries
if(USE_CUDA_BACKEND)
    target_link_libraries(ebpf_gpu_processor 
        CUDA::cuda_driver 
        CUDA::cudart
    )
    
    set_target_properties(ebpf_gpu_processor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

if(USE_OPENCL_BACKEND)
    target_link_libraries(ebpf_gpu_processor 
        ${OpenCL_LIBRARIES}
    )
endif()

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

# Print configuration summary
message(STATUS "eBPF-on-GPU configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA backend: ${USE_CUDA_BACKEND}")
message(STATUS "  OpenCL backend: ${USE_OPENCL_BACKEND}") 