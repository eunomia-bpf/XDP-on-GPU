cmake_minimum_required(VERSION 3.18)
project(dpdk_example LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build configuration
option(ENABLE_DEBUG "Enable debug build" OFF)

# Set build type
if(ENABLE_DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-DDEBUG)
else()
    set(CMAKE_BUILD_TYPE Release)
endif()

# Make DPDK optional
option(WITH_DPDK "Build with DPDK support" ON)

# Find DPDK if enabled
if(WITH_DPDK)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DPDK libdpdk)
    
    if(DPDK_FOUND)
        message(STATUS "DPDK found, building DPDK examples")
        set(HAVE_DPDK ON)
        add_definitions(-DHAVE_DPDK)
    else()
        message(STATUS "DPDK not found, disabling DPDK examples")
        set(WITH_DPDK OFF)
    endif()
endif()

# Find CUDA if using CUDA backend
if(USE_CUDA_BACKEND OR USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Additional required libraries
find_library(PTHREAD_LIB pthread REQUIRED)
find_library(DL_LIB dl REQUIRED)
find_library(NUMA_LIB numa)
find_library(RT_LIB rt REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

if(WITH_DPDK)
    include_directories(${DPDK_INCLUDE_DIRS})
    add_definitions(-include rte_config.h)
endif()

# Add necessary definitions
add_definitions(-D_GNU_SOURCE)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
if(ENABLE_DEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
endif()

# Add DPDK compile flags if enabled
if(WITH_DPDK)
    add_compile_options(${DPDK_CFLAGS})
endif()

# Configure build targets based on available components
if(WITH_DPDK)
    # Source files for C library
    set(C_SOURCES
        dpdk_driver.c
    )

    # Create C library for DPDK driver
    add_library(dpdk_driver STATIC ${C_SOURCES})

    # Main application source
    set(SOURCES
        dpdk_example.cpp
    )

    # Main application executable
    add_executable(dpdk_example ${SOURCES})

    # Link libraries
    target_link_libraries(dpdk_example
        dpdk_driver
        ebpf_gpu_processor
        ${DPDK_LIBRARIES}
        ${PTHREAD_LIB}
        ${DL_LIB}
        ${RT_LIB}
    )
    
    # Add backend-specific libraries
    if(USE_CUDA_BACKEND OR USE_CUDA)
        target_link_libraries(dpdk_example 
            CUDA::cudart
            CUDA::cuda_driver
        )
    elseif(USE_OPENCL_BACKEND OR USE_OPENCL)
        target_link_libraries(dpdk_example 
            OpenCL::OpenCL
        )
    endif()

    # Link numa if available
    if(NUMA_LIB)
        target_link_libraries(dpdk_example ${NUMA_LIB})
    endif()

    # Make sure ebpf_gpu_processor is built first
    add_dependencies(dpdk_example ebpf_gpu_processor)

    # Installation
    install(TARGETS dpdk_example dpdk_driver
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )

    # Custom targets for testing
    add_custom_target(test-null
        COMMAND sudo $<TARGET_FILE:dpdk_example> --vdev=net_null0 -l 0
        DEPENDS dpdk_example
        COMMENT "Test with null PMD"
    )

    add_custom_target(test-gpu-null
        COMMAND sudo $<TARGET_FILE:dpdk_example> --vdev=net_null0 -l 0 -- --kernel=${CMAKE_SOURCE_DIR}/examples/simple_packet_filter.cu --function=packet_filter
        DEPENDS dpdk_example
        COMMENT "Test with null PMD and GPU processing"
    )

    add_custom_target(test-tap
        COMMAND sudo $<TARGET_FILE:dpdk_example> --vdev=net_tap0,iface=test0 -l 0
        DEPENDS dpdk_example
        COMMENT "Test with TAP interface"
    )

    add_custom_target(test-ring
        COMMAND sudo $<TARGET_FILE:dpdk_example> --vdev=net_ring0 -l 0
        DEPENDS dpdk_example
        COMMENT "Test with ring PMD"
    )

    add_custom_target(test-multi
        COMMAND sudo $<TARGET_FILE:dpdk_example> --vdev=net_null0 --vdev=net_null1 -l 0
        DEPENDS dpdk_example
        COMMENT "Test with multiple devices"
    )

    # Make sure examples directory exists
    add_custom_command(
        TARGET dpdk_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/examples
        COMMENT "Ensuring examples directory exists"
    )
    
    # Copy CUDA example to examples directory if not already there
    add_custom_command(
        TARGET dpdk_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/example/simple_packet_filter.cu
            ${CMAKE_SOURCE_DIR}/examples/simple_packet_filter.cu
        COMMENT "Copying CUDA example to examples directory"
    )
else()
    # Build a simple example without DPDK
    message(STATUS "Building basic examples without DPDK")
    
    # Create a simple example that demonstrates the library functionality
    add_executable(simple_example simple_example.cpp)
    
    # Pass the appropriate backend define to the compiler
    if(USE_CUDA_BACKEND OR USE_CUDA)
        target_compile_definitions(simple_example PRIVATE USE_CUDA_BACKEND)
    elseif(USE_OPENCL_BACKEND OR USE_OPENCL)
        target_compile_definitions(simple_example PRIVATE USE_OPENCL_BACKEND)
    endif()
    
    target_link_libraries(simple_example
        ebpf_gpu_processor
        ${PTHREAD_LIB}
        ${DL_LIB}
        ${RT_LIB}
    )
    
    # Add backend-specific libraries
    if(USE_CUDA_BACKEND OR USE_CUDA)
        target_link_libraries(simple_example 
            CUDA::cudart
            CUDA::cuda_driver
        )
    elseif(USE_OPENCL_BACKEND OR USE_OPENCL)
        target_link_libraries(simple_example 
            OpenCL::OpenCL
        )
    endif()
    
    add_dependencies(simple_example ebpf_gpu_processor)

    # Make sure examples directory exists
    add_custom_command(
        TARGET simple_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/examples
        COMMENT "Ensuring examples directory exists"
    )
    
    # Copy CUDA example to examples directory if not already there
    add_custom_command(
        TARGET simple_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/example/simple_packet_filter.cu
            ${CMAKE_SOURCE_DIR}/examples/simple_packet_filter.cu
        COMMENT "Copying CUDA example to examples directory"
    )
endif()

# Configuration summary
message(STATUS "Build configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  ENABLE_DEBUG: ${ENABLE_DEBUG}")
if(WITH_DPDK)
    message(STATUS "  DPDK_LIBRARIES: ${DPDK_LIBRARIES}")
    message(STATUS "  DPDK_CFLAGS: ${DPDK_CFLAGS}")
endif() 