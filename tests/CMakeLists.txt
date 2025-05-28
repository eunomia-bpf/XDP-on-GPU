# Test framework for eBPF GPU Processor

# Find or download a simple test framework (using a header-only approach)
include(FetchContent)

# Use Catch2 for testing (header-only version)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)

FetchContent_MakeAvailable(Catch2)

# Generate PTX from CUDA kernels at compile time
set(PTX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")
file(MAKE_DIRECTORY ${PTX_OUTPUT_DIR})

# Generate PTX from cuda_kernels.cu
set(CUDA_KERNEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels.cu")
set(PTX_OUTPUT_FILE "${PTX_OUTPUT_DIR}/cuda_kernels.ptx")

# Custom command to generate PTX
add_custom_command(
    OUTPUT ${PTX_OUTPUT_FILE}
    COMMAND ${CMAKE_CUDA_COMPILER} -ptx -arch=sm_61 
            ${CUDA_KERNEL_FILE} -o ${PTX_OUTPUT_FILE}
            -I${CMAKE_SOURCE_DIR}/include
    DEPENDS ${CUDA_KERNEL_FILE}
    COMMENT "Generating PTX from CUDA kernels"
)

# Create a target for PTX generation
add_custom_target(generate_ptx DEPENDS ${PTX_OUTPUT_FILE})

# Test kernels
add_library(test_kernels OBJECT
    kernels/test_kernels.cu
)

set_target_properties(test_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Basic functionality tests
add_executable(test_basic
    test_basic.cpp
    test_device_manager.cpp
    test_kernel_loader.cpp
    test_error_handling.cpp
    $<TARGET_OBJECTS:test_kernels>
)

target_link_libraries(test_basic 
    ebpf_gpu_processor
    Catch2::Catch2WithMain
)

# Add dependency on PTX generation and pass PTX path as compile definition
add_dependencies(test_basic generate_ptx)
target_compile_definitions(test_basic PRIVATE 
    PTX_FILE_PATH="${PTX_OUTPUT_FILE}"
)

# Performance tests
add_executable(test_performance
    test_performance.cpp
    $<TARGET_OBJECTS:test_kernels>
)

target_link_libraries(test_performance 
    ebpf_gpu_processor
    Catch2::Catch2WithMain
)

# Integration tests
add_executable(test_integration
    test_integration.cpp
    $<TARGET_OBJECTS:test_kernels>
)

target_link_libraries(test_integration 
    ebpf_gpu_processor
    Catch2::Catch2WithMain
)

# Legacy C API tests
add_executable(test_c_api
    test_c_api.cpp
    $<TARGET_OBJECTS:test_kernels>
)

target_link_libraries(test_c_api 
    ebpf_gpu_processor
    Catch2::Catch2WithMain
)

# Register tests with CTest
include(CTest)

# Only include Catch if it's available
if(TARGET Catch2::Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(Catch)
    catch_discover_tests(test_basic)
    catch_discover_tests(test_performance)
    catch_discover_tests(test_integration)
    catch_discover_tests(test_c_api)
endif() 