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

# Copy test PTX file to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ptx.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/test_ptx.ptx
    COPYONLY
)

# Set up backend-specific test configuration
if(USE_CUDA)
    # Generate PTX from CUDA kernels at compile time
    set(PTX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")
    file(MAKE_DIRECTORY ${PTX_OUTPUT_DIR})
    
    # Generate PTX from cuda_kernels.cu
    set(CUDA_KERNEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels.cu")
    set(PTX_OUTPUT_FILE "${PTX_OUTPUT_DIR}/cuda_kernels.ptx")
    
    # Get the first detected CUDA architecture for PTX generation
    list(GET CMAKE_CUDA_ARCHITECTURES 0 PTX_ARCH)
    
    # Custom command to generate PTX
    add_custom_command(
        OUTPUT ${PTX_OUTPUT_FILE}
        COMMAND ${CMAKE_CUDA_COMPILER} -ptx -arch=sm_${PTX_ARCH}
                ${CUDA_KERNEL_FILE} -o ${PTX_OUTPUT_FILE}
                -I${CMAKE_SOURCE_DIR}/include
        DEPENDS ${CUDA_KERNEL_FILE}
        COMMENT "Generating PTX from CUDA kernels for architecture sm_${PTX_ARCH}"
    )
    
    # Create a target for PTX generation
    add_custom_target(generate_ptx DEPENDS ${PTX_OUTPUT_FILE})
    
    # Test kernels
    add_library(test_kernels OBJECT
        cuda_kernels.cu
    )
    
    set_target_properties(test_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    # Define test configuration
    set(TEST_DEFINITIONS 
        PTX_FILE_PATH="${PTX_OUTPUT_FILE}"
        CUDA_ARCH_SM=${PTX_ARCH}
        USE_CUDA_BACKEND
    )
    
    set(TEST_KERNEL_OBJECTS $<TARGET_OBJECTS:test_kernels>)
    set(TEST_PTX_DEPENDENCY generate_ptx)
    
elseif(USE_OPENCL)
    # OpenCL test configuration
    set(TEST_DEFINITIONS
        USE_OPENCL_BACKEND
    )
    
    set(TEST_KERNEL_OBJECTS "")
    set(TEST_PTX_DEPENDENCY "")
endif()

# Performance tests (GPU only)
add_executable(test_performance
    test_performance.cpp
    ${TEST_KERNEL_OBJECTS}
)

target_link_libraries(test_performance 
    ebpf_gpu_processor
    Catch2::Catch2WithMain
)

if(USE_CUDA)
    add_dependencies(test_performance ${TEST_PTX_DEPENDENCY})
endif()

target_compile_definitions(test_performance PRIVATE ${TEST_DEFINITIONS})

# CPU Performance tests (benchmarks)
add_executable(test_performance_cpu
    test_performance_cpu.cpp
    ${TEST_KERNEL_OBJECTS}
)

target_link_libraries(test_performance_cpu 
    ebpf_gpu_processor
    Catch2::Catch2WithMain
)

if(USE_CUDA)
    add_dependencies(test_performance_cpu ${TEST_PTX_DEPENDENCY})
endif()

target_compile_definitions(test_performance_cpu PRIVATE ${TEST_DEFINITIONS})

# Integration tests
add_executable(test_integration
    test_integration.cpp
    ${TEST_KERNEL_OBJECTS}
)

target_link_libraries(test_integration 
    ebpf_gpu_processor
    Catch2::Catch2WithMain
)

if(USE_CUDA)
    add_dependencies(test_integration ${TEST_PTX_DEPENDENCY})
endif()

target_compile_definitions(test_integration PRIVATE ${TEST_DEFINITIONS})

# Device manager tests
add_executable(test_device_manager
    test_device_manager.cpp
)

target_link_libraries(test_device_manager
    ebpf_gpu_processor
    Catch2::Catch2WithMain
)

target_compile_definitions(test_device_manager PRIVATE ${TEST_DEFINITIONS})

# Kernel loader tests
add_executable(test_kernel_loader
    test_kernel_loader.cpp
)

target_link_libraries(test_kernel_loader
    ebpf_gpu_processor
    Catch2::Catch2WithMain
)

target_compile_definitions(test_kernel_loader PRIVATE ${TEST_DEFINITIONS})

# Register tests with CTest
include(CTest)

# Only include Catch if it's available
if(TARGET Catch2::Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(Catch)
    
    # Regular tests (exclude benchmark tests)
    catch_discover_tests(test_integration)
    catch_discover_tests(test_device_manager)
    catch_discover_tests(test_kernel_loader)
    
    # Performance tests without benchmarks (exclude [benchmark] tag)
    catch_discover_tests(test_performance 
        TEST_SPEC "~[benchmark]"
    )
    
    # Benchmark tests (separate target, not included in regular test runs)
    catch_discover_tests(test_performance_cpu 
        TEST_PREFIX "bench_"
        TEST_SPEC "[benchmark]"
    )
    
    # Also add GPU benchmarks
    catch_discover_tests(test_performance 
        TEST_PREFIX "bench_"
        TEST_SPEC "[benchmark]"
    )
endif() 