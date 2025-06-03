# Main source directory

# Common source files
set(COMMON_SOURCES
    ebpf_gpu_processor.cpp
    gpu_device_manager.cpp
    kernel_loader.cpp
    backend_factory.cpp
)

# Add backend subdirectories
add_subdirectory(cuda)
add_subdirectory(opencl)

# Create the main library
add_library(ebpf_gpu_processor STATIC ${COMMON_SOURCES})

# Set target properties
set_target_properties(ebpf_gpu_processor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(ebpf_gpu_processor PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

# Always link both backends (implementations will have appropriate stubs when disabled)
target_link_libraries(ebpf_gpu_processor PUBLIC 
    cuda_backend
    opencl_backend
)

# Add additional backend-specific properties
if(USE_CUDA_BACKEND)
    set_target_properties(ebpf_gpu_processor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Install targets
install(TARGETS ebpf_gpu_processor
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Install header files
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
) 