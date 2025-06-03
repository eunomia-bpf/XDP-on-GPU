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

# Add backend dependencies
if(USE_CUDA_BACKEND)
    target_link_libraries(ebpf_gpu_processor PUBLIC cuda_backend)
    set_target_properties(ebpf_gpu_processor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

if(USE_OPENCL_BACKEND)
    target_link_libraries(ebpf_gpu_processor PUBLIC opencl_backend)
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