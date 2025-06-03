# CUDA backend implementation

# Source files
set(CUDA_BACKEND_SOURCES
    cuda_device_manager.cpp
)

if(USE_CUDA_BACKEND)
    # Add CUDA-specific files when CUDA is enabled
    list(APPEND CUDA_BACKEND_SOURCES
        cuda_backend.cpp
    )

    # Create a static library for the CUDA backend
    add_library(cuda_backend OBJECT ${CUDA_BACKEND_SOURCES})

    # Set properties
    set_target_properties(cuda_backend PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Include directories
    target_include_directories(cuda_backend PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
    )

    # Compile definitions
    target_compile_definitions(cuda_backend PRIVATE
        USE_CUDA_BACKEND
    )

    # Link libraries
    target_link_libraries(cuda_backend PRIVATE
        CUDA::cuda_driver
        CUDA::cudart
    )
else()
    # Even when CUDA is disabled, we need to compile the stub implementations
    # to provide the weak symbols
    add_library(cuda_backend OBJECT ${CUDA_BACKEND_SOURCES})
    
    # Set properties
    set_target_properties(cuda_backend PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Include directories
    target_include_directories(cuda_backend PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
    )
endif() 