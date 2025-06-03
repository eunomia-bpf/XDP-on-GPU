# CUDA backend implementation

# Only build if CUDA is enabled
if(USE_CUDA_BACKEND)
    # Source files
    set(CUDA_BACKEND_SOURCES
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
endif() 