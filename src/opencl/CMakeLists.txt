# OpenCL backend implementation

# Source files
set(OPENCL_BACKEND_SOURCES
    opencl_device_manager.cpp
)

if(USE_OPENCL_BACKEND)
    # Add OpenCL-specific files when OpenCL is enabled
    list(APPEND OPENCL_BACKEND_SOURCES
        opencl_backend.cpp
    )

    # Create a static library for the OpenCL backend
    add_library(opencl_backend OBJECT ${OPENCL_BACKEND_SOURCES})

    # Set properties
    set_target_properties(opencl_backend PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Include directories
    target_include_directories(opencl_backend PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCL_INCLUDE_DIRS}
    )

    # Compile definitions
    target_compile_definitions(opencl_backend PRIVATE
        USE_OPENCL_BACKEND
    )

    # Link libraries
    target_link_libraries(opencl_backend PRIVATE
        ${OpenCL_LIBRARIES}
    )
else()
    # Even when OpenCL is disabled, we need to compile the stub implementations
    # to provide the weak symbols
    add_library(opencl_backend OBJECT ${OPENCL_BACKEND_SOURCES})
    
    # Set properties
    set_target_properties(opencl_backend PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Include directories
    target_include_directories(opencl_backend PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
    )
endif() 