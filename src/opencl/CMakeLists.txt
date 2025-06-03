# OpenCL backend implementation

# Only build if OpenCL is enabled
if(USE_OPENCL_BACKEND)
    # Source files
    set(OPENCL_BACKEND_SOURCES
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
endif() 