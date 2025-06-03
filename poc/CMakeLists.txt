# POC (Proof of Concept) examples

# Simple OpenCL example
add_executable(simple_opencl_example
    simple_opencl_example.cpp
)

# Link with the main library
target_link_libraries(simple_opencl_example
    ebpf_gpu_processor
)

# Set properties
set_target_properties(simple_opencl_example PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(simple_opencl_example PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Install the examples
install(TARGETS simple_opencl_example
    RUNTIME DESTINATION bin/examples
) 