cmake_minimum_required(VERSION 3.18)
project(dpdk_example LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find DPDK
find_package(PkgConfig REQUIRED)
pkg_check_modules(DPDK REQUIRED libdpdk)

# Include directories
include_directories(${DPDK_INCLUDE_DIRS})

# DPDK example executable
add_executable(dpdk_example dpdk_example.c)

# Set target properties
set_target_properties(dpdk_example PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
)

# Link libraries
target_link_libraries(dpdk_example
    ${DPDK_LIBRARIES}
    pthread
    dl
    numa
)

# Add DPDK compile flags
target_compile_options(dpdk_example PRIVATE ${DPDK_CFLAGS})

# Installation
install(TARGETS dpdk_example
    RUNTIME DESTINATION bin
) 