cmake_minimum_required(VERSION 3.18)
project(dpdk_example LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build configuration
option(ENABLE_DEBUG "Enable debug build" OFF)

# Set build type
if(ENABLE_DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-DDEBUG)
else()
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find DPDK
find_package(PkgConfig REQUIRED)
pkg_check_modules(DPDK REQUIRED libdpdk)

# Additional required libraries
find_library(PTHREAD_LIB pthread REQUIRED)
find_library(DL_LIB dl REQUIRED)
find_library(NUMA_LIB numa)
find_library(RT_LIB rt REQUIRED)

# Include directories
include_directories(${DPDK_INCLUDE_DIRS})

# Add necessary definitions
add_definitions(-D_GNU_SOURCE)
add_definitions(-include rte_config.h)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3")
if(ENABLE_DEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")
endif()

# Add DPDK compile flags
add_compile_options(${DPDK_CFLAGS})

# Source files
set(SOURCES
    dpdk_example.c
    dpdk_driver.c
)

# Main application executable
add_executable(dpdk_example ${SOURCES})

# Link libraries
target_link_libraries(dpdk_example
    ${DPDK_LIBRARIES}
    ${PTHREAD_LIB}
    ${DL_LIB}
    ${RT_LIB}
)

# Link numa if available
if(NUMA_LIB)
    target_link_libraries(dpdk_example ${NUMA_LIB})
endif()

# Installation
install(TARGETS dpdk_example
    RUNTIME DESTINATION bin
)

# Custom targets for testing
add_custom_target(test-null
    COMMAND sudo $<TARGET_FILE:dpdk_example> --vdev=net_null0 -l 0
    DEPENDS dpdk_example
    COMMENT "Test with null PMD"
)

add_custom_target(test-tap
    COMMAND sudo $<TARGET_FILE:dpdk_example> --vdev=net_tap0,iface=test0 -l 0
    DEPENDS dpdk_example
    COMMENT "Test with TAP interface"
)

add_custom_target(test-ring
    COMMAND sudo $<TARGET_FILE:dpdk_example> --vdev=net_ring0 -l 0
    DEPENDS dpdk_example
    COMMENT "Test with ring PMD"
)

add_custom_target(test-multi
    COMMAND sudo $<TARGET_FILE:dpdk_example> --vdev=net_null0 --vdev=net_null1 -l 0
    DEPENDS dpdk_example
    COMMENT "Test with multiple devices"
)

# Configuration summary
message(STATUS "Build configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  ENABLE_DEBUG: ${ENABLE_DEBUG}")
message(STATUS "  DPDK_LIBRARIES: ${DPDK_LIBRARIES}")
message(STATUS "  DPDK_CFLAGS: ${DPDK_CFLAGS}") 