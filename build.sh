#!/bin/bash

# CUDA Event Processor Build Script

set -e

echo "Building CUDA Event Processor..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..

# Build the project
echo "Building..."
make -j$(nproc)

echo "Build completed successfully!"
echo ""
echo "To run the test:"
echo "  cd build && ./test_processor"
echo ""
echo "Library files:"
echo "  libcuda_event_processor.a - Static library"
echo "  test_processor - Test executable" 