#!/bin/bash

# Simple DPDK Packet Processor Build Script

set -e

echo "=== Building Simple DPDK Packet Processor ==="

# Check if DPDK is available
if ! pkg-config --exists libdpdk; then
    echo "Error: DPDK not found. Installing DPDK..."
    apt update && apt install -y dpdk dpdk-dev libdpdk-dev
fi

echo "DPDK version: $(pkg-config --modversion libdpdk)"
echo "DPDK CFLAGS: $(pkg-config --cflags libdpdk)"

# Create build directory
BUILD_DIR="build_simple"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
echo "Building..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Executable: $PWD/dpdk_example"

# Show usage
echo ""
echo "=== Usage Examples ==="
echo "1. Test with null PMD (generates packets automatically):"
echo "   sudo ./dpdk_example -c 1 --vdev=net_null0 -- -p 0"
echo ""
echo "2. Test with TAP interface:"
echo "   sudo ./dpdk_example -c 1 --vdev=net_tap0,iface=test0 -- -p 0"
echo ""
echo "3. Test with ring PMD:"
echo "   sudo ./dpdk_example -c 1 --vdev=net_ring0 -- -p 0"
echo ""
echo "Note: Run as root or with appropriate permissions for DPDK" 