# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the eBPF-on-GPU project - a high-performance GPU-accelerated packet processing framework that brings eBPF-style network filtering capabilities to GPUs. It supports both CUDA and OpenCL backends with automatic detection.

## Common Development Commands

### Building
```bash
# Standard build (automatically detects CUDA or OpenCL)
make

# Debug build
make BUILD_TYPE=Debug

# Build without tests
make BUILD_TESTS=OFF

# Clean build
make clean
```

### Testing
```bash
# Run all tests
make test

# Run specific test
./build/tests/test_device_manager
./build/tests/test_kernel_loader
./build/tests/test_integration
```

### Benchmarking
```bash
# Run performance benchmarks
make bench
```

### DPDK Example
```bash
# Build and run DPDK example (requires DPDK installation)
cd example
make
sudo ./build/dpdk_example -l 0-1 -n 4 -- -p 0x1
```

## Architecture Overview

The project implements a layered architecture for GPU-accelerated packet processing:

1. **EventProcessor API** (`include/ebpf_gpu_processor.hpp`): High-level C++ interface for packet processing
   - Manages GPU memory allocation and transfers
   - Executes kernels on packet batches
   - Provides both synchronous and asynchronous processing

2. **Backend Abstraction**: Unified interface supporting both CUDA and OpenCL
   - CUDA implementation in `src/cuda/`
   - OpenCL implementation in `src/opencl/`
   - Runtime backend selection based on available hardware

3. **Core Components**:
   - **GpuDeviceManager**: Handles GPU device discovery and capability queries
   - **KernelLoader**: Manages kernel compilation and loading (PTX for CUDA, OpenCL C for OpenCL)
   - **NetworkEvent**: Common packet representation structure for GPU processing

## Key Development Patterns

### GPU Kernel Development
- Kernels are written in CUDA C (`*.cu`) or OpenCL C (`*.cl`)
- Located in `example/` for reference implementations
- Must follow the kernel signature: `extern "C" __global__ void kernel_name(NetworkEvent* events, int num_events)`
- Use `NetworkEvent` structure for packet data access

### Memory Management
- The project uses RAII patterns - GPU memory is automatically managed
- EventProcessor handles all device memory allocation/deallocation
- Zero-copy transfers when possible for performance

### Error Handling
- All GPU operations return `GpuErrorCode` enum values
- Check return codes and use `getLastError()` for detailed error messages
- GPU errors are automatically converted to exceptions in C++ API

### Testing New Features
- Add unit tests to appropriate test files in `tests/`
- Use Catch2 framework for test cases
- Include both functional and performance tests
- Test kernels go in `tests/cuda_kernels.cu`

## Backend-Specific Notes

### CUDA Backend
- Requires CUDA Toolkit 11.0+
- Uses PTX for kernel loading
- Supports CUDA streams for asynchronous execution
- Automatically detects compute capability

### OpenCL Backend
- Works with any OpenCL 1.2+ implementation
- Uses runtime kernel compilation
- Platform/device selection based on performance heuristics

## Performance Considerations
- Process packets in large batches (1000+ for optimal GPU utilization)
- Use asynchronous processing for overlapped computation
- Monitor GPU memory usage - the framework reports available memory
- Kernel execution time is measured and available via `getStats()`