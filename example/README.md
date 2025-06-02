# Simple DPDK Packet Processor Example

This is a minimal DPDK example that demonstrates packet processing using virtual devices. **No GPU or physical NICs required!**

## Features

- Works with DPDK virtual devices (null PMD, TAP, ring)
- Simple packet capture and processing
- Easy to test and understand
- No hardware dependencies

## Quick Start

### 1. Install DPDK (if not already installed)

```bash
sudo apt update
sudo apt install -y dpdk dpdk-dev libdpdk-dev
```

## DPDK Packet Processing Example

The `dpdk_example.cpp` application demonstrates how to process packets with DPDK and optionally use eBPF GPU processing.

### Building

To build the DPDK example:

```bash
# First make sure DPDK is installed on your system
make dpdk_example
```

### Running

The application accepts both DPDK EAL options and application-specific options:

```bash
./dpdk_example [EAL options] -- [application options]
```

Application options:
- `--kernel=PATH`: Path to the CUDA kernel file (.cu or .ptx)
- `--function=NAME`: CUDA kernel function name to use
- `--no-gpu`: Disable GPU processing
- `--device=ID`: GPU device ID to use (-1 for auto)
- `--batch-size=SIZE`: Maximum batch size for GPU processing
- `--help`: Display help message

### Examples

Run with a null PMD device and process packets using the simple_packet_filter CUDA kernel:

```bash
build/example/dpdk_example --vdev=net_null0 -l 0 -- --kernel=examples/simple_packet_filter.cu --function=packet_filter
```

Run with a TAP interface without GPU processing:

```bash
./dpdk_example --vdev=net_tap0,iface=test0 -l 0 -- --no-gpu
```

## Available CUDA Kernels

### simple_packet_filter.cu

This example demonstrates basic packet filtering:

1. `packet_filter`: Filters TCP packets with destination port 80 (HTTP)
2. `packet_counter`: Counts packets based on their size ranges

Usage:

```bash
./dpdk_example --vdev=net_null0 -l 0 -- --kernel=examples/simple_packet_filter.cu --function=packet_filter
```

Or:

```bash
./dpdk_example --vdev=net_null0 -l 0 -- --kernel=examples/simple_packet_filter.cu --function=packet_counter
```

## Implementation Notes

The actual integration with the eBPF GPU processor is simulated in the current implementation. In a real-world scenario, you would:

1. Initialize the eBPF GPU processor with the specified kernel and function
2. Format packet data in a way that matches the kernel's expected input format
3. Call the GPU processor with batches of packets
4. Process the results from the GPU

The example code structure is designed to make it easy to extend with real GPU processing in the future. 