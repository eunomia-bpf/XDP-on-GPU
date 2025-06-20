# Simple DPDK Packet Processor Example

This is a minimal DPDK example that demonstrates packet processing using virtual devices. **No GPU or physical NICs required!**

## Features

- Works with DPDK virtual devices (null PMD, TAP, ring)
- Simple packet capture and processing
- **NEW: Hash-based load balancing with GPU acceleration**
- **NEW: Performance benchmarking mode**
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
- `--benchmark`: Enable benchmark mode
- `--duration=SEC`: Benchmark duration in seconds (default: 10)
- `--verbose`: Enable verbose output
- `--help`: Display help message

### Available Kernel Functions

#### 1. packet_filter
Basic TCP/HTTP packet filtering
```bash
build/example/dpdk_example --vdev=net_null0 -l 0 -- --kernel=examples/simple_packet_filter.cu --function=packet_filter
```

#### 2. hash_load_balancer ⭐ NEW
Hash-based load balancing across 8 workers using FNV-1a hash algorithm
```bash
build/example/dpdk_example --vdev=net_null0 -l 0 -- --kernel=examples/simple_packet_filter.cu --function=hash_load_balancer --benchmark
```

#### 3. batch_hash_load_balancer ⭐ NEW  
Optimized batch processing version of hash load balancing
```bash
./dpdk_example --vdev=net_null0 -l 0 -- --kernel=examples/simple_packet_filter.cu --function=batch_hash_load_balancer --benchmark --duration=30
```

#### 4. packet_counter
Count packets by size categories
```bash
./dpdk_example --vdev=net_null0 -l 0 -- --kernel=examples/simple_packet_filter.cu --function=packet_counter
```

## Hash Load Balancing Features ⭐

### What it does:
- Uses FNV-1a hash algorithm to distribute packets across 8 workers
- Ensures even load distribution for better parallel processing
- Provides detailed statistics on load balancing quality
- Supports both single and batch processing modes

### Performance Metrics:
- **Packet throughput** (packets/second)
- **Worker distribution** (percentage per worker)
- **Load balance quality** (coefficient of variation)
- **Processing efficiency** (processed vs dropped packets)

### Quick Test:
```bash
# Run the automated hash load balancing test suite
sudo ./example/test_hash_load_balancing.sh
```

### Example Output:
```
=== Hash Load Balancing Statistics ===
Total Processed: 1542750
Total Dropped: 1542750

Worker Distribution:
  Worker 0: 385687 packets (12.50%)
  Worker 1: 385688 packets (12.50%)
  Worker 2: 385687 packets (12.50%)
  Worker 3: 385688 packets (12.50%)
  Worker 4: 385687 packets (12.50%)
  Worker 5: 385688 packets (12.50%)
  Worker 6: 385687 packets (12.50%)
  Worker 7: 385688 packets (12.50%)

Load Balance Quality:
  Average per worker: 385687.5
  Standard deviation: 0.5
  Coefficient of variation: 0.000 (Excellent)
```

## Examples

### Basic Usage

Run with a null PMD device and process packets using the simple_packet_filter CUDA kernel:

```bash
build/example/dpdk_example --vdev=net_null0 -l 0 -- --kernel=examples/simple_packet_filter.cu --function=packet_filter
```

### Hash Load Balancing Benchmark

Test hash load balancing performance for 30 seconds:

```bash
sudo build/example/dpdk_example --vdev=net_null0 -l 0 -- \
    --kernel=examples/simple_packet_filter.cu \
    --function=hash_load_balancer \
    --benchmark \
    --duration=30 \
    --verbose
```

### Batch Processing Performance Test

Compare batch vs single processing:

```bash
# Single processing
sudo build/example/dpdk_example --vdev=net_null0 -l 0 -- \
    --kernel=examples/simple_packet_filter.cu \
    --function=hash_load_balancer \
    --benchmark --duration=15

# Batch processing  
sudo build/example/dpdk_example --vdev=net_null0 -l 0 -- \
    --kernel=examples/simple_packet_filter.cu \
    --function=batch_hash_load_balancer \
    --benchmark --duration=15
```

### CPU Mode (No GPU)

Run with a TAP interface without GPU processing:

```bash
./dpdk_example --vdev=net_tap0,iface=test0 -l 0 -- --no-gpu
```

## Available CUDA Kernels

### simple_packet_filter.cu

This example demonstrates both basic packet filtering and advanced hash load balancing:

1. `packet_filter`: Filters TCP packets with destination port 80 (HTTP)
2. `hash_load_balancer`: Distributes packets across 8 workers using FNV-1a hash
3. `batch_hash_load_balancer`: Optimized batch version of hash load balancing
4. `packet_counter`: Counts packets based on their size ranges

## Performance Tips

1. **Use benchmark mode** (`--benchmark`) for accurate performance testing
2. **Adjust batch size** (`--batch-size=N`) based on your GPU memory
3. **Use verbose mode** (`--verbose`) for detailed packet information
4. **Run as root** for DPDK hardware access
5. **Use null PMD** (`--vdev=net_null0`) for maximum packet generation rate

## Implementation Notes

The hash load balancing implementation uses:
- **FNV-1a hash algorithm** for excellent distribution properties
- **GPU-accelerated processing** for high throughput
- **Statistical analysis** for load balancing quality assessment
- **Flexible worker assignment** (configurable number of workers)

The integration with the eBPF GPU processor provides real GPU acceleration with comprehensive performance metrics and load balancing analysis. 