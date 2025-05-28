# CUDA Event Processor

A basic proof-of-concept library for executing CUDA kernel code to process incoming network events. This library provides two main interfaces for high-performance packet processing on GPU.

## Features

- **PTX Interface**: Load and execute PTX (Parallel Thread Execution) code dynamically
- **Buffer Interface**: Process events directly from memory buffers for zero-copy scenarios
- **Simple API**: Straightforward C interface without classes or complex abstractions
- **Network Event Processing**: Built-in support for common network packet structures

## Architecture

The library consists of:
- `cuda_event_processor.h` - Main API header
- `cuda_event_processor.c` - Core implementation using CUDA Driver API
- `cuda_event_processor.cu` - Sample CUDA kernels
- `test_basic.c` - Basic test demonstrating both interfaces

## Prerequisites

- CUDA Toolkit (11.0 or later)
- CMake (3.18 or later)
- GCC/G++ compiler
- NVIDIA GPU with compute capability 7.5 or higher

## Building

```bash
cd eBPF-on-GPU
mkdir build
cd build
cmake ..
make
```

## API Reference

### Core Functions

#### Initialization
```c
int init_processor(processor_handle_t *handle, int device_id, size_t buffer_size);
int cleanup_processor(processor_handle_t *handle);
```

#### Interface 1: PTX/Kernel Loading
```c
// Load PTX code directly
int load_ptx_kernel(processor_handle_t *handle, const char *ptx_code, const char *function_name);

// Load PTX from file
int load_kernel_function(processor_handle_t *handle, const char *kernel_file, const char *function_name);
```

#### Interface 2: Event Processing
```c
// Process structured events
int process_events(processor_handle_t *handle, network_event_t *events, size_t num_events);

// Process raw buffer (zero-copy)
int process_events_buffer(processor_handle_t *handle, void *buffer, size_t buffer_size, size_t num_events);
```

#### Utility Functions
```c
int get_cuda_device_count(void);
const char* get_last_error(void);
```

### Network Event Structure

```c
typedef struct {
    uint8_t *data;      // Packet data
    uint32_t length;    // Packet length
    uint64_t timestamp; // Timestamp
    uint32_t src_ip;    // Source IP
    uint32_t dst_ip;    // Destination IP
    uint16_t src_port;  // Source port
    uint16_t dst_port;  // Destination port
    uint8_t protocol;   // Protocol (TCP=6, UDP=17, etc.)
    uint8_t action;     // Processing result (0=drop, 1=pass, 2=redirect)
} network_event_t;
```

## Usage Examples

### Basic Usage

```c
#include "cuda_event_processor.h"

int main() {
    processor_handle_t handle;
    
    // Initialize
    if (init_processor(&handle, 0, 1024 * 1024) != 0) {
        printf("Failed to initialize: %s\n", get_last_error());
        return -1;
    }
    
    // Load PTX kernel
    const char* ptx_code = "..."; // Your PTX code
    if (load_ptx_kernel(&handle, ptx_code, "my_filter") != 0) {
        printf("Failed to load kernel: %s\n", get_last_error());
        cleanup_processor(&handle);
        return -1;
    }
    
    // Create and process events
    network_event_t events[100];
    // ... populate events ...
    
    if (process_events(&handle, events, 100) != 0) {
        printf("Failed to process events: %s\n", get_last_error());
    }
    
    // Cleanup
    cleanup_processor(&handle);
    return 0;
}
```

### Loading PTX from File

```c
// Load pre-compiled PTX
if (load_kernel_function(&handle, "examples/simple_filter.ptx", "simple_packet_filter") != 0) {
    printf("Failed to load PTX file: %s\n", get_last_error());
    return -1;
}
```

### Zero-Copy Buffer Processing

```c
// Process events directly from a buffer
void *packet_buffer = get_network_buffer(); // Your buffer source
size_t buffer_size = get_buffer_size();
size_t num_packets = get_packet_count();

if (process_events_buffer(&handle, packet_buffer, buffer_size, num_packets) != 0) {
    printf("Failed to process buffer: %s\n", get_last_error());
    return -1;
}
```

## Running Tests

```bash
cd build
./test_processor
```

Expected output:
```
CUDA Event Processor - Basic Test
==================================

Found 1 CUDA device(s)

=== Testing PTX Interface ===
Events before processing:
Event 0:
  Length: 1234
  Src IP: 0xC0A80001
  ...

Events after processing:
Event 0:
  Length: 1234
  Src IP: 0xC0A80001
  Action: 0 (DROP)
  ...

=== Testing Buffer Interface ===
...

All tests completed successfully!
```

## Writing Custom Kernels

### PTX Kernel Requirements

Your PTX kernel should:
1. Accept two parameters: `events_buffer` (pointer) and `num_events` (size)
2. Use thread indexing to process events in parallel
3. Modify the `action` field of each event based on your logic

### Example PTX Kernel

```ptx
.version 7.0
.target sm_75
.address_size 64

.visible .entry my_filter(
    .param .u64 my_filter_param_0,  // events buffer
    .param .u64 my_filter_param_1   // num_events
)
{
    // Calculate thread index
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    
    // Your processing logic here
    // Access event at: events_buffer + (thread_idx * sizeof(network_event_t))
    // Modify the action field based on your criteria
    
    ret;
}
```

### CUDA C Kernel (Alternative)

You can also write kernels in CUDA C and compile to PTX:

```cuda
__global__ void my_filter(network_event_t *events, size_t num_events) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;
    
    network_event_t *event = &events[idx];
    
    // Your filtering logic
    if (event->dst_port == 80) {
        event->action = 1; // PASS
    } else {
        event->action = 0; // DROP
    }
}
```

Compile to PTX:
```bash
nvcc -ptx -arch=sm_75 my_kernel.cu -o my_kernel.ptx
```

## Performance Considerations

This is a basic proof-of-concept. For production use, consider:

1. **Memory Management**: Use pinned memory for faster transfers
2. **Streaming**: Use CUDA streams for overlapped execution
3. **Batch Processing**: Process larger batches for better GPU utilization
4. **Persistent Kernels**: Use persistent kernels to reduce launch overhead
5. **Memory Coalescing**: Optimize memory access patterns

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.