/**
 * Simple Packet Filter Example for eBPF on GPU
 * 
 * This example demonstrates a basic packet filtering kernel
 * compatible with the eBPF-on-GPU library.
 */

#include <cuda_runtime.h>
#include <stdint.h>

// Network event structure - must match the one used in the library
struct NetworkEvent {
    uint8_t* data;       // Pointer to packet data
    uint32_t length;     // Packet length
    uint64_t timestamp;  // Timestamp
    uint32_t src_ip;     // Source IP
    uint32_t dst_ip;     // Destination IP
    uint16_t src_port;   // Source port
    uint16_t dst_port;   // Destination port
    uint8_t protocol;    // Protocol
    uint8_t action;      // Action (0 = DROP, 1 = PASS)
};

/**
 * Simple packet filter kernel for DPDK integration
 * 
 * This kernel processes NetworkEvent structures and looks for:
 * - TCP traffic (protocol 6)
 * - HTTP traffic (destination port 80)
 * 
 * @param events      Pointer to array of NetworkEvent structures
 * @param num_events  Number of events to process
 */
extern "C" __global__ void packet_filter(NetworkEvent* events, size_t num_events) {
    // Get thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (tid >= num_events) {
        return;
    }
    
    // Get event to process
    NetworkEvent* event = &events[tid];
    
    // Default to DROP
    event->action = 0;
    
    // Simple filtering logic
    if (event->protocol == 6) {  // TCP
        if (event->dst_port == 80) {  // HTTP
            event->action = 1;  // PASS
        }
        else if (event->dst_port == 443) {  // HTTPS
            event->action = 1;  // PASS
        }
    }
    else if (event->protocol == 17) {  // UDP
        if (event->dst_port > 1024) {  // High ports
            event->action = 1;  // PASS
        }
    }
}

/**
 * Another example: packet counter kernel
 * 
 * This kernel simply counts packets of different sizes
 * 
 * @param packets      Pointer to the buffer containing packet data
 * @param packet_sizes Array of packet sizes
 * @param packet_count Number of packets to process
 * @param size_counts  Output array to store counts for different size ranges
 * @param num_buckets  Number of size buckets
 */
extern "C" __global__ void packet_counter(
    const uint8_t* packets,
    const uint32_t* packet_sizes,
    uint32_t packet_count,
    uint32_t* size_counts,
    uint32_t num_buckets)
{
    // Get the global thread ID
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (tid >= packet_count)
        return;
    
    // Get size of this packet
    uint32_t size = packet_sizes[tid];
    
    // Determine which bucket this packet belongs to
    // For example, with 5 buckets:
    // 0: 0-127 bytes
    // 1: 128-255 bytes
    // 2: 256-511 bytes
    // 3: 512-1023 bytes
    // 4: 1024+ bytes
    uint32_t bucket;
    if (size < 128)
        bucket = 0;
    else if (size < 256)
        bucket = 1;
    else if (size < 512)
        bucket = 2;
    else if (size < 1024)
        bucket = 3;
    else
        bucket = 4;
    
    // Make sure the bucket is valid
    if (bucket < num_buckets) {
        // Atomically increment the counter for this bucket
        atomicAdd(&size_counts[bucket], 1);
    }
} 