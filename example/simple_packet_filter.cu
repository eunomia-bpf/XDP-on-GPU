/**
 * Simple Packet Filter Example for eBPF on GPU
 * 
 * This example demonstrates a basic packet filtering kernel that counts
 * packets with a specific pattern.
 */

#include <stdint.h>

/**
 * Simple packet filter kernel
 * 
 * This kernel examines packet data and performs a simple filter operation:
 * - Looks for TCP packets (IP protocol 6)
 * - Checks if the destination port is 80 (HTTP)
 * 
 * @param packets     Pointer to the buffer containing packet data
 * @param packet_sizes Array of packet sizes
 * @param packet_count Number of packets to process
 * @param results     Output array to store results (1 for match, 0 for no match)
 */
extern "C" __global__ void packet_filter(
    const uint8_t* packets,
    const uint32_t* packet_sizes,
    uint32_t packet_count,
    uint32_t* results)
{
    // Get the global thread ID
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (tid >= packet_count)
        return;
    
    // Initialize result to 0 (no match)
    results[tid] = 0;
    
    // Get size of this packet
    uint32_t size = packet_sizes[tid];
    
    // Basic bounds checking - packet must be at least 34 bytes for our check
    // (14 bytes Ethernet + 20 bytes IP header)
    if (size < 34)
        return;
    
    // Get offset to the current packet in the buffer
    uint32_t offset = 0;
    for (uint32_t i = 0; i < tid; i++) {
        offset += packet_sizes[i];
    }
    
    const uint8_t* packet = packets + offset;
    
    // Check for IPv4 (Ethernet type 0x0800)
    if (packet[12] != 0x08 || packet[13] != 0x00)
        return;
    
    // IP header starts at offset 14
    const uint8_t* ip_header = packet + 14;
    
    // Check for TCP (protocol 6)
    if (ip_header[9] != 6)
        return;
    
    // Get IP header length (lower 4 bits of byte 0, multiply by 4)
    uint8_t ip_header_length = (ip_header[0] & 0x0F) * 4;
    
    // TCP header starts after IP header
    const uint8_t* tcp_header = ip_header + ip_header_length;
    
    // Make sure we have enough bytes for TCP header
    if ((uint32_t)(tcp_header - packet + 4) > size)
        return;
    
    // Check destination port (HTTP = 80 = 0x0050)
    if (tcp_header[2] == 0x00 && tcp_header[3] == 0x50) {
        // Found a match - HTTP traffic
        results[tid] = 1;
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