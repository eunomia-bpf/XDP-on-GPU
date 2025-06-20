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
 * FNV-1a hash function for load balancing
 */
__device__ uint32_t fnv1a_hash(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port, uint16_t dst_port) {
    // FNV-1a hash algorithm for load balancing
    uint32_t hash = 2166136261U; // FNV offset basis
    const uint32_t prime = 16777619U; // FNV prime
    
    // Hash source IP
    hash ^= (src_ip & 0xFF);
    hash *= prime;
    hash ^= ((src_ip >> 8) & 0xFF);
    hash *= prime;
    hash ^= ((src_ip >> 16) & 0xFF);
    hash *= prime;
    hash ^= ((src_ip >> 24) & 0xFF);
    hash *= prime;
    
    // Hash destination IP
    hash ^= (dst_ip & 0xFF);
    hash *= prime;
    hash ^= ((dst_ip >> 8) & 0xFF);
    hash *= prime;
    hash ^= ((dst_ip >> 16) & 0xFF);
    hash *= prime;
    hash ^= ((dst_ip >> 24) & 0xFF);
    hash *= prime;
    
    // Hash source port
    hash ^= (src_port & 0xFF);
    hash *= prime;
    hash ^= ((src_port >> 8) & 0xFF);
    hash *= prime;
    
    // Hash destination port
    hash ^= (dst_port & 0xFF);
    hash *= prime;
    hash ^= ((dst_port >> 8) & 0xFF);
    hash *= prime;
    
    return hash;
}

/**
 * Hash-based load balancing kernel
 * Distributes network events across workers using FNV-1a hash
 */
extern "C" __global__ void hash_load_balancer(NetworkEvent* events, size_t num_events) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_events) {
        return;
    }
    
    NetworkEvent* event = &events[idx];
    
    // Calculate hash for load balancing
    uint32_t hash = fnv1a_hash(event->src_ip, event->dst_ip, event->src_port, event->dst_port);
    
    // Assign to worker queue based on hash (use fixed number of workers)
    const uint32_t num_workers = 8; // Fixed number of workers
    uint32_t worker_id = hash % num_workers;
    
    // Store worker assignment in the action field (for demonstration)
    // In real implementation, this would route to appropriate worker queue
    event->action = worker_id % 256; // Clamp to uint8_t range
    
    // For demonstration: also apply basic filtering
    // Allow traffic only if assigned to even worker IDs
    if (worker_id % 2 == 0) {
        // Keep the worker assignment but mark as pass
        event->action = (event->action & 0x7F) | 0x80; // Set high bit for PASS
    }
    // Odd worker IDs remain as DROP (just worker assignment without high bit)
}

/**
 * Batch hash load balancing with optimized processing
 */
extern "C" __global__ void batch_hash_load_balancer(NetworkEvent* events, size_t num_events) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Fixed number of workers
    const uint32_t num_workers = 8;
    
    // Batch processing: each thread processes multiple events
    const int batch_size = 16;
    int start_idx = tid * batch_size;
    int end_idx = min(start_idx + batch_size, (int)num_events);
    
    // Process batch of events
    for (int i = start_idx; i < end_idx; i++) {
        if (i >= num_events) break;
        
        NetworkEvent* event = &events[i];
        
        // Calculate hash for load balancing
        uint32_t hash = fnv1a_hash(event->src_ip, event->dst_ip, event->src_port, event->dst_port);
        
        // Assign to worker queue based on hash
        uint32_t worker_id = hash % num_workers;
        
        // Store worker assignment in the action field
        event->action = worker_id % 256;
        
        // Apply load balancing logic - balance traffic across workers
        // Use consistent hashing for better distribution
        uint32_t balanced_worker = (hash >> 16) % num_workers;
        if (balanced_worker != worker_id) {
            // Reassign for better balance
            event->action = balanced_worker % 256;
        }
        
        // Mark as processed (set high bit)
        event->action |= 0x80;
    }
}

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