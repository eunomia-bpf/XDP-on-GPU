/**
 * Simple OpenCL test kernel for eBPF on GPU tests
 * 
 * This kernel processes NetworkEvent structures which have the following layout:
 * offset 0:  uint8_t* data (8 bytes) - ignored in testing
 * offset 8:  uint32_t length (4 bytes)
 * offset 16: uint64_t timestamp (8 bytes)
 * offset 24: uint32_t src_ip (4 bytes)
 * offset 28: uint32_t dst_ip (4 bytes)
 * offset 32: uint16_t src_port (2 bytes)
 * offset 34: uint16_t dst_port (2 bytes)
 * offset 36: uint8_t protocol (1 byte)
 * offset 37: uint8_t action (1 byte) - output field
 */

__kernel void simple_kernel(
    __global unsigned char* input_ptr,  // Changed to non-const to allow writing
    __global unsigned int* output_ptr,
    const unsigned int length
)
{
    // Get thread ID
    int id = get_global_id(0);
    
    // Only process valid threads
    if (id < length) {
        // Get a pointer to the current NetworkEvent
        __global unsigned char* event = input_ptr + (id * 40); // NetworkEvent is 40 bytes
        
        // Extract src_ip at offset 24
        uint src_ip = 0;
        if (id * 40 + 27 < length * 40) { // Ensure we're within bounds
            src_ip = (uint)event[24] | 
                    ((uint)event[25] << 8) | 
                    ((uint)event[26] << 16) | 
                    ((uint)event[27] << 24);
        }
        
        // Extract protocol at offset 36
        uchar protocol = 0;
        if (id * 40 + 36 < length * 40) {
            protocol = event[36];
        }
        
        // Simple filtering logic matching the CPU implementation
        uint result = 0;
        if (src_ip == 0xC0A80001) { // 192.168.0.1
            result = 0; // DROP
        } else if (protocol == 6) { // TCP
            result = 1; // PASS
        } else if (protocol == 17) { // UDP
            result = 1; // PASS
        } else {
            result = 0; // DROP unknown protocols
        }
        
        // Store the result in the output buffer
        output_ptr[id] = result;
        
        // Add a memory barrier to ensure visibility of writes
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        // Write the result directly back to the action field in the input buffer
        if (id * 40 + 37 < length * 40) {
            event[37] = (uchar)result;
        }
    }
}

/**
 * Hash load balancing OpenCL kernel
 * Implements FNV-1a hash for distributing network events across workers
 */
__kernel void hash_load_balancer(
    __global unsigned char* input_ptr,
    __global unsigned int* output_ptr,
    const unsigned int length,
    const unsigned int num_workers
)
{
    // Get thread ID
    int id = get_global_id(0);
    
    // Only process valid threads
    if (id < length) {
        // Get a pointer to the current NetworkEvent
        __global unsigned char* event = input_ptr + (id * 40); // NetworkEvent is 40 bytes
        
        // Extract network fields for hashing
        uint src_ip = 0, dst_ip = 0;
        ushort src_port = 0, dst_port = 0;
        
        if (id * 40 + 31 < length * 40) { // Ensure we're within bounds
            // Extract src_ip at offset 24
            src_ip = (uint)event[24] | 
                    ((uint)event[25] << 8) | 
                    ((uint)event[26] << 16) | 
                    ((uint)event[27] << 24);
            
            // Extract dst_ip at offset 28
            dst_ip = (uint)event[28] | 
                    ((uint)event[29] << 8) | 
                    ((uint)event[30] << 16) | 
                    ((uint)event[31] << 24);
        }
        
        if (id * 40 + 35 < length * 40) {
            // Extract src_port at offset 32
            src_port = (ushort)event[32] | ((ushort)event[33] << 8);
            
            // Extract dst_port at offset 34
            dst_port = (ushort)event[34] | ((ushort)event[35] << 8);
        }
        
        // FNV-1a hash algorithm for load balancing
        uint hash = 2166136261U; // FNV offset basis
        const uint prime = 16777619U; // FNV prime
        
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
        
        // Assign to worker queue based on hash
        uint worker_id = hash % num_workers;
        
        // Store worker assignment in output buffer
        output_ptr[id] = worker_id;
        
        // Apply basic filtering - allow traffic only if assigned to even worker IDs
        uint result = worker_id % 256; // Store worker ID
        if (worker_id % 2 == 0) {
            result |= 0x80; // Set high bit for PASS
        }
        
        // Add a memory barrier to ensure visibility of writes
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        // Write the result directly back to the action field in the input buffer
        if (id * 40 + 37 < length * 40) {
            event[37] = (uchar)result;
        }
    }
} 