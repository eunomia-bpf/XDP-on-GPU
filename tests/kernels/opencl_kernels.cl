/**
 * Simple test kernels for OpenCL backend
 */

// Simple kernel that increments each element in the input buffer
__kernel void increment(__global uint* buffer) {
    const uint id = get_global_id(0);
    buffer[id] += 1;
}

// Square each element in the buffer
__kernel void square(__global uint* buffer) {
    const uint id = get_global_id(0);
    buffer[id] = buffer[id] * buffer[id];
}

// Add a constant value to each element
__kernel void add_constant(__global uint* buffer, const uint value) {
    const uint id = get_global_id(0);
    buffer[id] += value;
}

// Filter events (simple eBPF-like filter)
__kernel void filter_events(__global uint* buffer, const uint threshold) {
    const uint id = get_global_id(0);
    if (buffer[id] > threshold) {
        buffer[id] = 0; // Filter out values above threshold
    }
}

// Compute sum of all elements in buffer (using local memory)
__kernel void compute_sum(__global uint* buffer, __global uint* result, 
                         __local uint* local_sum, const uint size) {
    const uint id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint group_size = get_local_size(0);
    
    // Load data into local memory
    local_sum[local_id] = (id < size) ? buffer[id] : 0;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduction in local memory
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            local_sum[local_id] += local_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result for this work group
    if (local_id == 0) {
        atomic_add(result, local_sum[0]);
    }
} 