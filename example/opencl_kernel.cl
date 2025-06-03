/**
 * Simple OpenCL kernel for the eBPF-on-GPU example
 */

// Simple kernel that increments each element by 1
__kernel void increment(__global uint* buffer) {
    const uint id = get_global_id(0);
    buffer[id] += 1;
} 