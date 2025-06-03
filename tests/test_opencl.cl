/**
 * Simple OpenCL test kernel for eBPF on GPU tests
 */

__kernel void simple_kernel(
    __global const unsigned char* input_ptr,
    __global unsigned int* output_ptr,
    const unsigned int length
)
{
    // Get thread ID
    int id = get_global_id(0);
    
    // Set output to 1 (simple test pass condition)
    if (id == 0) {
        output_ptr[0] = 1;
    }
} 