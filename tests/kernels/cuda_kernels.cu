/**
 * Simple test kernels for CUDA backend
 */

extern "C" {

// Simple kernel that increments each element in the input buffer
__global__ void increment(unsigned int* buffer, unsigned int size) {
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        buffer[id] += 1;
    }
}

// Square each element in the buffer
__global__ void square(unsigned int* buffer, unsigned int size) {
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        buffer[id] = buffer[id] * buffer[id];
    }
}

// Add a constant value to each element
__global__ void add_constant(unsigned int* buffer, unsigned int value, unsigned int size) {
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        buffer[id] += value;
    }
}

// Filter events (simple eBPF-like filter)
__global__ void filter_events(unsigned int* buffer, unsigned int threshold, unsigned int size) {
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size && buffer[id] > threshold) {
        buffer[id] = 0; // Filter out values above threshold
    }
}

// Compute sum of all elements in buffer (using shared memory)
__global__ void compute_sum(unsigned int* buffer, unsigned int* result, unsigned int size) {
    extern __shared__ unsigned int shared_mem[];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int id = blockIdx.x * blockDim.x + tid;
    
    // Load data into shared memory
    shared_mem[tid] = (id < size) ? buffer[id] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        atomicAdd(result, shared_mem[0]);
    }
}

} // extern "C" 