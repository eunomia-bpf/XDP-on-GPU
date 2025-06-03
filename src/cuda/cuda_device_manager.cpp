#include "../../include/gpu_device_manager.hpp"
#include <stdexcept>

namespace ebpf_gpu {

namespace cuda {

#ifdef USE_CUDA_BACKEND
#include <cuda_runtime.h>
#include <cuda.h>

int get_device_count() {
    int device_count = 0;
    
    // Initialize CUDA driver
    CUresult cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS) {
        return 0;
    }
    
    // Get CUDA device count
    cudaError_t cuda_result = cudaGetDeviceCount(&device_count);
    if (cuda_result != cudaSuccess) {
        return 0;
    }
    
    return device_count;
}

size_t get_available_memory(int device_id) {
    size_t free_memory = 0;
    size_t total_memory = 0;
    
    cudaError_t result = cudaSetDevice(device_id);
    if (result != cudaSuccess) {
        return 0;
    }
    
    result = cudaMemGetInfo(&free_memory, &total_memory);
    if (result != cudaSuccess) {
        return 0;
    }
    
    return free_memory;
}

GpuDeviceInfo query_device_info(int device_id) {
    GpuDeviceInfo info;
    
    cudaDeviceProp prop;
    cudaError_t result = cudaGetDeviceProperties(&prop, device_id);
    if (result != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA device properties");
    }
    
    info.backend_type = BackendType::CUDA;
    info.device_id = device_id;
    info.name = prop.name;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.clock_rate = prop.clockRate;
    info.num_cores = prop.multiProcessorCount;
    
    // Get memory information
    size_t free_memory = 0;
    size_t total_memory = 0;
    
    result = cudaSetDevice(device_id);
    if (result != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device");
    }
    
    result = cudaMemGetInfo(&free_memory, &total_memory);
    if (result != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA memory info");
    }
    
    info.total_memory = total_memory;
    info.available_memory = free_memory;
    
    return info;
}

#else // Stub implementations when CUDA is not available

// Use weak attribute to allow linking even when CUDA is disabled
__attribute__((weak)) int get_device_count() {
    return 0;
}

__attribute__((weak)) size_t get_available_memory(int device_id) {
    return 0;
}

__attribute__((weak)) GpuDeviceInfo query_device_info(int device_id) {
    GpuDeviceInfo info;
    info.device_id = -1;
    info.backend_type = BackendType::CUDA;
    info.name = "CUDA support not compiled in";
    return info;
}

#endif // USE_CUDA_BACKEND

} // namespace cuda

} // namespace ebpf_gpu 