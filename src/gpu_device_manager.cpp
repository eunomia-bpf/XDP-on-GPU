#include "../include/gpu_device_manager.hpp"
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <vector>
#include <mutex>

#ifdef USE_CUDA_BACKEND
#include <cuda_runtime.h>
#include <cuda.h>
#endif

#ifdef USE_OPENCL_BACKEND
#include <CL/cl.h>
#endif

namespace ebpf_gpu {

// Global mutex for device operations
static std::mutex device_mutex;

GpuDeviceManager::GpuDeviceManager() {
    initialize_devices();
}

GpuDeviceManager::~GpuDeviceManager() {
    // Nothing to clean up
}

int GpuDeviceManager::get_device_count(BackendType type) const {
    std::lock_guard<std::mutex> lock(device_mutex);
    
    switch (type) {
        case BackendType::CUDA:
            return cuda_device_count_;
        case BackendType::OpenCL:
            return opencl_device_count_;
        default:
            return 0;
    }
}

int GpuDeviceManager::get_best_device(BackendType type) const {
    std::lock_guard<std::mutex> lock(device_mutex);
    
    int device_count = get_device_count(type);
    if (device_count <= 0) {
        // No devices available for this backend
        return -1;
    }
    
    // For now, just return the first device
    // TODO: Implement more sophisticated device selection based on capabilities
    return 0;
}

size_t GpuDeviceManager::get_available_memory(int device_id) const {
    std::lock_guard<std::mutex> lock(device_mutex);
    
#ifdef USE_CUDA_BACKEND
    if (cuda_device_count_ > 0) {
        size_t free_memory = 0;
        size_t total_memory = 0;
        
        if (device_id >= cuda_device_count_) {
            throw std::runtime_error("Invalid CUDA device ID");
        }
        
        cudaSetDevice(device_id);
        cudaMemGetInfo(&free_memory, &total_memory);
        
        return free_memory;
    }
#endif

#ifdef USE_OPENCL_BACKEND
    if (opencl_device_count_ > 0) {
        // OpenCL doesn't have a direct way to query available memory
        // Return a conservative estimate or query device-specific extensions
        return 256 * 1024 * 1024; // 256 MB as a conservative default
    }
#endif

    // No devices available or no backends enabled
    return 0;
}

GpuDeviceInfo GpuDeviceManager::query_device_info(int device_id) const {
    std::lock_guard<std::mutex> lock(device_mutex);
    GpuDeviceInfo info;
    
#ifdef USE_CUDA_BACKEND
    if (cuda_device_count_ > 0) {
        if (device_id >= cuda_device_count_) {
            throw std::runtime_error("Invalid CUDA device ID");
        }
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        
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
        
        cudaSetDevice(device_id);
        cudaMemGetInfo(&free_memory, &total_memory);
        
        info.total_memory = total_memory;
        info.available_memory = free_memory;
        
        return info;
    }
#endif

#ifdef USE_OPENCL_BACKEND
    if (opencl_device_count_ > 0) {
        // OpenCL device information
        cl_int err;
        cl_platform_id platform_id;
        cl_device_id device;
        
        // Get platform and device
        err = clGetPlatformIDs(1, &platform_id, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get OpenCL platform ID");
        }
        
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get OpenCL device ID");
        }
        
        // Get device name
        char device_name[256];
        err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get OpenCL device name");
        }
        
        // Get compute units
        cl_uint compute_units;
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get OpenCL compute units");
        }
        
        // Get clock frequency
        cl_uint clock_frequency;
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get OpenCL clock frequency");
        }
        
        // Get global memory
        cl_ulong global_memory;
        err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_memory), &global_memory, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get OpenCL global memory");
        }
        
        info.backend_type = BackendType::OpenCL;
        info.device_id = device_id;
        info.name = device_name;
        info.compute_capability_major = 0; // Not applicable for OpenCL
        info.compute_capability_minor = 0; // Not applicable for OpenCL
        info.clock_rate = clock_frequency * 1000; // Convert MHz to kHz for consistency
        info.num_cores = compute_units;
        info.total_memory = global_memory;
        info.available_memory = global_memory / 2; // Conservative estimate
        
        return info;
    }
#endif

    // If no backends are available, return a stub device info
    info.backend_type = BackendType::CUDA; // Default
    info.device_id = -1;
    info.name = "Stub Device (No GPU backends available)";
    info.compute_capability_major = 0;
    info.compute_capability_minor = 0;
    info.clock_rate = 0;
    info.num_cores = 0;
    info.total_memory = 0;
    info.available_memory = 0;
    
    return info;
}

void GpuDeviceManager::initialize_devices() {
    std::lock_guard<std::mutex> lock(device_mutex);
    
    // Initialize CUDA devices
    cuda_device_count_ = 0;
    
#ifdef USE_CUDA_BACKEND
    // Initialize CUDA driver
    CUresult cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS) {
        // Failed to initialize CUDA driver, but don't throw an exception
        // as we might still have OpenCL available
    }
    else {
        // Get CUDA device count
        cudaError_t cuda_result = cudaGetDeviceCount(&cuda_device_count_);
        if (cuda_result != cudaSuccess) {
            cuda_device_count_ = 0;
        }
    }
#endif

    // Initialize OpenCL devices
    opencl_device_count_ = 0;
    
#ifdef USE_OPENCL_BACKEND
    cl_int err;
    cl_uint num_platforms;
    
    // Get number of platforms
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err == CL_SUCCESS && num_platforms > 0) {
        std::vector<cl_platform_id> platforms(num_platforms);
        err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        
        if (err == CL_SUCCESS) {
            for (cl_uint i = 0; i < num_platforms; i++) {
                cl_uint num_devices;
                err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
                
                if (err == CL_SUCCESS) {
                    opencl_device_count_ += num_devices;
                }
            }
        }
    }
#endif
}

} // namespace ebpf_gpu 