#include "../include/gpu_device_manager.hpp"
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <vector>
#include <mutex>
#include <iostream>

// Forward declarations for backend-specific functions
namespace ebpf_gpu {

namespace cuda {
    int get_device_count();
    size_t get_available_memory(int device_id);
    GpuDeviceInfo query_device_info(int device_id);
}

namespace opencl {
    int get_device_count();
    size_t get_available_memory(int device_id);
    GpuDeviceInfo query_device_info(int device_id);
}

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
        // Try the other backend
        BackendType other_type = (type == BackendType::CUDA) ? 
                                  BackendType::OpenCL : BackendType::CUDA;
        device_count = get_device_count(other_type);
        if (device_count <= 0) {
            return -1;  // No devices available at all
        }
        return 0;  // Return the first device of the other backend
    }
    
    // For now, just return the first device
    // TODO: Implement more sophisticated device selection based on capabilities
    return 0;
}

size_t GpuDeviceManager::get_available_memory(int device_id) const {
    std::lock_guard<std::mutex> lock(device_mutex);
    
    // Try OpenCL backend first since we know it's available on this machine
    if (opencl_device_count_ > 0) {
        // Adjust device_id for OpenCL if needed
        int opencl_device_id = (device_id >= cuda_device_count_) ? 
                               device_id - cuda_device_count_ : device_id;
        
        if (opencl_device_id < opencl_device_count_) {
            return opencl::get_available_memory(opencl_device_id);
        }
    }
    
    // Try CUDA backend if OpenCL failed or not available
    if (cuda_device_count_ > 0 && device_id < cuda_device_count_) {
        return cuda::get_available_memory(device_id);
    }
    
    // No devices available or invalid device_id
    return 0;
}

GpuDeviceInfo GpuDeviceManager::query_device_info(int device_id) const {
    std::lock_guard<std::mutex> lock(device_mutex);
    
    // Try OpenCL backend first since we know it's available on this machine
    if (opencl_device_count_ > 0) {
        // Adjust device_id for OpenCL if needed
        int opencl_device_id = (device_id >= cuda_device_count_) ? 
                               device_id - cuda_device_count_ : device_id;
        
        if (opencl_device_id < opencl_device_count_) {
            try {
                return opencl::query_device_info(opencl_device_id);
            } catch (const std::exception& e) {
                // Log the error
                std::cerr << "OpenCL device query failed: " << e.what() << std::endl;
            }
        }
    }
    
    // Try CUDA backend if OpenCL failed or not available
    if (cuda_device_count_ > 0 && device_id < cuda_device_count_) {
        try {
            return cuda::query_device_info(device_id);
        } catch (const std::exception& e) {
            // Log the error but continue to try OpenCL
            std::cerr << "CUDA device query failed: " << e.what() << std::endl;
        }
    }
    
    // If no backends are available or all queries failed, return a stub device info
    GpuDeviceInfo info;
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
    
    // Initialize OpenCL devices first (more likely to be available)
    opencl_device_count_ = opencl::get_device_count();
    
    // Initialize CUDA devices
    cuda_device_count_ = cuda::get_device_count();
    
    // If no devices available, print a warning
    if (cuda_device_count_ == 0 && opencl_device_count_ == 0) {
        std::cerr << "Warning: No GPU devices detected for any backend" << std::endl;
    } else {
        std::cout << "Detected GPU devices: "
                  << cuda_device_count_ << " CUDA, "
                  << opencl_device_count_ << " OpenCL" << std::endl;
    }
}

} // namespace ebpf_gpu 