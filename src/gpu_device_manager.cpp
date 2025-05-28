#include "gpu_device_manager.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace ebpf_gpu {

GpuDeviceManager::GpuDeviceManager() : initialized_(false) {
    initialize_devices();
}

void GpuDeviceManager::initialize_devices() {
    if (initialized_) return;
    
    // Initialize CUDA driver first
    CUresult cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS) {
        // If driver init fails, no devices available
        devices_.clear();
        initialized_ = true;
        return;
    }
    
    int device_count = 0;
    cudaError_t result = cudaGetDeviceCount(&device_count);
    if (result != cudaSuccess) {
        // If device count query fails, no devices available
        devices_.clear();
        initialized_ = true;
        return;
    }
    
    devices_.clear();
    devices_.reserve(device_count);
    
    for (int i = 0; i < device_count; ++i) {
        try {
            devices_.push_back(query_device_info(i));
        } catch (const std::exception& e) {
            // Skip devices that can't be queried
            continue;
        }
    }
    
    initialized_ = true;
}

int GpuDeviceManager::get_device_count() const {
    return static_cast<int>(devices_.size());
}

std::vector<GpuDeviceInfo> GpuDeviceManager::get_all_devices() const {
    return devices_;
}

GpuDeviceInfo GpuDeviceManager::get_device_info(int device_id) const {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        throw std::out_of_range("Invalid device ID: " + std::to_string(device_id));
    }
    return devices_[device_id];
}

int GpuDeviceManager::select_best_device() const {
    if (devices_.empty()) {
        throw std::runtime_error("No CUDA devices available");
    }
    
    // Select device with highest compute capability and most memory
    auto best_device = std::max_element(devices_.begin(), devices_.end(),
        [](const GpuDeviceInfo& a, const GpuDeviceInfo& b) {
            // First compare compute capability
            if (a.compute_capability_major != b.compute_capability_major) {
                return a.compute_capability_major < b.compute_capability_major;
            }
            if (a.compute_capability_minor != b.compute_capability_minor) {
                return a.compute_capability_minor < b.compute_capability_minor;
            }
            // Then compare memory
            return a.total_memory < b.total_memory;
        });
    
    return best_device->device_id;
}

bool GpuDeviceManager::is_device_suitable(int device_id, size_t min_memory) const {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        return false;
    }
    
    const auto& device = devices_[device_id];
    
    // Check minimum compute capability (3.0 for basic CUDA features)
    // Tesla P40 has compute capability 6.1, so this should pass
    if (device.compute_capability_major < 3) {
        return false;
    }
    
    // Check memory requirement
    if (device.free_memory < min_memory) {
        return false;
    }
    
    return true;
}

bool GpuDeviceManager::supports_unified_memory(int device_id) const {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        return false;
    }
    return devices_[device_id].unified_addressing;
}

bool GpuDeviceManager::supports_compute_capability(int device_id, int major, int minor) const {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        return false;
    }
    
    const auto& device = devices_[device_id];
    return (device.compute_capability_major > major) ||
           (device.compute_capability_major == major && device.compute_capability_minor >= minor);
}

size_t GpuDeviceManager::get_available_memory(int device_id) const {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        return 0;
    }
    
    // Get current memory info (may have changed since initialization)
    size_t free_mem, total_mem;
    cudaSetDevice(device_id);
    cudaError_t result = cudaMemGetInfo(&free_mem, &total_mem);
    if (result != cudaSuccess) {
        return 0;
    }
    
    return free_mem;
}

size_t GpuDeviceManager::get_total_memory(int device_id) const {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        return 0;
    }
    return devices_[device_id].total_memory;
}

GpuDeviceInfo GpuDeviceManager::query_device_info(int device_id) const {
    GpuDeviceInfo info;
    info.device_id = device_id;
    
    cudaDeviceProp prop;
    cudaError_t result = cudaGetDeviceProperties(&prop, device_id);
    if (result != cudaSuccess) {
        throw std::runtime_error("Failed to get device properties");
    }
    
    info.name = prop.name;
    info.total_memory = prop.totalGlobalMem;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    info.unified_addressing = prop.unifiedAddressing;
    
    // Get current memory info
    size_t free_mem, total_mem;
    cudaSetDevice(device_id);
    result = cudaMemGetInfo(&free_mem, &total_mem);
    if (result != cudaSuccess) {
        throw std::runtime_error("Failed to get memory info");
    }
    info.free_memory = free_mem;
    
    return info;
}

} // namespace ebpf_gpu 