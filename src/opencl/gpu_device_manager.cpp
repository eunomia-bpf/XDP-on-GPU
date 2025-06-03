#include "../../include/gpu_device_manager.hpp"
#include <CL/cl.h>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace ebpf_gpu {

GpuDeviceManager::GpuDeviceManager() : initialized_(false) {
    initialize_devices();
}

void GpuDeviceManager::initialize_devices() {
    if (initialized_) return;
    
    // Get platforms
    cl_uint platform_count = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platform_count);
    if (err != CL_SUCCESS || platform_count == 0) {
        // If platform query fails, no devices available
        devices_.clear();
        initialized_ = true;
        return;
    }
    
    std::vector<cl_platform_id> platforms(platform_count);
    err = clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        devices_.clear();
        initialized_ = true;
        return;
    }
    
    // Go through all platforms and get their devices
    devices_.clear();
    
    for (cl_uint platform_index = 0; platform_index < platform_count; ++platform_index) {
        cl_platform_id platform = platforms[platform_index];
        
        // Get GPU devices on this platform
        cl_uint device_count = 0;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        
        // Skip if no GPU devices or error
        if (err != CL_SUCCESS || device_count == 0) {
            continue;
        }
        
        std::vector<cl_device_id> cl_devices(device_count);
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, cl_devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            continue;
        }
        
        // Create GpuDeviceInfo for each device
        for (cl_uint device_index = 0; device_index < device_count; ++device_index) {
            cl_device_id device = cl_devices[device_index];
            
            try {
                GpuDeviceInfo info;
                // Use a combined platform+device index as device_id to ensure uniqueness
                info.device_id = (platform_index << 16) | device_index;
                
                // Query device properties
                char device_name[256];
                err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
                if (err == CL_SUCCESS) {
                    info.name = device_name;
                } else {
                    info.name = "Unknown OpenCL Device";
                }
                
                // Query memory information
                cl_ulong global_mem_size = 0;
                err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, nullptr);
                if (err == CL_SUCCESS) {
                    info.total_memory = global_mem_size;
                } else {
                    info.total_memory = 0;
                }
                
                // Set free memory to total memory (OpenCL doesn't provide free memory info)
                info.free_memory = info.total_memory;
                
                // Query compute units (similar to CUDA multiprocessors)
                cl_uint compute_units = 0;
                err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
                if (err == CL_SUCCESS) {
                    info.multiprocessor_count = compute_units;
                } else {
                    info.multiprocessor_count = 0;
                }
                
                // Query max work group size (similar to CUDA max threads per block)
                size_t max_work_group_size = 0;
                err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr);
                if (err == CL_SUCCESS) {
                    info.max_threads_per_block = max_work_group_size;
                } else {
                    info.max_threads_per_block = 0;
                }
                
                // Calculate max threads per multiprocessor (approximate)
                info.max_threads_per_multiprocessor = info.multiprocessor_count > 0 ? 
                    info.max_threads_per_block * 2 : 0;
                
                // Set OpenCL version as compute capability
                char version[256];
                err = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, nullptr);
                if (err == CL_SUCCESS) {
                    // Parse version string - usually in format "OpenCL X.Y"
                    if (std::strstr(version, "OpenCL 3.0")) {
                        info.compute_capability_major = 3;
                        info.compute_capability_minor = 0;
                    } else if (std::strstr(version, "OpenCL 2.2")) {
                        info.compute_capability_major = 2;
                        info.compute_capability_minor = 2;
                    } else if (std::strstr(version, "OpenCL 2.1")) {
                        info.compute_capability_major = 2;
                        info.compute_capability_minor = 1;
                    } else if (std::strstr(version, "OpenCL 2.0")) {
                        info.compute_capability_major = 2;
                        info.compute_capability_minor = 0;
                    } else if (std::strstr(version, "OpenCL 1.2")) {
                        info.compute_capability_major = 1;
                        info.compute_capability_minor = 2;
                    } else if (std::strstr(version, "OpenCL 1.1")) {
                        info.compute_capability_major = 1;
                        info.compute_capability_minor = 1;
                    } else if (std::strstr(version, "OpenCL 1.0")) {
                        info.compute_capability_major = 1;
                        info.compute_capability_minor = 0;
                    } else {
                        // Default to OpenCL 1.2 if we can't parse
                        info.compute_capability_major = 1;
                        info.compute_capability_minor = 2;
                    }
                } else {
                    // Default to OpenCL 1.2 if query fails
                    info.compute_capability_major = 1;
                    info.compute_capability_minor = 2;
                }
                
                // Check if device supports shared virtual memory (like CUDA unified memory)
                cl_device_svm_capabilities svm_caps = 0;
                err = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr);
                if (err == CL_SUCCESS && (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) {
                    info.unified_addressing = true;
                } else {
                    info.unified_addressing = false;
                }
                
                // Store the OpenCL device ID for later use
                // We'll use the platform_id as high 32 bits and device_id as low 32 bits
                // This is for internal use only, not exposed in the API
                
                // Add the device to the list
                devices_.push_back(info);
            } catch (const std::exception& e) {
                // Skip devices that can't be queried
                continue;
            }
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
        throw std::runtime_error("No OpenCL devices available");
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
    
    // Check minimum OpenCL version (OpenCL 1.2 is roughly equivalent to CUDA 3.0)
    if (device.compute_capability_major < 1 || 
        (device.compute_capability_major == 1 && device.compute_capability_minor < 2)) {
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
    
    // OpenCL doesn't have a direct way to query free memory
    // Just return the total memory as the available memory
    return devices_[device_id].total_memory;
}

size_t GpuDeviceManager::get_total_memory(int device_id) const {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        return 0;
    }
    return devices_[device_id].total_memory;
}

GpuDeviceInfo GpuDeviceManager::query_device_info(int device_id) const {
    // This function is not used in the OpenCL implementation
    // It's only here to satisfy the interface
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        throw std::out_of_range("Invalid device ID: " + std::to_string(device_id));
    }
    return devices_[device_id];
}

} // namespace ebpf_gpu 