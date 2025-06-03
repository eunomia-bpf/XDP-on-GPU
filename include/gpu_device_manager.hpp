#pragma once

#include "gpu_backend.hpp"
#include <string>
#include <vector>

namespace ebpf_gpu {

/**
 * @brief Manager for GPU devices
 * Handles device discovery, selection, and information retrieval
 */
class GpuDeviceManager {
public:
    GpuDeviceManager();
    ~GpuDeviceManager();
    
    /**
     * @brief Get the number of available devices for a specific backend
     * @param type Backend type (CUDA or OpenCL)
     * @return Number of devices
     */
    int get_device_count(BackendType type) const;
    
    /**
     * @brief Get the best device for a specific backend
     * @param type Backend type (CUDA or OpenCL)
     * @return Device ID, or -1 if no devices are available
     */
    int get_best_device(BackendType type) const;
    
    /**
     * @brief Get available memory on a device
     * @param device_id Device ID
     * @return Available memory in bytes
     */
    size_t get_available_memory(int device_id) const;
    
    /**
     * @brief Query information about a device
     * @param device_id Device ID
     * @return Device information
     */
    GpuDeviceInfo query_device_info(int device_id) const;

private:
    /**
     * @brief Initialize and discover GPU devices
     */
    void initialize_devices();
    
    int cuda_device_count_ = 0;
    int opencl_device_count_ = 0;
};

} // namespace ebpf_gpu 