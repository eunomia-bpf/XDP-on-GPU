#pragma once

#include <vector>
#include <string>
#include <memory>

namespace ebpf_gpu {

struct GpuDeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    bool unified_addressing;
};

class GpuDeviceManager {
public:
    GpuDeviceManager();
    ~GpuDeviceManager() = default;

    // Non-copyable, movable
    GpuDeviceManager(const GpuDeviceManager&) = delete;
    GpuDeviceManager& operator=(const GpuDeviceManager&) = delete;
    GpuDeviceManager(GpuDeviceManager&&) = default;
    GpuDeviceManager& operator=(GpuDeviceManager&&) = default;

    // Device discovery and information
    int get_device_count() const;
    std::vector<GpuDeviceInfo> get_all_devices() const;
    GpuDeviceInfo get_device_info(int device_id) const;
    
    // Device selection
    int select_best_device() const;
    bool is_device_suitable(int device_id, size_t min_memory = 0) const;
    
    // Device capabilities
    bool supports_unified_memory(int device_id) const;
    bool supports_compute_capability(int device_id, int major, int minor) const;
    
    // Memory information
    size_t get_available_memory(int device_id) const;
    size_t get_total_memory(int device_id) const;

private:
    void initialize_devices();
    GpuDeviceInfo query_device_info(int device_id) const;
    
    std::vector<GpuDeviceInfo> devices_;
    bool initialized_;
};

} // namespace ebpf_gpu 