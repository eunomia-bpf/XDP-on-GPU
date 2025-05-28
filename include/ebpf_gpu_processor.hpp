#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_device_manager.hpp"
#include "kernel_loader.hpp"
#include <memory>
#include <vector>
#include <functional>

namespace ebpf_gpu {

// Forward declarations
class EventProcessor;

enum class ProcessingResult {
    Success = 0,
    Error = -1,
    InvalidInput = -2,
    DeviceError = -3,
    KernelError = -4
};

class EventProcessor {
public:
    struct Config {
        int device_id;  // -1 for auto-select
        size_t buffer_size;  // 1MB default
        bool enable_profiling;
        bool use_unified_memory;
        
        Config() : device_id(-1), buffer_size(1024 * 1024), enable_profiling(false), use_unified_memory(false) {}
    };

    explicit EventProcessor(const Config& config = Config{});
    ~EventProcessor();

    // Non-copyable, movable
    EventProcessor(const EventProcessor&) = delete;
    EventProcessor& operator=(const EventProcessor&) = delete;
    EventProcessor(EventProcessor&&) noexcept;
    EventProcessor& operator=(EventProcessor&&) noexcept;

    // Kernel management
    void load_kernel_from_ptx(const std::string& ptx_code, const std::string& function_name);
    void load_kernel_from_file(const std::string& file_path, const std::string& function_name);
    void load_kernel_from_source(const std::string& cuda_source, const std::string& function_name,
                                const std::vector<std::string>& include_paths = {},
                                const std::vector<std::string>& compile_options = {});

    // Event processing - simplified interface
    // Single event processing
    ProcessingResult process_event(void* event_data, size_t event_size);
    
    // Multiple events processing (zero-copy)
    ProcessingResult process_events(void* events_buffer, size_t buffer_size, size_t event_count);

    // Device information
    GpuDeviceInfo get_device_info() const;
    size_t get_available_memory() const;
    bool is_ready() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Utility functions
std::vector<GpuDeviceInfo> get_available_devices();
int select_best_device(size_t min_memory = 0);
bool validate_ptx_code(const std::string& ptx_code);

} // namespace ebpf_gpu 