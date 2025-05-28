#pragma once

#include "error_handling.hpp"
#include "gpu_device_manager.hpp"
#include "kernel_loader.hpp"
#include <memory>
#include <vector>
#include <functional>

namespace ebpf_gpu {

// Forward declarations
struct NetworkEvent;
class EventProcessor;

// Modern C++ API
struct NetworkEvent {
    uint8_t* data = nullptr;
    uint32_t length = 0;
    uint64_t timestamp = 0;
    uint32_t src_ip = 0;
    uint32_t dst_ip = 0;
    uint16_t src_port = 0;
    uint16_t dst_port = 0;
    uint8_t protocol = 0;
    uint8_t action = 0;  // 0=drop, 1=pass, 2=redirect
};

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

    // Event processing
    ProcessingResult process_events(std::vector<NetworkEvent>& events);
    ProcessingResult process_events(NetworkEvent* events, size_t count);
    ProcessingResult process_events_async(std::vector<NetworkEvent>& events,
                                        std::function<void(ProcessingResult)> callback = {});

    // Buffer-based processing (zero-copy)
    ProcessingResult process_buffer(void* buffer, size_t buffer_size, size_t event_count);

    // Device information
    GpuDeviceInfo get_device_info() const;
    size_t get_available_memory() const;
    bool is_ready() const;

    // Performance monitoring
    struct PerformanceStats {
        uint64_t events_processed = 0;
        uint64_t total_processing_time_us = 0;
        uint64_t kernel_execution_time_us = 0;
        uint64_t memory_transfer_time_us = 0;
        double events_per_second = 0.0;
    };
    
    PerformanceStats get_performance_stats() const;
    void reset_performance_stats();

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Utility functions
std::vector<GpuDeviceInfo> get_available_devices();
int select_best_device(size_t min_memory = 0);
bool validate_ptx_code(const std::string& ptx_code);

} // namespace ebpf_gpu 