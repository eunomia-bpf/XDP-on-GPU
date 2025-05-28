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

// Callback type for asynchronous processing results
using EventProcessingCallback = std::function<void(ProcessingResult, void*, size_t)>;

class EventProcessor {
public:
    struct Config {
        int device_id;  // -1 for auto-select
        size_t buffer_size;  // 1MB default
        bool enable_profiling;
        bool use_unified_memory;
        
        // Kernel launch configuration
        int block_size;  // CUDA block size (threads per block)
        size_t shared_memory_size;  // Shared memory per block in bytes
        int max_grid_size;  // Maximum grid size (0 for unlimited)
        
        // Async configuration
        int max_stream_count;  // Maximum number of CUDA streams (0 for default)
        
        Config() : device_id(-1), 
                  buffer_size(1024 * 1024), 
                  enable_profiling(false), 
                  use_unified_memory(false),
                  block_size(256),  // Good default for most GPUs
                  shared_memory_size(0),  // No shared memory by default
                  max_grid_size(65535),  // CUDA maximum grid dimension
                  max_stream_count(4)    // Default number of CUDA streams
        {}
    };

    explicit EventProcessor(const Config& config = Config{});
    ~EventProcessor();

    // Non-copyable, movable
    EventProcessor(const EventProcessor&) = delete;
    EventProcessor& operator=(const EventProcessor&) = delete;
    EventProcessor(EventProcessor&&) noexcept;
    EventProcessor& operator=(EventProcessor&&) noexcept;

    // Kernel management
    ProcessingResult load_kernel_from_ptx(const std::string& ptx_code, const std::string& function_name);
    ProcessingResult load_kernel_from_file(const std::string& file_path, const std::string& function_name);
    ProcessingResult load_kernel_from_source(const std::string& cuda_source, const std::string& function_name,
                                const std::vector<std::string>& include_paths = {},
                                const std::vector<std::string>& compile_options = {});

    // Event processing - simplified interface
    // Single event processing
    ProcessingResult process_event(void* event_data, size_t event_size);
    
    // Multiple events processing (zero-copy)
    ProcessingResult process_events(void* events_buffer, size_t buffer_size, size_t event_count);
    
    // Enhanced batch processing
    /**
     * @brief Process a batch of events asynchronously with a callback
     * 
     * This method processes multiple events in batch mode and executes the provided callback
     * function when processing is complete. This allows for overlapping computation with 
     * other operations in the application.
     * 
     * @param events_buffer Pointer to the buffer containing multiple events
     * @param buffer_size Total size of the buffer in bytes
     * @param event_count Number of events in the buffer
     * @param callback Function to call when processing completes with (result, data, size)
     * 
     * @return ProcessingResult::Success if the batch was successfully queued for processing,
     *         or an error code if the batch could not be queued
     */
    ProcessingResult process_batch_async(void* events_buffer, size_t buffer_size, size_t event_count,
                                        EventProcessingCallback callback);
    
    // Utility functions for pinned memory management
    static void* allocate_pinned_buffer(size_t size);
    static void free_pinned_buffer(void* pinned_ptr);

    // Utility functions for registering/unregistering existing host memory
    static ProcessingResult register_host_buffer(void* ptr, size_t size, unsigned int flags = 0 /* cudaHostRegisterDefault */);
    static ProcessingResult unregister_host_buffer(void* ptr);
    
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