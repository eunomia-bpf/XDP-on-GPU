#pragma once

#include "../include/ebpf_gpu_processor.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "../include/gpu_device_manager.hpp"
#include "../include/kernel_loader.hpp"

namespace ebpf_gpu {

// Implementation class using PIMPL idiom
class EventProcessor::Impl {
public:
    explicit Impl(const Config& config);
    ~Impl();

    // Non-copyable, movable
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = default;
    Impl& operator=(Impl&&) = default;

    ProcessingResult load_kernel_from_ir(const std::string& ptx_code, const std::string& function_name);
    ProcessingResult load_kernel_from_file(const std::string& file_path, const std::string& function_name);
    ProcessingResult load_kernel_from_source(const std::string& cuda_source, const std::string& function_name,
                                const std::vector<std::string>& include_paths,
                                const std::vector<std::string>& compile_options);

    ProcessingResult process_event(void* event_data, size_t event_size);
    ProcessingResult process_events(void* events_buffer, size_t buffer_size, size_t event_count,
                                   bool is_async = false);
    
    // Synchronization for async operations
    ProcessingResult synchronize_async_operations();

    GpuDeviceInfo get_device_info() const;
    size_t get_available_memory() const;
    bool is_ready() const;

private:
    Config config_;
    CUcontext context_;
    void* device_buffer_;
    size_t buffer_size_;
    std::unique_ptr<GpuModule> module_;
    CUfunction kernel_function_;
    GpuDeviceManager device_manager_;
    int device_id_;
    
    // Async processing support with multiple device buffers for better overlap
    std::vector<cudaStream_t> cuda_streams_;
    std::vector<cudaEvent_t> cuda_events_;  // Events for pipeline synchronization
    std::vector<void*> device_buffers_;     // Multiple device buffers, one per stream
    std::vector<size_t> buffer_sizes_;      // Size of each device buffer
    size_t current_stream_idx_;  // Instance variable for round-robin stream selection
    
    void initialize_device();
    ProcessingResult ensure_buffer_size(size_t required_size);
    ProcessingResult launch_kernel(void* device_data, size_t event_count);
    
    // Async processing methods
    void initialize_streams();
    void cleanup_streams();
    cudaStream_t get_available_stream();
    
    // Device buffer management
    ProcessingResult ensure_stream_buffers(size_t required_buffer_size);
    
    // Context management
    ProcessingResult ensure_context_current();
    
    // Fast path processing for single batch optimization
    ProcessingResult process_events_single_batch(void* events_buffer, size_t buffer_size, 
                                                size_t event_count, bool is_async);
    
    // Optimized multi-batch processing with pipelined execution
    ProcessingResult process_events_multi_batch_pipelined(void* events_buffer, size_t buffer_size, 
                                                         size_t event_count, bool is_async);
                                                         
    // Helper method to get total device buffer memory usage
    size_t get_total_device_buffer_memory() const;
};

} // namespace ebpf_gpu 