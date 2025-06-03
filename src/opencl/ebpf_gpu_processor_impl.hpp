#pragma once

#include "../../include/ebpf_gpu_processor.hpp"
#include <CL/cl.h>
#include <memory>
#include <vector>
#include "../../include/gpu_device_manager.hpp"
#include "../../include/kernel_loader.hpp"

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

    ProcessingResult load_kernel_from_ptx(const std::string& ptx_code, const std::string& function_name);
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
    
    // OpenCL specific members
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue command_queue_;
    
    // Memory buffers
    cl_mem device_buffer_;
    size_t buffer_size_;
    
    // Kernel
    cl_program program_;
    cl_kernel kernel_;
    
    // Device management
    GpuDeviceManager device_manager_;
    int device_id_;
    
    // Async processing support
    std::vector<cl_command_queue> command_queues_;
    std::vector<cl_mem> device_buffers_;     // Multiple device buffers, one per queue
    std::vector<size_t> buffer_sizes_;       // Size of each device buffer
    size_t current_queue_idx_;               // Round-robin queue selection
    
    void initialize_device();
    ProcessingResult ensure_buffer_size(size_t required_size);
    ProcessingResult launch_kernel(cl_mem device_data, size_t event_count);
    
    // Async processing methods
    void initialize_command_queues();
    void cleanup_command_queues();
    cl_command_queue get_available_command_queue();
    
    // Device buffer management
    ProcessingResult ensure_command_queue_buffers(size_t required_buffer_size);
    
    // Fast path processing for single batch optimization
    ProcessingResult process_events_single_batch(void* events_buffer, size_t buffer_size, 
                                                size_t event_count, bool is_async);
    
    // Optimized multi-batch processing
    ProcessingResult process_events_multi_batch(void* events_buffer, size_t buffer_size, 
                                               size_t event_count, bool is_async);
                                               
    // Helper method to get total device buffer memory usage
    size_t get_total_device_buffer_memory() const;
    
    // OpenCL specific cleanup
    void cleanup_opencl_resources();
};

} // namespace ebpf_gpu 