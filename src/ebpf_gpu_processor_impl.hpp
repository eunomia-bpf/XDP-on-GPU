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
    CUcontext context_;
    void* device_buffer_;
    size_t buffer_size_;
    std::unique_ptr<CudaModule> module_;
    CUfunction kernel_function_;
    GpuDeviceManager device_manager_;
    int device_id_;
    
    // Async processing support
    struct EventBatch {
        void* data;
        size_t size;
        size_t count;
        cudaStream_t stream;
        bool owns_memory;  // Whether the batch owns the data buffer
        
        // New fields for low-latency processing
        bool owns_buffer;  // Whether this batch owns its device buffer
        void* device_buffer;  // Device buffer used for this batch (may be different from device_buffer_)
        
        // Constructor with default initialization
        EventBatch() : data(nullptr), size(0), count(0), stream(nullptr),
                      owns_memory(false), owns_buffer(false), device_buffer(nullptr) {}
        
        // Constructor with parameters
        EventBatch(void* _data, size_t _size, size_t _count,
                  cudaStream_t _stream, bool _owns_memory) 
            : data(_data), size(_size), count(_count), stream(_stream),
              owns_memory(_owns_memory), owns_buffer(false), device_buffer(nullptr) {}
    };
    
    std::vector<cudaStream_t> cuda_streams_;
    size_t current_stream_idx_;  // Instance variable for round-robin stream selection
    
    void initialize_device();
    ProcessingResult ensure_buffer_size(size_t required_size);
    ProcessingResult launch_kernel(void* device_data, size_t event_count);
    
    // Async processing methods
    void initialize_streams();
    void cleanup_streams();
    cudaStream_t get_available_stream();
    
    // Context management
    ProcessingResult ensure_context_current();
    
    // Fast path processing for single batch optimization
    ProcessingResult process_events_single_batch(void* events_buffer, size_t buffer_size, 
                                                size_t event_count, bool is_async);
    
    // Optimized multi-batch processing with pipelined execution
    ProcessingResult process_events_multi_batch_pipelined(void* events_buffer, size_t buffer_size, 
                                                         size_t event_count, bool is_async);
};

} // namespace ebpf_gpu 