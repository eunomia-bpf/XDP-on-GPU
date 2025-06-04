#include "ebpf_gpu_processor_impl.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <functional>
#include <thread>

namespace ebpf_gpu {

EventProcessor::Impl::Impl(const Config& config) 
    : config_(config), context_(nullptr), device_buffer_(nullptr), buffer_size_(0), kernel_function_(nullptr), device_id_(-1), current_stream_idx_(0) {
    initialize_device();
}

EventProcessor::Impl::~Impl() {
    // Clean up multiple device buffers
    for (void* buffer : device_buffers_) {
        if (buffer) {
            cudaFree(buffer);
        }
    }
    device_buffers_.clear();
    buffer_sizes_.clear();
    
    // Clean up original device buffer
    if (device_buffer_) {
        cudaFree(device_buffer_);
    }
    
    // Clean up streams and events
    cleanup_streams();
    
    if (context_) {
        cuCtxDestroy(context_);
    }
}

void EventProcessor::Impl::initialize_device() {
    device_id_ = config_.device_id;
    
    // Validate kernel launch configuration
    if (config_.block_size <= 0 || config_.block_size > 1024) {
        throw std::runtime_error("Invalid block size: must be between 1 and 1024");
    }
    
    if (config_.max_grid_size < 0) {
        throw std::runtime_error("Invalid max grid size: must be non-negative");
    }
    
    // Check if any devices are available first
    if (device_manager_.get_device_count() == 0) {
        throw std::runtime_error("No CUDA devices available");
    }
    
    if (device_id_ < 0) {
        device_id_ = device_manager_.select_best_device();
    }
    
    if (!device_manager_.is_device_suitable(device_id_, config_.buffer_size)) {
        throw std::runtime_error("Selected device is not suitable for processing");
    }
    
    // Get device and create context (CUDA driver already initialized by device manager)
    CUdevice device;
    CUresult result = cuDeviceGet(&device, device_id_);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to get CUDA device");
    }
    
    result = cuCtxCreate(&context_, 0, device);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to create CUDA context");
    }
    
    // Set this context as current immediately after creation
    result = cuCtxSetCurrent(context_);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to set CUDA context as current");
    }
    
    // Allocate device buffer
    cudaError_t cuda_result = cudaMalloc(&device_buffer_, config_.buffer_size);
    if (cuda_result != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory");
    }
    buffer_size_ = config_.buffer_size;
    
    // Initialize CUDA streams after context is created and set
    initialize_streams();
}

ProcessingResult EventProcessor::Impl::load_kernel_from_ir(const std::string& ptx_code, const std::string& function_name) {
    try {
        if (ptx_code.empty()) {
            return ProcessingResult::InvalidInput;
        }
        if (function_name.empty()) {
            return ProcessingResult::InvalidInput;
        }
        
        KernelLoader loader;
        module_ = loader.load_from_ir(ptx_code);
        if (!module_ || !module_->is_valid()) {
            return ProcessingResult::KernelError;
        }
        
        kernel_function_ = static_cast<CUfunction>(module_->get_function(function_name));
        return ProcessingResult::Success;
    } catch (const std::invalid_argument&) {
        return ProcessingResult::InvalidInput;
    } catch (const std::runtime_error&) {
        return ProcessingResult::KernelError;
    } catch (...) {
        return ProcessingResult::Error;
    }
}

ProcessingResult EventProcessor::Impl::load_kernel_from_file(const std::string& file_path, const std::string& function_name) {
    try {
        if (file_path.empty()) {
            return ProcessingResult::InvalidInput;
        }
        if (function_name.empty()) {
            return ProcessingResult::InvalidInput;
        }
        
        KernelLoader loader;
        module_ = loader.load_from_file(file_path);
        if (!module_ || !module_->is_valid()) {
            return ProcessingResult::KernelError;
        }
        
        kernel_function_ = static_cast<CUfunction>(module_->get_function(function_name));
        return ProcessingResult::Success;
    } catch (const std::invalid_argument&) {
        return ProcessingResult::InvalidInput;
    } catch (const std::runtime_error&) {
        return ProcessingResult::KernelError;
    } catch (...) {
        return ProcessingResult::Error;
    }
}

ProcessingResult EventProcessor::Impl::load_kernel_from_source(const std::string& cuda_source, const std::string& function_name,
                                                  const std::vector<std::string>& include_paths,
                                                  const std::vector<std::string>& compile_options) {
    try {
        if (cuda_source.empty()) {
            return ProcessingResult::InvalidInput;
        }
        if (function_name.empty()) {
            return ProcessingResult::InvalidInput;
        }
        
        KernelLoader loader;
        module_ = loader.load_from_source(cuda_source, include_paths, compile_options);
        if (!module_ || !module_->is_valid()) {
            return ProcessingResult::KernelError;
        }
        
        kernel_function_ = static_cast<CUfunction>(module_->get_function(function_name));
        return ProcessingResult::Success;
    } catch (const std::invalid_argument&) {
        return ProcessingResult::InvalidInput;
    } catch (const std::runtime_error&) {
        return ProcessingResult::KernelError;
    } catch (...) {
        return ProcessingResult::Error;
    }
}

ProcessingResult EventProcessor::Impl::process_event(void* event_data, size_t event_size) {
    if (!is_ready()) {
        return ProcessingResult::KernelError;
    }
    
    if (!event_data || event_size == 0) {
        return ProcessingResult::InvalidInput;
    }
    
    if (ensure_buffer_size(event_size) != ProcessingResult::Success) {
        return ProcessingResult::DeviceError;
    }
    
    // Use the new method to ensure context is current
    ProcessingResult ctx_result = ensure_context_current();
    if (ctx_result != ProcessingResult::Success) {
        return ProcessingResult::DeviceError;
    }
    
    cudaError_t result;
    
    // Direct copy from pageable memory to device
    result = cudaMemcpy(device_buffer_, event_data, event_size, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        return ProcessingResult::DeviceError;
    }
    
    // Launch kernel for single event
    ProcessingResult launch_result = launch_kernel(device_buffer_, 1);
    if (launch_result != ProcessingResult::Success) {
        return launch_result;
    }
    
    // Direct copy from device to user buffer
    result = cudaMemcpy(event_data, device_buffer_, event_size, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        return ProcessingResult::DeviceError;
    }
    
    // Ensure all operations complete
    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        return ProcessingResult::DeviceError;
    }
    
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::Impl::process_events(void* events_buffer, size_t buffer_size, size_t event_count,
                                                   bool is_async) {
    if (!is_ready()) {
        return ProcessingResult::KernelError;
    }
    
    if (!events_buffer || buffer_size == 0 || event_count == 0) {
        return ProcessingResult::InvalidInput;
    }
    
    // Ensure the CUDA context is current
    ProcessingResult ctx_result = ensure_context_current();
    if (ctx_result != ProcessingResult::Success) {
        return ProcessingResult::DeviceError;
    }
    
    // Determine the optimal batch size for processing
    // If max_batch_size is 0 or greater than event_count, use event_count as batch_size
    size_t batch_size = (config_.max_batch_size > 0 && config_.max_batch_size < event_count) 
                      ? config_.max_batch_size : event_count;
    
    // Calculate the number of batches
    size_t num_batches = (event_count + batch_size - 1) / batch_size; // Ceiling division
    
    // For a single small batch, optimize by avoiding overhead
    if (num_batches == 1) {
        return process_events_single_batch(events_buffer, buffer_size, event_count, is_async);
    }
    else {
        // Use the optimized multi-batch processing with pipelining
        return process_events_multi_batch_pipelined(events_buffer, buffer_size, event_count, is_async);
    }
    
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::Impl::process_events_single_batch(void* events_buffer, size_t buffer_size, 
                                                                  size_t event_count, bool is_async) {
    // Process all events in one go for maximum performance
    // Ensure the device buffer is large enough
    ProcessingResult buffer_result = ensure_buffer_size(buffer_size);
    if (buffer_result != ProcessingResult::Success) {
        return ProcessingResult::DeviceError;
    }
    
    // Get a stream for this operation
    cudaStream_t stream = get_available_stream();
    
    // Check if we can use zero-copy
    bool use_zero_copy = config_.use_zero_copy;
    void* device_ptr = device_buffer_;
    
    if (use_zero_copy) {
        // Try to get the device pointer for the host memory
        cudaError_t result = cudaHostGetDevicePointer(&device_ptr, events_buffer, 0);
        if (result != cudaSuccess) {
            // If we can't get the device pointer, fall back to normal copy
            use_zero_copy = false;
        }
    }
    
    cudaError_t result;
    if (!use_zero_copy && !config_.use_unified_memory) {
        // Transfer data to device asynchronously
        result = cudaMemcpyAsync(device_buffer_, events_buffer, buffer_size, 
                               cudaMemcpyHostToDevice, stream);
        if (result != cudaSuccess) {
            return ProcessingResult::DeviceError;
        }
    }
    
    // Launch kernel with all events
    void* kernel_args[] = {
        use_zero_copy ? &device_ptr : &device_buffer_,
        &event_count
    };
    
    // Calculate grid and block dimensions using config values
    int block_size = config_.block_size;
    int grid_size = (event_count + block_size - 1) / block_size;
    
    // Apply max grid size limit if configured
    if (config_.max_grid_size > 0 && grid_size > config_.max_grid_size) {
        grid_size = config_.max_grid_size;
    }
    
    // Launch kernel asynchronously using the CUDA driver API
    CUresult cu_result = cuLaunchKernel(
        kernel_function_,
        grid_size, 1, 1,    // Grid dimensions
        block_size, 1, 1,   // Block dimensions
        config_.shared_memory_size,  // Shared memory (configurable)
        stream,             // Stream
        kernel_args,        // Parameters
        nullptr             // Extra
    );
    
    if (cu_result != CUDA_SUCCESS) {
        return ProcessingResult::KernelError;
    }
    
    // Only copy results back if we didn't use zero-copy or unified memory
    if (!use_zero_copy && !config_.use_unified_memory) {
        // Copy results back to host asynchronously
        result = cudaMemcpyAsync(events_buffer, device_buffer_, buffer_size,
                               cudaMemcpyDeviceToHost, stream);
        if (result != cudaSuccess) {
            return ProcessingResult::DeviceError;
        }
    }
    
    // For synchronous operation, wait for everything to complete
    if (!is_async) {
        result = cudaStreamSynchronize(stream);
        if (result != cudaSuccess) {
            return ProcessingResult::DeviceError;
        }
    }
    
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::Impl::process_events_multi_batch_pipelined(void* events_buffer, size_t buffer_size, 
                                                                           size_t event_count, bool is_async) {
    // Calculate batch parameters
    size_t batch_size = (config_.max_batch_size > 0 && config_.max_batch_size < event_count) 
                      ? config_.max_batch_size : event_count;
    size_t num_batches = (event_count + batch_size - 1) / batch_size;
    size_t event_size = buffer_size / event_count;
    size_t max_batch_bytes = batch_size * event_size;
    
    // Initialize streams if not already done
    if (cuda_streams_.empty()) {
        initialize_streams();
    }
    
    size_t num_streams = cuda_streams_.size();
    if (num_streams == 0) {
        return ProcessingResult::DeviceError;
    }
    
    // Check if we can use zero-copy
    bool use_zero_copy = config_.use_zero_copy;
    void* device_ptr = nullptr;
    
    if (use_zero_copy) {
        // Try to get the device pointer for the host memory
        cudaError_t result = cudaHostGetDevicePointer(&device_ptr, events_buffer, 0);
        if (result != cudaSuccess) {
            // If we can't get the device pointer, fall back to normal copy
            use_zero_copy = false;
        }
    }
    
    // Only ensure stream buffers if we're not using zero-copy or unified memory
    if (!use_zero_copy && !config_.use_unified_memory) {
        ProcessingResult buffer_result = ensure_stream_buffers(max_batch_bytes);
        if (buffer_result != ProcessingResult::Success) {
            return buffer_result;
        }
    }
    
    // Ensure we have enough events for synchronization (3 per stream: H2D, kernel, D2H)
    size_t required_events = num_streams * 3;
    if (cuda_events_.size() < required_events) {
        // Clean up old events
        for (auto& event : cuda_events_) {
            cudaEventDestroy(event);
        }
        cuda_events_.clear();
        
        // Create new events
        cuda_events_.resize(required_events);
        for (size_t i = 0; i < required_events; i++) {
            cudaError_t result = cudaEventCreate(&cuda_events_[i]);
            if (result != cudaSuccess) {
                // Clean up partially created events
                for (size_t j = 0; j < i; j++) {
                    cudaEventDestroy(cuda_events_[j]);
                }
                cuda_events_.clear();
                return ProcessingResult::DeviceError;
            }
        }
    }
    
    // Process batches with optimized pipeline overlap
    for (size_t batch = 0; batch < num_batches; batch++) {
        size_t offset = batch * batch_size;
        size_t current_batch_events = std::min(batch_size, event_count - offset);
        size_t current_batch_bytes = current_batch_events * event_size;
        char* batch_buffer = static_cast<char*>(events_buffer) + (offset * event_size);
        
        // Get stream and device buffer for this batch using round-robin
        size_t stream_idx = batch % num_streams;
        cudaStream_t current_stream = cuda_streams_[stream_idx];
        void* current_device_buffer = use_zero_copy ? 
            static_cast<char*>(device_ptr) + (offset * event_size) : 
            device_buffers_[stream_idx];
        
        // Calculate event indices for this stream
        size_t h2d_event_idx = stream_idx * 3;
        size_t kernel_event_idx = stream_idx * 3 + 1;
        size_t d2h_event_idx = stream_idx * 3 + 2;
        
        // Wait for previous iteration on this stream to complete D2H before starting new H2D
        if (batch >= num_streams) {
            cudaError_t wait_result = cudaStreamWaitEvent(current_stream, cuda_events_[d2h_event_idx], 0);
            if (wait_result != cudaSuccess) return ProcessingResult::DeviceError;
        }
        
        if (!use_zero_copy && !config_.use_unified_memory) {
            // Stage 1: Host to Device transfer
            cudaError_t result = cudaMemcpyAsync(current_device_buffer, batch_buffer, current_batch_bytes, 
                                               cudaMemcpyHostToDevice, current_stream);
            if (result != cudaSuccess) return ProcessingResult::DeviceError;
            
            // Record H2D completion
            result = cudaEventRecord(cuda_events_[h2d_event_idx], current_stream);
            if (result != cudaSuccess) return ProcessingResult::DeviceError;
        }
        
        // Stage 2: Kernel execution
        void* kernel_args[] = { &current_device_buffer, &current_batch_events };
        int block_size = config_.block_size;
        int grid_size = (current_batch_events + block_size - 1) / block_size;
        if (config_.max_grid_size > 0 && grid_size > config_.max_grid_size) {
            grid_size = config_.max_grid_size;
        }
        
        CUresult cu_result = cuLaunchKernel(kernel_function_, grid_size, 1, 1, block_size, 1, 1,
                                          config_.shared_memory_size, current_stream, kernel_args, nullptr);
        if (cu_result != CUDA_SUCCESS) {
            return ProcessingResult::KernelError;
        }
        
        // Record kernel completion
        cudaError_t result = cudaEventRecord(cuda_events_[kernel_event_idx], current_stream);
        if (result != cudaSuccess) return ProcessingResult::DeviceError;
        
        if (!use_zero_copy && !config_.use_unified_memory) {
            // Stage 3: Device to Host transfer
            result = cudaMemcpyAsync(batch_buffer, current_device_buffer, current_batch_bytes,
                                   cudaMemcpyDeviceToHost, current_stream);
            if (result != cudaSuccess) {
                return ProcessingResult::DeviceError;
            }
            
            // Record D2H completion
            result = cudaEventRecord(cuda_events_[d2h_event_idx], current_stream);
            if (result != cudaSuccess) return ProcessingResult::DeviceError;
        }
    }
    
    // Final synchronization based on async flag
    if (!is_async) {
        // Wait for all streams to complete
        for (auto& stream : cuda_streams_) {
            cudaError_t result = cudaStreamSynchronize(stream);
            if (result != cudaSuccess) return ProcessingResult::DeviceError;
        }
    }
    
    return ProcessingResult::Success;
}

GpuDeviceInfo EventProcessor::Impl::get_device_info() const {
    if (!context_) {
        throw std::runtime_error("Device not initialized");
    }
    
    return device_manager_.get_device_info(device_id_);
}

size_t EventProcessor::Impl::get_available_memory() const {
    if (!context_) {
        return 0;
    }
    
    return device_manager_.get_available_memory(device_id_);
}

bool EventProcessor::Impl::is_ready() const {
    return context_ && device_buffer_ && module_ && kernel_function_;
}

ProcessingResult EventProcessor::Impl::ensure_buffer_size(size_t required_size) {
    if (!device_buffer_ || buffer_size_ < required_size) {
        if (device_buffer_) {
            cudaFree(device_buffer_);
        }
        
        size_t new_size = std::max(required_size, config_.buffer_size);
        cudaError_t result;
        
        if (config_.use_unified_memory) {
            result = cudaMallocManaged(&device_buffer_, new_size);
        } else {
            result = cudaMalloc(&device_buffer_, new_size);
        }
        
        if (result != cudaSuccess) {
            device_buffer_ = nullptr;
            buffer_size_ = 0;
            return ProcessingResult::MemoryError;
        }
        buffer_size_ = new_size;
    }
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::Impl::launch_kernel(void* device_data, size_t event_count) {
    if (!kernel_function_) {
        return ProcessingResult::KernelError;
    }
    
    // Set up kernel parameters
    void* args[] = {
        &device_data,
        &event_count
    };
    
    // Calculate grid and block dimensions using config values
    int block_size = config_.block_size;
    int grid_size = (event_count + block_size - 1) / block_size;
    
    // Apply max grid size limit if configured
    if (config_.max_grid_size > 0 && grid_size > config_.max_grid_size) {
        grid_size = config_.max_grid_size;
    }
    
    // Launch kernel
    CUresult result = cuLaunchKernel(
        kernel_function_,
        grid_size, 1, 1,    // Grid dimensions
        block_size, 1, 1,   // Block dimensions
        config_.shared_memory_size,  // Shared memory (configurable)
        0,                  // Stream
        args,               // Parameters
        nullptr             // Extra
    );
    
    if (result != CUDA_SUCCESS) {
        return ProcessingResult::KernelError;
    }
    return ProcessingResult::Success;
}

// EventProcessor public interface implementation
EventProcessor::EventProcessor(const Config& config) 
    : pimpl_(std::make_unique<Impl>(config)) {}

EventProcessor::~EventProcessor() = default;

EventProcessor::EventProcessor(EventProcessor&&) noexcept = default;
EventProcessor& EventProcessor::operator=(EventProcessor&&) noexcept = default;

ProcessingResult EventProcessor::load_kernel_from_ir(const std::string& ptx_code, const std::string& function_name) {
    return pimpl_->load_kernel_from_ir(ptx_code, function_name);
}

ProcessingResult EventProcessor::load_kernel_from_file(const std::string& file_path, const std::string& function_name) {
    return pimpl_->load_kernel_from_file(file_path, function_name);
}

ProcessingResult EventProcessor::load_kernel_from_source(const std::string& cuda_source, const std::string& function_name,
                                            const std::vector<std::string>& include_paths,
                                            const std::vector<std::string>& compile_options) {
    return pimpl_->load_kernel_from_source(cuda_source, function_name, include_paths, compile_options);
}

ProcessingResult EventProcessor::process_event(void* event_data, size_t event_size) {
    return pimpl_->process_event(event_data, event_size);
}

ProcessingResult EventProcessor::process_events(void* events_buffer, size_t buffer_size, size_t event_count,
                                            bool is_async) {
    return pimpl_->process_events(events_buffer, buffer_size, event_count, is_async);
}

GpuDeviceInfo EventProcessor::get_device_info() const {
    return pimpl_->get_device_info();
}

size_t EventProcessor::get_available_memory() const {
    return pimpl_->get_available_memory();
}

bool EventProcessor::is_ready() const {
    return pimpl_->is_ready();
}

// Initialize CUDA streams for asynchronous processing
void EventProcessor::Impl::initialize_streams() {
    // Cleanup existing streams first
    cleanup_streams();
    
    // Determine the number of streams to create from config
    int num_streams = config_.max_stream_count;
    if (num_streams <= 0) {
        // Default to 2 streams if not specified (minimum for effective pipelining)
        num_streams = 2;
    }
    
    // Ensure we don't create too many streams (practical limit)
    if (num_streams > 32) {
        num_streams = 32;
    }
    
    // Create the streams with priorities when possible
    cuda_streams_.reserve(num_streams);
    
    // Initialize device buffer vectors (actual allocation happens in process_events_multi_batch_pipelined)
    device_buffers_.clear();
    buffer_sizes_.clear();
    device_buffers_.reserve(num_streams);
    buffer_sizes_.reserve(num_streams);
    
    // Create CUDA events for pipeline synchronization
    cuda_events_.resize(2);
    for (size_t i = 0; i < cuda_events_.size(); i++) {
        cudaError_t result = cudaEventCreate(&cuda_events_[i]);
        if (result != cudaSuccess) {
            // Handle event creation failure
            cuda_events_.resize(i);  // Keep only the successfully created events
            break;
        }
    }
    
    // Try to create streams with different priorities
    // Use high priority for the first stream if supported
    int lowest_priority = 0;
    int highest_priority = 0;
    
    // Get the range of priorities supported by the device
    cudaError_t priority_result = cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);
    bool use_priorities = (priority_result == cudaSuccess);
    
    for (int i = 0; i < num_streams; i++) {
        cudaStream_t stream;
        cudaError_t result = cudaSuccess;
        
        if (use_priorities) {
            // Assign priorities: first stream gets highest priority, rest get normal priority
            int priority = (i == 0) ? highest_priority : lowest_priority;
            
            // Create stream with priority
            result = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
        }
        
        if (!use_priorities || result != cudaSuccess) {
            // Fall back to regular stream creation if priority streams fail
            result = cudaStreamCreate(&stream);
        }
        
        if (result != cudaSuccess) {
            // Log warning but continue with streams we have
            break;
        }
        
        cuda_streams_.push_back(stream);
    }
    
    // Ensure we have at least one stream
    if (cuda_streams_.empty()) {
        throw std::runtime_error("Failed to create any CUDA streams");
    }
    
    // Reset current stream index
    current_stream_idx_ = 0;
    
    // Initialize device buffers for each stream with default buffer size
    // This ensures buffers are ready when streams are first used
    ProcessingResult buffer_result = ensure_stream_buffers(config_.buffer_size);
    if (buffer_result != ProcessingResult::Success) {
        throw std::runtime_error("Failed to allocate device buffers for streams");
    }
}

// Get an available stream for a new batch of processing
cudaStream_t EventProcessor::Impl::get_available_stream() {
    // Initialize streams if they haven't been created yet
    if (cuda_streams_.empty()) {
        initialize_streams();
    }
    
    // If still empty after initialization, create at least one stream
    if (cuda_streams_.empty()) {
        cuda_streams_.resize(1);
        cudaError_t result = cudaStreamCreate(&cuda_streams_[0]);
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
        current_stream_idx_ = 0;
    }
    
    // Get the next stream in a round-robin fashion
    cudaStream_t stream = cuda_streams_[current_stream_idx_];
    
    // Update the index for the next call
    current_stream_idx_ = (current_stream_idx_ + 1) % cuda_streams_.size();
    
    return stream;
}

// Cleanup all CUDA streams
void EventProcessor::Impl::cleanup_streams() {
    // Synchronize and destroy all streams
    for (auto& stream : cuda_streams_) {
        // Make sure all operations in the stream are complete
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    
    // Clear the vector
    cuda_streams_.clear();
    
    // Destroy all CUDA events
    for (auto& event : cuda_events_) {
        cudaEventDestroy(event);
    }
    
    // Clear the events vector
    cuda_events_.clear();
    
    // Clean up device buffers associated with streams
    for (void* buffer : device_buffers_) {
        if (buffer) {
            cudaFree(buffer);
        }
    }
    device_buffers_.clear();
    buffer_sizes_.clear();
}

// Ensure we have adequate device buffers for each stream
ProcessingResult EventProcessor::Impl::ensure_stream_buffers(size_t required_buffer_size) {
    size_t num_streams = cuda_streams_.size();
    if (num_streams == 0) {
        return ProcessingResult::DeviceError;
    }
    
    // Check if we need to allocate or reallocate buffers
    if (device_buffers_.size() != num_streams) {
        // Clean up old buffers
        for (void* buffer : device_buffers_) {
            if (buffer) {
                cudaFree(buffer);
            }
        }
        device_buffers_.clear();
        buffer_sizes_.clear();
        
        // Allocate new buffers for each stream
        device_buffers_.resize(num_streams);
        buffer_sizes_.resize(num_streams);
        
        for (size_t i = 0; i < num_streams; i++) {
            cudaError_t result;
            if (config_.use_unified_memory) {
                result = cudaMallocManaged(&device_buffers_[i], required_buffer_size);
            } else {
                result = cudaMalloc(&device_buffers_[i], required_buffer_size);
            }
            
            if (result != cudaSuccess) {
                // Clean up partially allocated buffers
                for (size_t j = 0; j < i; j++) {
                    cudaFree(device_buffers_[j]);
                }
                device_buffers_.clear();
                buffer_sizes_.clear();
                return ProcessingResult::MemoryError;
            }
            buffer_sizes_[i] = required_buffer_size;
        }
    } else {
        // Ensure existing buffers are large enough
        for (size_t i = 0; i < num_streams; i++) {
            if (buffer_sizes_[i] < required_buffer_size) {
                cudaFree(device_buffers_[i]);
                
                cudaError_t result;
                if (config_.use_unified_memory) {
                    result = cudaMallocManaged(&device_buffers_[i], required_buffer_size);
                } else {
                    result = cudaMalloc(&device_buffers_[i], required_buffer_size);
                }
                
                if (result != cudaSuccess) {
                    return ProcessingResult::MemoryError;
                }
                buffer_sizes_[i] = required_buffer_size;
            }
        }
    }
    
    return ProcessingResult::Success;
}

// Implementation of the synchronize_async_operations method
ProcessingResult EventProcessor::Impl::synchronize_async_operations() {
    if (!is_ready()) {
        return ProcessingResult::KernelError;
    }
    
    // Ensure the CUDA context is current
    ProcessingResult ctx_result = ensure_context_current();
    if (ctx_result != ProcessingResult::Success) {
        return ProcessingResult::DeviceError;
    }
    
    // Synchronize all streams
    for (auto& stream : cuda_streams_) {
        cudaError_t result = cudaStreamSynchronize(stream);
        if (result != cudaSuccess) {
            return ProcessingResult::DeviceError;
        }
    }
    
    // Also do a device synchronize to catch any operations not tied to our streams
    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        return ProcessingResult::DeviceError;
    }
    
    return ProcessingResult::Success;
}

// Utility functions for pinned memory management
void* EventProcessor::allocate_pinned_buffer(size_t size) {
    void* pinned_ptr = nullptr;
    cudaError_t result = cudaMallocHost(&pinned_ptr, size);
    if (result != cudaSuccess) {
        return nullptr;
    }
    return pinned_ptr;
}

void EventProcessor::free_pinned_buffer(void* pinned_ptr) {
    if (pinned_ptr) {
        cudaFreeHost(pinned_ptr);
    }
}

// Utility functions for registering/unregistering existing host memory
ProcessingResult EventProcessor::register_host_buffer(void* ptr, size_t size, unsigned int flags) {
    if (!ptr || size == 0) {
        return ProcessingResult::InvalidInput;
    }
    cudaError_t result = cudaHostRegister(ptr, size, flags);
    if (result != cudaSuccess) {
        return ProcessingResult::MemoryError;
    }
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::unregister_host_buffer(void* ptr) {
    if (!ptr) {
        return ProcessingResult::InvalidInput;
    }
    cudaError_t result = cudaHostUnregister(ptr);
    if (result != cudaSuccess) {
        return ProcessingResult::MemoryError;
    }
    return ProcessingResult::Success;
}

// Global utility functions
std::vector<GpuDeviceInfo> get_available_devices() {
    static GpuDeviceManager manager; // Use static singleton for consistent device detection
    return manager.get_all_devices();
}

int select_best_device(size_t min_memory) {
    static GpuDeviceManager manager; // Use static singleton
    return manager.select_best_device();
}

bool validate_ptx_code(const std::string& ptx_code) {
    return KernelLoader::validate_ir(ptx_code);
}

ProcessingResult EventProcessor::Impl::ensure_context_current() {
    // Only set the context if it's not already current
    CUcontext current = nullptr;
    CUresult result = cuCtxGetCurrent(&current);
    if (result != CUDA_SUCCESS) {
        return ProcessingResult::DeviceError;
    }
    
    // If the context is already set correctly, no need to set it again
    if (current == context_) {
        return ProcessingResult::Success;
    }
    
    // Set the context since it's different
    result = cuCtxSetCurrent(context_);
    if (result != CUDA_SUCCESS) {
        return ProcessingResult::DeviceError;
    }
    
    return ProcessingResult::Success;
}

// Implementation of the public synchronize_async_operations method
ProcessingResult EventProcessor::synchronize_async_operations() {
    return pimpl_->synchronize_async_operations();
}

// Helper method to get total device buffer memory usage
size_t EventProcessor::Impl::get_total_device_buffer_memory() const {
    size_t total = buffer_size_;  // Original device buffer
    for (size_t size : buffer_sizes_) {
        total += size;  // Additional per-stream buffers
    }
    return total;
}

} // namespace ebpf_gpu 