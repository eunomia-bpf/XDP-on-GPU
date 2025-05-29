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
    if (device_buffer_) {
        cudaFree(device_buffer_);
    }
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
}

ProcessingResult EventProcessor::Impl::load_kernel_from_ptx(const std::string& ptx_code, const std::string& function_name) {
    try {
        if (ptx_code.empty()) {
            return ProcessingResult::InvalidInput;
        }
        if (function_name.empty()) {
            return ProcessingResult::InvalidInput;
        }
        
        KernelLoader loader;
        module_ = loader.load_from_ptx(ptx_code);
        if (!module_ || !module_->is_valid()) {
            return ProcessingResult::KernelError;
        }
        
        kernel_function_ = module_->get_function(function_name);
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
        
        kernel_function_ = module_->get_function(function_name);
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
        module_ = loader.load_from_cuda_source(cuda_source, include_paths, compile_options);
        if (!module_ || !module_->is_valid()) {
            return ProcessingResult::KernelError;
        }
        
        kernel_function_ = module_->get_function(function_name);
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
    
    // Transfer data to device asynchronously
    cudaError_t result = cudaMemcpyAsync(device_buffer_, events_buffer, buffer_size, 
                           cudaMemcpyHostToDevice, stream);
    if (result != cudaSuccess) {
        return ProcessingResult::DeviceError;
    }
    
    // Launch kernel with all events
    void* kernel_args[] = {
        &device_buffer_,
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
    
    // Copy results back to host asynchronously
    result = cudaMemcpyAsync(events_buffer, device_buffer_, buffer_size,
                           cudaMemcpyDeviceToHost, stream);
    if (result != cudaSuccess) {
        return ProcessingResult::DeviceError;
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
    
    // Ensure we have adequate device memory
    ProcessingResult buffer_result = ensure_buffer_size(max_batch_bytes);
    if (buffer_result != ProcessingResult::Success) {
        return ProcessingResult::DeviceError;
    }
    
    // Initialize streams if not already done
    if (cuda_streams_.empty()) {
        initialize_streams();
    }
    
    // CRITICAL FIX: Process batches sequentially to avoid race conditions
    // Using one stream ensures no buffer conflicts
    cudaStream_t stream = get_available_stream();
    
    // Process each batch sequentially
    for (size_t batch = 0; batch < num_batches; batch++) {
        // Calculate batch parameters
        size_t offset = batch * batch_size;
        size_t current_batch_events = std::min(batch_size, event_count - offset);
        size_t current_batch_bytes = current_batch_events * event_size;
        char* batch_buffer = static_cast<char*>(events_buffer) + (offset * event_size);
        
        // Execute 3-stage pipeline: H2D -> Kernel -> D2H
        // Stage 1: Host to Device transfer
        cudaError_t result = cudaMemcpyAsync(device_buffer_, batch_buffer, current_batch_bytes, 
                                           cudaMemcpyHostToDevice, stream);
        if (result != cudaSuccess) {
            return ProcessingResult::DeviceError;
        }
        
        // Stage 2: Launch kernel
        void* kernel_args[] = {
            &device_buffer_,
            &current_batch_events
        };
        
        int block_size = config_.block_size;
        int grid_size = (current_batch_events + block_size - 1) / block_size;
        if (config_.max_grid_size > 0 && grid_size > config_.max_grid_size) {
            grid_size = config_.max_grid_size;
        }
        
        CUresult cu_result = cuLaunchKernel(
            kernel_function_,
            grid_size, 1, 1,
            block_size, 1, 1,
            config_.shared_memory_size,
            stream,
            kernel_args,
            nullptr
        );
        
        if (cu_result != CUDA_SUCCESS) {
            return ProcessingResult::KernelError;
        }
        
        // Stage 3: Device to Host transfer
        result = cudaMemcpyAsync(batch_buffer, device_buffer_, current_batch_bytes,
                               cudaMemcpyDeviceToHost, stream);
        if (result != cudaSuccess) {
            return ProcessingResult::DeviceError;
        }
        
        // Synchronize after each batch to ensure correctness
        result = cudaStreamSynchronize(stream);
        if (result != cudaSuccess) {
            return ProcessingResult::DeviceError;
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
        cudaError_t result = cudaMalloc(&device_buffer_, new_size);
        if (result != cudaSuccess) {
            device_buffer_ = nullptr;
            buffer_size_ = 0;
            return ProcessingResult::DeviceError;
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

ProcessingResult EventProcessor::load_kernel_from_ptx(const std::string& ptx_code, const std::string& function_name) {
    return pimpl_->load_kernel_from_ptx(ptx_code, function_name);
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
    
    // Determine the number of streams to create
    int num_streams = config_.max_stream_count;
    if (num_streams <= 0) {
        // Default to 4 streams if not specified
        num_streams = 4;
    }
    
    // Create the streams with priorities when possible
    cuda_streams_.resize(num_streams);
    
    // Try to create streams with different priorities
    // Use high priority for the first stream if supported
    int lowest_priority = 0;
    int highest_priority = 0;
    
    // Get the range of priorities supported by the device
    cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);
    
    for (int i = 0; i < num_streams; i++) {
        // Assign priorities: first stream gets highest priority, rest get normal priority
        int priority = (i == 0) ? highest_priority : lowest_priority;
        
        // Create stream with priority
        cudaError_t result = cudaStreamCreateWithPriority(&cuda_streams_[i], 
                                                     cudaStreamNonBlocking, priority);
        
        if (result != cudaSuccess) {
            // Fall back to regular stream creation if priority streams fail
            result = cudaStreamCreate(&cuda_streams_[i]);
            if (result != cudaSuccess) {
                // Handle stream creation failure
                cuda_streams_.resize(i);  // Keep only the successfully created streams
                break;
            }
        }
    }
}

// Get an available stream for a new batch of processing
cudaStream_t EventProcessor::Impl::get_available_stream() {
    // Handle the case where no streams have been created
    if (cuda_streams_.empty()) {
        // Initialize one stream on demand
        cuda_streams_.resize(1);
        cudaStreamCreate(&cuda_streams_[0]);
        current_stream_idx_ = 0;
        return cuda_streams_[0];
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
        // Consider mapping CUDA errors to ProcessingResult more granularly if needed
        return ProcessingResult::DeviceError; 
    }
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::unregister_host_buffer(void* ptr) {
    if (!ptr) {
        return ProcessingResult::InvalidInput;
    }
    cudaError_t result = cudaHostUnregister(ptr);
    if (result != cudaSuccess) {
        return ProcessingResult::DeviceError;
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
    return KernelLoader::validate_ptx(ptx_code);
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

} // namespace ebpf_gpu 