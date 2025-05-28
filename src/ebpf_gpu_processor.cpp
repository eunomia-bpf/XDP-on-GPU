#include "ebpf_gpu_processor_impl.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <functional>

namespace ebpf_gpu {

EventProcessor::Impl::Impl(const Config& config) 
    : config_(config), context_(nullptr), device_buffer_(nullptr), buffer_size_(0), kernel_function_(nullptr), device_id_(-1) {
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
    
    // Set current context to ensure we're using the right device
    CUresult cu_result = cuCtxSetCurrent(context_);
    if (cu_result != CUDA_SUCCESS) {
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

ProcessingResult EventProcessor::Impl::process_events(void* events_buffer, size_t buffer_size, size_t event_count) {
    if (!is_ready()) {
        return ProcessingResult::KernelError;
    }
    
    if (!events_buffer || buffer_size == 0 || event_count == 0) {
        return ProcessingResult::InvalidInput;
    }
    
    if (ensure_buffer_size(buffer_size) != ProcessingResult::Success) {
        return ProcessingResult::DeviceError;
    }
    
    // Set current context to ensure we're using the right device
    CUresult cu_result = cuCtxSetCurrent(context_);
    if (cu_result != CUDA_SUCCESS) {
        return ProcessingResult::DeviceError;
    }
    
    cudaError_t result;
    
    // Direct copy from pageable memory to device
    result = cudaMemcpy(device_buffer_, events_buffer, buffer_size, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        return ProcessingResult::DeviceError;
    }
    
    // Launch kernel
    ProcessingResult launch_result = launch_kernel(device_buffer_, event_count);
    if (launch_result != ProcessingResult::Success) {
        return launch_result;
    }
    
    // Direct copy from device to user buffer
    result = cudaMemcpy(events_buffer, device_buffer_, buffer_size, cudaMemcpyDeviceToHost);
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
    
    // Set current context
    CUresult result = cuCtxSetCurrent(context_);
    if (result != CUDA_SUCCESS) {
        return ProcessingResult::DeviceError;
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
    result = cuLaunchKernel(
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
    
    // Synchronize
    cudaError_t cuda_result = cudaDeviceSynchronize();
    if (cuda_result != cudaSuccess) {
        return ProcessingResult::DeviceError;
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

ProcessingResult EventProcessor::process_events(void* events_buffer, size_t buffer_size, size_t event_count) {
    return pimpl_->process_events(events_buffer, buffer_size, event_count);
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

// Initialize CUDA streams for async operations
void EventProcessor::Impl::initialize_streams() {
    // Clean up any existing streams first
    cleanup_streams();
    
    int num_streams = config_.max_stream_count;
    if (num_streams <= 0) {
        num_streams = 4;  // Default to 4 streams if not specified
    }
    
    cuda_streams_.resize(num_streams);
    
    for (int i = 0; i < num_streams; i++) {
        cudaError_t result = cudaStreamCreateWithFlags(&cuda_streams_[i], cudaStreamNonBlocking);
        if (result != cudaSuccess) {
            cleanup_streams();
            throw std::runtime_error("Failed to create CUDA streams");
        }
    }
}

void EventProcessor::Impl::cleanup_streams() {
    for (auto& stream : cuda_streams_) {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }
    cuda_streams_.clear();
}

cudaStream_t EventProcessor::Impl::get_available_stream() {
    if (cuda_streams_.empty()) {
        initialize_streams();
    }
    
    // Simple round-robin selection for now
    // Could be enhanced with stream priority or load balancing
    static size_t next_stream_index = 0;
    
    if (next_stream_index >= cuda_streams_.size()) {
        next_stream_index = 0;
    }
    
    return cuda_streams_[next_stream_index++];
}

// Static callback function for batch completion
void CUDART_CB EventProcessor::Impl::batch_completion_callback(cudaStream_t stream, cudaError_t status, void* user_data) {
    EventBatch* batch = static_cast<EventBatch*>(user_data);
    
    if (batch) {
        ProcessingResult result = (status == cudaSuccess) ? 
                                  ProcessingResult::Success : 
                                  ProcessingResult::DeviceError;
                                  
        // Call the user-provided callback with the result
        if (batch->callback) {
            batch->callback(result, batch->data, batch->size);
        }
        
        // Free memory if the batch owns it
        if (batch->owns_memory && batch->data) {
            cudaFreeHost(batch->data);
        }
        
        // Delete the batch object that was dynamically allocated
        delete batch;
    }
}

ProcessingResult EventProcessor::Impl::process_batch_internal(const EventBatch& batch) {
    if (!is_ready()) {
        return ProcessingResult::KernelError;
    }
    
    if (!batch.data || batch.size == 0 || batch.count == 0) {
        return ProcessingResult::InvalidInput;
    }
    
    // Set current context to ensure we're using the right device
    CUresult cu_result = cuCtxSetCurrent(context_);
    if (cu_result != CUDA_SUCCESS) {
        return ProcessingResult::DeviceError;
    }
    
    // Create a device buffer for this specific batch if needed
    void* batch_device_buffer = nullptr;
    cudaError_t result = cudaMalloc(&batch_device_buffer, batch.size);
    if (result != cudaSuccess) {
        return ProcessingResult::DeviceError;
    }
    
    // Copy data to device asynchronously using the provided stream
    result = cudaMemcpyAsync(batch_device_buffer, batch.data, batch.size, 
                           cudaMemcpyHostToDevice, batch.stream);
    if (result != cudaSuccess) {
        cudaFree(batch_device_buffer);
        return ProcessingResult::DeviceError;
    }
    
    // Set up kernel parameters
    void* args[] = {
        &batch_device_buffer,
        (void*)&batch.count
    };
    
    // Calculate grid and block dimensions using config values
    int block_size = config_.block_size;
    int grid_size = (batch.count + block_size - 1) / block_size;
    
    // Apply max grid size limit if configured
    if (config_.max_grid_size > 0 && grid_size > config_.max_grid_size) {
        grid_size = config_.max_grid_size;
    }
    
    // Launch kernel asynchronously
    cu_result = cuLaunchKernel(
        kernel_function_,
        grid_size, 1, 1,    // Grid dimensions
        block_size, 1, 1,   // Block dimensions
        config_.shared_memory_size,  // Shared memory (configurable)
        batch.stream,       // Use the provided stream
        args,               // Parameters
        nullptr             // Extra
    );
    
    if (cu_result != CUDA_SUCCESS) {
        cudaFree(batch_device_buffer);
        return ProcessingResult::KernelError;
    }
    
    // Copy results back to host asynchronously
    result = cudaMemcpyAsync(batch.data, batch_device_buffer, batch.size,
                           cudaMemcpyDeviceToHost, batch.stream);
    if (result != cudaSuccess) {
        cudaFree(batch_device_buffer);
        return ProcessingResult::DeviceError;
    }
    
    // Schedule cleanup of device memory after all operations complete
    result = cudaStreamAddCallback(batch.stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        void* buffer = static_cast<void*>(userData);
        cudaFree(buffer);
    }, batch_device_buffer, 0);
    
    if (result != cudaSuccess) {
        // If we can't add the callback, free the memory now
        cudaFree(batch_device_buffer);
    }
    
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::Impl::process_event_async(void* events_buffer, size_t buffer_size, size_t event_count,
                                                     EventProcessingCallback callback) {
    if (!is_ready()) {
        return ProcessingResult::KernelError;
    }
    
    if (!events_buffer || buffer_size == 0 || event_count == 0) {
        return ProcessingResult::InvalidInput;
    }
    
    // Create a cuda stream for this batch
    cudaStream_t stream = get_available_stream();
    
    // Create a batch object to pass to the callback
    // This needs to be dynamically allocated since it will be accessed after this function returns
    EventBatch* batch = new EventBatch{
        events_buffer,
        buffer_size,
        event_count,
        callback,
        stream,
        false  // doesn't own the memory
    };
    
    // Process the batch
    ProcessingResult result = process_batch_internal(*batch);
    if (result != ProcessingResult::Success) {
        delete batch;
        return result;
    }
    
    // Add a callback to be executed when all operations in the stream complete
    cudaError_t cuda_result = cudaStreamAddCallback(stream, batch_completion_callback, batch, 0);
    if (cuda_result != cudaSuccess) {
        delete batch;
        return ProcessingResult::DeviceError;
    }
    
    return ProcessingResult::Success;
}

// EventProcessor public interface implementation
ProcessingResult EventProcessor::process_event_async(void* events_buffer, size_t buffer_size, size_t event_count,
                                               EventProcessingCallback callback) {
    return pimpl_->process_event_async(events_buffer, buffer_size, event_count, callback);
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

} // namespace ebpf_gpu 