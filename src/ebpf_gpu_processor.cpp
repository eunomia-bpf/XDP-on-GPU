#include "ebpf_gpu_processor_impl.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <functional>
#include <thread>

namespace ebpf_gpu {

EventProcessor::Impl::Impl(const Config& config) 
    : config_(config), device_buffer_(nullptr), buffer_size_(0), device_id_(-1), current_stream_idx_(0) {
    initialize_device();
}

EventProcessor::Impl::~Impl() {
    // Clean up multiple device buffers
    for (void* buffer : device_buffers_) {
        if (buffer) {
            backend_->free_device_memory(buffer);
        }
    }
    device_buffers_.clear();
    buffer_sizes_.clear();
    
    // Clean up original device buffer
    if (device_buffer_) {
        backend_->free_device_memory(device_buffer_);
    }
    
    // Clean up streams
    cleanup_streams();
}

void EventProcessor::Impl::initialize_device() {
    GpuDeviceManager device_manager;
    
    // Get device count for the specified backend type
    if (device_manager.get_device_count(config_.backend_type) == 0) {
        throw std::runtime_error("No compatible GPU devices found for the selected backend");
    }
    
    // If device_id is -1, auto-select best device
    if (device_id_ < 0) {
        device_id_ = device_manager.get_best_device(config_.backend_type);
        if (device_id_ < 0) {
            throw std::runtime_error("Failed to find a suitable GPU device");
        }
    }
    
    // Verify the device has enough memory
    size_t available_memory = device_manager.get_available_memory(device_id_);
    if (available_memory < config_.buffer_size) {
        throw std::runtime_error("Device does not have enough memory");
    }
    
    // Create and initialize the backend
    backend_ = create_backend(config_.backend_type);
    if (!backend_) {
        throw std::runtime_error("Failed to create GPU backend");
    }
    
    // Initialize the selected device
    backend_->initialize_device(device_id_);
    
    // Allocate device buffer
    device_buffer_ = backend_->allocate_device_memory(config_.buffer_size);
    if (!device_buffer_) {
        throw std::runtime_error("Failed to allocate device memory");
    }
    buffer_size_ = config_.buffer_size;
    
    // Initialize streams
    initialize_streams();
}

ProcessingResult EventProcessor::Impl::load_kernel_from_ptx(const std::string& ptx_code, const std::string& function_name) {
    try {
        if (ptx_code.empty()) {
            return ProcessingResult::InvalidInput;
        }
        if (function_name.empty()) {
            return ProcessingResult::InvalidInput;
        }
        
        // For CUDA backend, we can load PTX directly
        if (backend_->get_type() == BackendType::CUDA) {
            bool success = backend_->load_kernel_from_binary(ptx_code, function_name);
            return success ? ProcessingResult::Success : ProcessingResult::KernelError;
        } else {
            // For OpenCL, we don't support direct PTX loading
            return ProcessingResult::InvalidInput;
        }
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
        
        bool success = backend_->load_kernel_from_binary(file_path, function_name);
        return success ? ProcessingResult::Success : ProcessingResult::KernelError;
    } catch (const std::invalid_argument&) {
        return ProcessingResult::InvalidInput;
    } catch (const std::runtime_error&) {
        return ProcessingResult::KernelError;
    } catch (...) {
        return ProcessingResult::Error;
    }
}

ProcessingResult EventProcessor::Impl::load_kernel_from_source(const std::string& source_code, const std::string& function_name,
                                                  const std::vector<std::string>& include_paths,
                                                  const std::vector<std::string>& compile_options) {
    try {
        if (source_code.empty()) {
            return ProcessingResult::InvalidInput;
        }
        if (function_name.empty()) {
            return ProcessingResult::InvalidInput;
        }
        
        bool success = backend_->load_kernel_from_source(source_code, function_name, include_paths, compile_options);
        return success ? ProcessingResult::Success : ProcessingResult::KernelError;
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
    
    // Copy from host to device
    backend_->copy_host_to_device(device_buffer_, event_data, event_size);
    
    // Launch kernel for single event
    ProcessingResult launch_result = launch_kernel(device_buffer_, 1);
    if (launch_result != ProcessingResult::Success) {
        return launch_result;
    }
    
    // All done
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
    
    // Get a stream for this operation
    void* stream = get_available_stream();
    
    // Check if we can use zero-copy
    bool use_zero_copy = config_.use_zero_copy;
    
    // For now, select optimized path based on batch size
    if (event_count <= config_.max_batch_size) {
        return process_events_single_batch(events_buffer, buffer_size, event_count, is_async);
    } else {
        return process_events_multi_batch_pipelined(events_buffer, buffer_size, event_count, is_async);
    }
}

ProcessingResult EventProcessor::Impl::process_events_single_batch(void* events_buffer, size_t buffer_size, 
                                                                  size_t event_count, bool is_async) {
    // Get a stream
    void* stream = get_available_stream();
    if (!stream) {
        return ProcessingResult::DeviceError;
    }
    
    // Ensure device buffer is large enough
    if (ensure_buffer_size(buffer_size) != ProcessingResult::Success) {
            return ProcessingResult::DeviceError;
        }
    
    // Copy data to device
    backend_->copy_host_to_device_async(device_buffer_, events_buffer, buffer_size, stream);
    
    // Launch kernel
    bool success = backend_->launch_kernel_async(device_buffer_, event_count, 
                                                config_.block_size, config_.shared_memory_size, 
                                                config_.max_grid_size, stream);
    if (!success) {
        return ProcessingResult::KernelError;
    }
    
    // If synchronous operation, wait for completion
    if (!is_async) {
        if (!backend_->synchronize_stream(stream)) {
            return ProcessingResult::DeviceError;
        }
    }
    
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::Impl::process_events_multi_batch_pipelined(void* events_buffer, size_t buffer_size, 
                                                                           size_t event_count, bool is_async) {
    // Ensure we have streams available
    if (streams_.empty()) {
        initialize_streams();
    }
    
    size_t num_streams = streams_.size();
    if (num_streams == 0) {
        return ProcessingResult::DeviceError;
    }
    
    // Calculate event size assuming fixed-size events
    size_t event_size = buffer_size / event_count;
    
    // Ensure that each stream has a buffer of adequate size
    ProcessingResult buffer_result = ensure_stream_buffers(buffer_size / num_streams + event_size);
        if (buffer_result != ProcessingResult::Success) {
            return buffer_result;
    }
    
    // Check if we can use zero-copy memory
    bool use_zero_copy = config_.use_zero_copy;
    void* device_ptr = events_buffer;  // Will be used if zero-copy is enabled
    
    // Calculate number of events per batch to process in parallel
    size_t events_per_batch = std::min(event_count, config_.max_batch_size);
    
    // Calculate number of batches
    size_t num_batches = (event_count + events_per_batch - 1) / events_per_batch;
    
    // Process each batch, scheduling across available streams
    for (size_t batch = 0; batch < num_batches; ++batch) {
        // Calculate start and count for this batch
        size_t offset = batch * events_per_batch;
        size_t remaining = event_count - offset;
        size_t batch_count = std::min(events_per_batch, remaining);
        size_t batch_size = batch_count * event_size;
        
        // Get stream and device buffer for this batch using round-robin
        size_t stream_idx = batch % num_streams;
        void* current_stream = streams_[stream_idx];
        void* current_device_buffer = use_zero_copy ? 
            static_cast<char*>(device_ptr) + (offset * event_size) : 
            device_buffers_[stream_idx];
        
        if (!use_zero_copy) {
            // Copy this batch's data to the device
            void* host_ptr = static_cast<char*>(events_buffer) + (offset * event_size);
            backend_->copy_host_to_device_async(current_device_buffer, host_ptr, batch_size, current_stream);
        }
        
        // Launch kernel for this batch
        bool success = backend_->launch_kernel_async(current_device_buffer, batch_count, 
                                                    config_.block_size, config_.shared_memory_size, 
                                                    config_.max_grid_size, current_stream);
        if (!success) {
            return ProcessingResult::KernelError;
        }
    }
    
    // If synchronous operation, wait for completion
    if (!is_async) {
        // Wait for all streams to complete
        for (auto& stream : streams_) {
            if (!backend_->synchronize_stream(stream)) {
                return ProcessingResult::DeviceError;
            }
        }
    }
    
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::Impl::ensure_buffer_size(size_t required_size) {
    if (!device_buffer_ || buffer_size_ < required_size) {
        if (device_buffer_) {
            backend_->free_device_memory(device_buffer_);
        }
        
        // Allocate new buffer with some extra space to avoid frequent reallocations
        size_t new_size = required_size * 2;
        device_buffer_ = backend_->allocate_device_memory(new_size);
        if (!device_buffer_) {
            buffer_size_ = 0;
            return ProcessingResult::MemoryError;
        }
        buffer_size_ = new_size;
    }
    
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::Impl::launch_kernel(void* device_data, size_t event_count) {
    if (!backend_ || !is_ready()) {
        return ProcessingResult::KernelError;
    }
    
    // Launch kernel and wait for completion
    bool success = backend_->launch_kernel(device_data, event_count, 
                                          config_.block_size, config_.shared_memory_size, 
                                          config_.max_grid_size);
    
    return success ? ProcessingResult::Success : ProcessingResult::KernelError;
}

ProcessingResult EventProcessor::Impl::synchronize_async_operations() {
    // Synchronize all streams
    for (auto& stream : streams_) {
        if (!backend_->synchronize_stream(stream)) {
            return ProcessingResult::DeviceError;
        }
    }
    
    return ProcessingResult::Success;
}

GpuDeviceInfo EventProcessor::Impl::get_device_info() const {
    if (!backend_) {
        return GpuDeviceInfo{};
    }
    return backend_->get_device_info(device_id_);
}

size_t EventProcessor::Impl::get_available_memory() const {
    return backend_ ? backend_->get_available_memory(device_id_) : 0;
}

bool EventProcessor::Impl::is_ready() const {
    return backend_ != nullptr;
}

// Initialize CUDA streams for asynchronous processing
void EventProcessor::Impl::initialize_streams() {
    // Clean up existing streams first
    cleanup_streams();
    
    // Determine the number of streams to create
    int stream_count = config_.max_stream_count;
    if (stream_count <= 0) {
        stream_count = 4; // Default to 4 streams
    }
    
    // Create streams
    streams_.resize(stream_count);
    for (int i = 0; i < stream_count; ++i) {
        streams_[i] = backend_->create_stream();
        if (!streams_[i]) {
            throw std::runtime_error("Failed to create GPU stream");
        }
    }
    
    // Allocate per-stream device buffers
    device_buffers_.resize(stream_count, nullptr);
    buffer_sizes_.resize(stream_count, 0);
    
    // Reset stream index
    current_stream_idx_ = 0;
}

// Get an available stream for a new batch of processing
void* EventProcessor::Impl::get_available_stream() {
    if (streams_.empty()) {
        return nullptr;
    }
    
    // Simple round-robin stream selection
    void* stream = streams_[current_stream_idx_];
    current_stream_idx_ = (current_stream_idx_ + 1) % streams_.size();
    return stream;
    }
    
// Cleanup all CUDA streams
void EventProcessor::Impl::cleanup_streams() {
    // Destroy all streams
    for (auto stream : streams_) {
        if (stream) {
            backend_->destroy_stream(stream);
        }
    }
    streams_.clear();
}

// Ensure we have adequate device buffers for each stream
ProcessingResult EventProcessor::Impl::ensure_stream_buffers(size_t required_buffer_size) {
    size_t num_streams = streams_.size();
    if (num_streams == 0) {
        return ProcessingResult::DeviceError;
    }
    
    // Check if we need to resize or create buffers
    bool need_resize = false;
    
    if (device_buffers_.size() != num_streams) {
        // Resize the buffer vectors to match stream count
        
        // Free existing buffers first
        for (void* buffer : device_buffers_) {
            if (buffer) {
                backend_->free_device_memory(buffer);
            }
        }
        
        // Resize vectors
        device_buffers_.resize(num_streams, nullptr);
        buffer_sizes_.resize(num_streams, 0);
        need_resize = true;
    }
    
    // Now check if existing buffers are adequate size
    if (!need_resize) {
        for (size_t i = 0; i < num_streams; i++) {
            if (!device_buffers_[i] || buffer_sizes_[i] < required_buffer_size) {
                need_resize = true;
                break;
            }
        }
    }
    
    // Allocate or resize buffers as needed
    if (need_resize) {
        for (size_t i = 0; i < num_streams; i++) {
            // Allocate a new buffer of adequate size
            void* new_buffer = backend_->allocate_device_memory(required_buffer_size);
            if (!new_buffer) {
                // Allocation failed
                
                // Clean up partially allocated buffers
                for (size_t j = 0; j < i; j++) {
                    backend_->free_device_memory(device_buffers_[j]);
                }
                device_buffers_.clear();
                buffer_sizes_.clear();
                
                return ProcessingResult::MemoryError;
            }
            
            // Free old buffer if it exists
            if (device_buffers_[i]) {
                backend_->free_device_memory(device_buffers_[i]);
    }
            
            // Store new buffer
            device_buffers_[i] = new_buffer;
            buffer_sizes_[i] = required_buffer_size;
        }
    }
    
    return ProcessingResult::Success;
}

// Get the total amount of device memory allocated for buffers
size_t EventProcessor::Impl::get_total_device_buffer_memory() const {
    size_t total = buffer_size_;  // Main device buffer
    
    // Add all stream buffers
    for (size_t size : buffer_sizes_) {
        total += size;
    }
    
    return total;
}

// Add the new getter method for backend type
BackendType EventProcessor::Impl::get_backend_type() const {
    return backend_->get_type();
}

// EventProcessor implementation (public interface)

EventProcessor::EventProcessor(const Config& config) 
    : pimpl_(std::make_unique<Impl>(config)) {
}

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

ProcessingResult EventProcessor::process_events(void* events_buffer, size_t buffer_size, size_t event_count, bool is_async) {
    return pimpl_->process_events(events_buffer, buffer_size, event_count, is_async);
}

ProcessingResult EventProcessor::synchronize_async_operations() {
    return pimpl_->synchronize_async_operations();
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

BackendType EventProcessor::get_backend_type() const {
    return pimpl_->get_backend_type();
                }

void* EventProcessor::allocate_pinned_buffer(size_t size) {
    // Implementation uses static GPU backend to handle this
    #if defined(USE_CUDA_BACKEND)
    auto backend = create_backend(BackendType::CUDA);
    return backend->allocate_pinned_host_memory(size);
    #elif defined(USE_OPENCL_BACKEND)
    auto backend = create_backend(BackendType::OpenCL);
    return backend->allocate_pinned_host_memory(size);
    #else
    // Fallback implementation if no backend is available
        return nullptr;
    #endif
}

void EventProcessor::free_pinned_buffer(void* pinned_ptr) {
    if (!pinned_ptr) return;
    
    #if defined(USE_CUDA_BACKEND)
    auto backend = create_backend(BackendType::CUDA);
    backend->free_pinned_host_memory(pinned_ptr);
    #elif defined(USE_OPENCL_BACKEND)
    auto backend = create_backend(BackendType::OpenCL);
    backend->free_pinned_host_memory(pinned_ptr);
    #endif
    // No action needed if no backend is available
}

ProcessingResult EventProcessor::register_host_buffer(void* ptr, size_t size, unsigned int flags) {
    if (!ptr || size == 0) {
        return ProcessingResult::InvalidInput;
    }
    
    #if defined(USE_CUDA_BACKEND)
    auto backend = create_backend(BackendType::CUDA);
    bool success = backend->register_host_memory(ptr, size, flags);
    return success ? ProcessingResult::Success : ProcessingResult::DeviceError;
    #elif defined(USE_OPENCL_BACKEND)
    auto backend = create_backend(BackendType::OpenCL);
    bool success = backend->register_host_memory(ptr, size, flags);
    return success ? ProcessingResult::Success : ProcessingResult::DeviceError;
    #else
    // No action if no backend is available
    return ProcessingResult::DeviceError;
    #endif
}

ProcessingResult EventProcessor::unregister_host_buffer(void* ptr) {
    if (!ptr) {
        return ProcessingResult::InvalidInput;
    }
    
    #if defined(USE_CUDA_BACKEND)
    auto backend = create_backend(BackendType::CUDA);
    bool success = backend->unregister_host_memory(ptr);
    return success ? ProcessingResult::Success : ProcessingResult::DeviceError;
    #elif defined(USE_OPENCL_BACKEND)
    auto backend = create_backend(BackendType::OpenCL);
    bool success = backend->unregister_host_memory(ptr);
    return success ? ProcessingResult::Success : ProcessingResult::DeviceError;
    #else
    // No action if no backend is available
    return ProcessingResult::DeviceError;
    #endif
}

std::vector<GpuDeviceInfo> get_available_devices() {
    GpuDeviceManager manager;
    std::vector<GpuDeviceInfo> devices;
    
    // Get devices for both backend types
    for (int i = 0; i < manager.get_device_count(BackendType::CUDA); i++) {
        devices.push_back(manager.query_device_info(i));
    }
    
    for (int i = 0; i < manager.get_device_count(BackendType::OpenCL); i++) {
        devices.push_back(manager.query_device_info(i));
    }
    
    return devices;
}

int select_best_device(size_t min_memory) {
    GpuDeviceManager manager;
    
    // Try CUDA first, then OpenCL
    int device_id = manager.get_best_device(BackendType::CUDA);
    if (device_id >= 0) {
        size_t available_memory = manager.get_available_memory(device_id);
        if (available_memory >= min_memory) {
            return device_id;
        }
    }
    
    // Try OpenCL if CUDA not available or not enough memory
    device_id = manager.get_best_device(BackendType::OpenCL);
    if (device_id >= 0) {
        size_t available_memory = manager.get_available_memory(device_id);
        if (available_memory >= min_memory) {
            return device_id;
        }
}

    // No suitable device found
    return -1;
}

bool validate_ptx_code(const std::string& ptx_code) {
    return KernelLoader::validate_ptx(ptx_code);
}

} // namespace ebpf_gpu 