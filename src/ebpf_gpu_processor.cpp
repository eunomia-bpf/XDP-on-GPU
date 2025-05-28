#include "ebpf_gpu_processor.hpp"
#include "error_handling.hpp"
#include "gpu_device_manager.hpp"
#include "kernel_loader.hpp"
#include <chrono>
#include <algorithm>

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

    void load_kernel_from_ptx(const std::string& ptx_code, const std::string& function_name);
    void load_kernel_from_file(const std::string& file_path, const std::string& function_name);
    void load_kernel_from_source(const std::string& cuda_source, const std::string& function_name,
                                const std::vector<std::string>& include_paths,
                                const std::vector<std::string>& compile_options);

    ProcessingResult process_events(std::vector<NetworkEvent>& events);
    ProcessingResult process_events(NetworkEvent* events, size_t count);
    ProcessingResult process_events_async(std::vector<NetworkEvent>& events,
                                        std::function<void(ProcessingResult)> callback);
    ProcessingResult process_buffer(void* buffer, size_t buffer_size, size_t event_count);

    GpuDeviceInfo get_device_info() const;
    size_t get_available_memory() const;
    bool is_ready() const;

    PerformanceStats get_performance_stats() const;
    void reset_performance_stats();

private:
    Config config_;
    std::unique_ptr<CudaContext> context_;
    std::unique_ptr<DeviceMemory> device_buffer_;
    std::unique_ptr<CudaModule> module_;
    CUfunction kernel_function_;
    GpuDeviceManager device_manager_;
    
    mutable PerformanceStats stats_;
    
    void initialize_device();
    void ensure_buffer_size(size_t required_size);
    ProcessingResult launch_kernel(void* device_data, size_t event_count);
};

EventProcessor::Impl::Impl(const Config& config) 
    : config_(config), kernel_function_(nullptr) {
    initialize_device();
}

EventProcessor::Impl::~Impl() = default;

void EventProcessor::Impl::initialize_device() {
    int device_id = config_.device_id;
    
    if (device_id < 0) {
        device_id = device_manager_.select_best_device();
    }
    
    if (!device_manager_.is_device_suitable(device_id, config_.buffer_size)) {
        throw std::runtime_error("Selected device is not suitable for processing");
    }
    
    // Create CUDA context
    context_ = std::make_unique<CudaContext>(device_id);
    
    // Allocate device buffer
    device_buffer_ = std::make_unique<DeviceMemory>(config_.buffer_size);
}

void EventProcessor::Impl::load_kernel_from_ptx(const std::string& ptx_code, const std::string& function_name) {
    KernelLoader loader;
    module_ = loader.load_from_ptx(ptx_code);
    kernel_function_ = module_->get_function(function_name);
}

void EventProcessor::Impl::load_kernel_from_file(const std::string& file_path, const std::string& function_name) {
    KernelLoader loader;
    module_ = loader.load_from_file(file_path);
    kernel_function_ = module_->get_function(function_name);
}

void EventProcessor::Impl::load_kernel_from_source(const std::string& cuda_source, const std::string& function_name,
                                                  const std::vector<std::string>& include_paths,
                                                  const std::vector<std::string>& compile_options) {
    KernelLoader loader;
    module_ = loader.load_from_cuda_source(cuda_source, include_paths, compile_options);
    kernel_function_ = module_->get_function(function_name);
}

ProcessingResult EventProcessor::Impl::process_events(std::vector<NetworkEvent>& events) {
    return process_events(events.data(), events.size());
}

ProcessingResult EventProcessor::Impl::process_events(NetworkEvent* events, size_t count) {
    if (!is_ready()) {
        return ProcessingResult::KernelError;
    }
    
    if (count == 0) {
        return ProcessingResult::InvalidInput;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        size_t required_size = count * sizeof(NetworkEvent);
        ensure_buffer_size(required_size);
        
        // Copy events to device
        auto transfer_start = std::chrono::high_resolution_clock::now();
        device_buffer_->copy_from_host(events, required_size);
        auto transfer_end = std::chrono::high_resolution_clock::now();
        
        // Launch kernel
        auto kernel_start = std::chrono::high_resolution_clock::now();
        ProcessingResult result = launch_kernel(device_buffer_->get(), count);
        auto kernel_end = std::chrono::high_resolution_clock::now();
        
        if (result != ProcessingResult::Success) {
            return result;
        }
        
        // Copy results back
        device_buffer_->copy_to_host(events, required_size);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Update performance stats
        stats_.events_processed += count;
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start).count();
        auto transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(transfer_end - transfer_start).count();
        
        stats_.total_processing_time_us += total_time;
        stats_.kernel_execution_time_us += kernel_time;
        stats_.memory_transfer_time_us += transfer_time;
        
        if (stats_.total_processing_time_us > 0) {
            stats_.events_per_second = (stats_.events_processed * 1000000.0) / stats_.total_processing_time_us;
        }
        
        return ProcessingResult::Success;
        
    } catch (const CudaException&) {
        return ProcessingResult::DeviceError;
    } catch (const std::exception&) {
        return ProcessingResult::Error;
    }
}

ProcessingResult EventProcessor::Impl::process_events_async(std::vector<NetworkEvent>& events,
                                                           std::function<void(ProcessingResult)> callback) {
    // For now, implement synchronously
    ProcessingResult result = process_events(events);
    if (callback) {
        callback(result);
    }
    return result;
}

ProcessingResult EventProcessor::Impl::process_buffer(void* buffer, size_t buffer_size, size_t event_count) {
    if (!is_ready()) {
        return ProcessingResult::KernelError;
    }
    
    if (!buffer || buffer_size == 0 || event_count == 0) {
        return ProcessingResult::InvalidInput;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        ensure_buffer_size(buffer_size);
        
        // Copy buffer to device
        auto transfer_start = std::chrono::high_resolution_clock::now();
        device_buffer_->copy_from_host(buffer, buffer_size);
        auto transfer_end = std::chrono::high_resolution_clock::now();
        
        // Launch kernel
        auto kernel_start = std::chrono::high_resolution_clock::now();
        ProcessingResult result = launch_kernel(device_buffer_->get(), event_count);
        auto kernel_end = std::chrono::high_resolution_clock::now();
        
        if (result != ProcessingResult::Success) {
            return result;
        }
        
        // Copy results back
        device_buffer_->copy_to_host(buffer, buffer_size);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Update performance stats
        stats_.events_processed += event_count;
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start).count();
        auto transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(transfer_end - transfer_start).count();
        
        stats_.total_processing_time_us += total_time;
        stats_.kernel_execution_time_us += kernel_time;
        stats_.memory_transfer_time_us += transfer_time;
        
        if (stats_.total_processing_time_us > 0) {
            stats_.events_per_second = (stats_.events_processed * 1000000.0) / stats_.total_processing_time_us;
        }
        
        return ProcessingResult::Success;
        
    } catch (const CudaException&) {
        return ProcessingResult::DeviceError;
    } catch (const std::exception&) {
        return ProcessingResult::Error;
    }
}

GpuDeviceInfo EventProcessor::Impl::get_device_info() const {
    if (!context_) {
        throw std::runtime_error("Device not initialized");
    }
    
    // Get device ID from context and query device manager
    // For now, assume device 0 - in a real implementation, we'd track the device ID
    return device_manager_.get_device_info(0);
}

size_t EventProcessor::Impl::get_available_memory() const {
    if (!context_) {
        return 0;
    }
    
    return device_manager_.get_available_memory(0);
}

bool EventProcessor::Impl::is_ready() const {
    return context_ && device_buffer_ && module_ && kernel_function_;
}

EventProcessor::PerformanceStats EventProcessor::Impl::get_performance_stats() const {
    return stats_;
}

void EventProcessor::Impl::reset_performance_stats() {
    stats_ = PerformanceStats{};
}

void EventProcessor::Impl::ensure_buffer_size(size_t required_size) {
    if (!device_buffer_ || device_buffer_->size() < required_size) {
        device_buffer_ = std::make_unique<DeviceMemory>(std::max(required_size, config_.buffer_size));
    }
}

ProcessingResult EventProcessor::Impl::launch_kernel(void* device_data, size_t event_count) {
    if (!kernel_function_) {
        return ProcessingResult::KernelError;
    }
    
    try {
        context_->set_current();
        
        // Set up kernel parameters
        void* args[] = {
            &device_data,
            &event_count
        };
        
        // Calculate grid and block dimensions
        int block_size = 256;
        int grid_size = (event_count + block_size - 1) / block_size;
        
        // Launch kernel
        CUresult result = cuLaunchKernel(
            kernel_function_,
            grid_size, 1, 1,    // Grid dimensions
            block_size, 1, 1,   // Block dimensions
            0,                  // Shared memory
            0,                  // Stream
            args,               // Parameters
            nullptr             // Extra
        );
        
        check_cuda_driver(result, "cuLaunchKernel");
        
        // Synchronize
        check_cuda_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        
        return ProcessingResult::Success;
        
    } catch (const CudaException&) {
        return ProcessingResult::DeviceError;
    }
}

// EventProcessor public interface implementation
EventProcessor::EventProcessor(const Config& config) 
    : pimpl_(std::make_unique<Impl>(config)) {}

EventProcessor::~EventProcessor() = default;

EventProcessor::EventProcessor(EventProcessor&&) noexcept = default;
EventProcessor& EventProcessor::operator=(EventProcessor&&) noexcept = default;

void EventProcessor::load_kernel_from_ptx(const std::string& ptx_code, const std::string& function_name) {
    pimpl_->load_kernel_from_ptx(ptx_code, function_name);
}

void EventProcessor::load_kernel_from_file(const std::string& file_path, const std::string& function_name) {
    pimpl_->load_kernel_from_file(file_path, function_name);
}

void EventProcessor::load_kernel_from_source(const std::string& cuda_source, const std::string& function_name,
                                            const std::vector<std::string>& include_paths,
                                            const std::vector<std::string>& compile_options) {
    pimpl_->load_kernel_from_source(cuda_source, function_name, include_paths, compile_options);
}

ProcessingResult EventProcessor::process_events(std::vector<NetworkEvent>& events) {
    return pimpl_->process_events(events);
}

ProcessingResult EventProcessor::process_events(NetworkEvent* events, size_t count) {
    return pimpl_->process_events(events, count);
}

ProcessingResult EventProcessor::process_events_async(std::vector<NetworkEvent>& events,
                                                     std::function<void(ProcessingResult)> callback) {
    return pimpl_->process_events_async(events, callback);
}

ProcessingResult EventProcessor::process_buffer(void* buffer, size_t buffer_size, size_t event_count) {
    return pimpl_->process_buffer(buffer, buffer_size, event_count);
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

EventProcessor::PerformanceStats EventProcessor::get_performance_stats() const {
    return pimpl_->get_performance_stats();
}

void EventProcessor::reset_performance_stats() {
    pimpl_->reset_performance_stats();
}

// Utility functions
std::vector<GpuDeviceInfo> get_available_devices() {
    GpuDeviceManager manager;
    return manager.get_all_devices();
}

int select_best_device(size_t min_memory) {
    GpuDeviceManager manager;
    return manager.select_best_device();
}

bool validate_ptx_code(const std::string& ptx_code) {
    return KernelLoader::validate_ptx(ptx_code);
}

} // namespace ebpf_gpu 