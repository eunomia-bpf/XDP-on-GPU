#include "ebpf_gpu_processor.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include "gpu_device_manager.hpp"
#include "kernel_loader.hpp"

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
    ProcessingResult process_events(void* events_buffer, size_t buffer_size, size_t event_count);

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
    
    void initialize_device();
    ProcessingResult ensure_buffer_size(size_t required_size);
    ProcessingResult launch_kernel(void* device_data, size_t event_count);
};

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
    
    // Copy event to device
    cudaError_t result = cudaMemcpy(device_buffer_, event_data, event_size, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        return ProcessingResult::DeviceError;
    }
    
    // Launch kernel for single event
    ProcessingResult launch_result = launch_kernel(device_buffer_, 1);
    if (launch_result != ProcessingResult::Success) {
        return launch_result;
    }
    
    // Copy result back
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
    
    // Copy buffer to device
    cudaError_t result = cudaMemcpy(device_buffer_, events_buffer, buffer_size, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        return ProcessingResult::DeviceError;
    }
    
    // Launch kernel
    ProcessingResult launch_result = launch_kernel(device_buffer_, event_count);
    if (launch_result != ProcessingResult::Success) {
        return launch_result;
    }
    
    // Copy results back
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

// Utility functions
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