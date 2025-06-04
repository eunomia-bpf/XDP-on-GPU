#include "ebpf_gpu_processor_impl.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iterator>

namespace ebpf_gpu {

EventProcessor::Impl::Impl(const Config& config) 
    : config_(config), platform_(nullptr), device_(nullptr), context_(nullptr), command_queue_(nullptr),
      device_buffer_(nullptr), buffer_size_(0), program_(nullptr), kernel_(nullptr), device_id_(-1) {
    initialize_device();
}

EventProcessor::Impl::~Impl() {
    cleanup_opencl_resources();
}

void EventProcessor::Impl::cleanup_opencl_resources() {
    // Clean up device buffer
    if (device_buffer_) {
        clReleaseMemObject(device_buffer_);
        device_buffer_ = nullptr;
    }
    
    // Clean up kernel
    if (kernel_) {
        clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
    
    // Clean up program
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }
    
    // Clean up command queue
    if (command_queue_) {
        clReleaseCommandQueue(command_queue_);
        command_queue_ = nullptr;
    }
    
    // Clean up context
    if (context_) {
        clReleaseContext(context_);
        context_ = nullptr;
    }
    
    // Clean up device (OpenCL doesn't require explicit release)
    device_ = nullptr;
    
    // Clean up platform (OpenCL doesn't require explicit release)
    platform_ = nullptr;
}

void EventProcessor::Impl::initialize_device() {
    device_id_ = config_.device_id;
    
    // Validate kernel launch configuration
    if (config_.block_size <= 0 || config_.block_size > 1024) {
        throw std::runtime_error("Invalid block size: must be between 1 and 1024");
    }
    
    // Check if any devices are available first
    if (device_manager_.get_device_count() == 0) {
        throw std::runtime_error("No OpenCL devices available");
    }
    
    if (device_id_ < 0) {
        device_id_ = device_manager_.select_best_device();
    }
    
    if (!device_manager_.is_device_suitable(device_id_, config_.buffer_size)) {
        throw std::runtime_error("Selected device is not suitable for processing");
    }
    
    // Get OpenCL platform and device
    cl_int err;
    cl_uint platform_count;
    err = clGetPlatformIDs(0, nullptr, &platform_count);
    if (err != CL_SUCCESS || platform_count == 0) {
        throw std::runtime_error("Failed to get OpenCL platforms");
    }
    
    std::vector<cl_platform_id> platforms(platform_count);
    err = clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL platforms");
    }
    
    // For simplicity, use the first platform
    platform_ = platforms[0];
    
    // Get device
    cl_uint device_count;
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
    if (err != CL_SUCCESS || device_count == 0) {
        throw std::runtime_error("Failed to get OpenCL GPU devices");
    }
    
    std::vector<cl_device_id> devices(device_count);
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL GPU devices");
    }
    
    // Use the device_id_ to select from available devices
    if (device_id_ < static_cast<int>(device_count)) {
        device_ = devices[device_id_];
    } else {
        // Fallback to first device
        device_ = devices[0];
    }
    
    // Create OpenCL context
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL context");
    }
    
    // Create command queue
    command_queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context_);
        throw std::runtime_error("Failed to create OpenCL command queue");
    }
    
    // Allocate device buffer with initial size
    device_buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, config_.buffer_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(command_queue_);
        clReleaseContext(context_);
        throw std::runtime_error("Failed to allocate device memory");
    }
    buffer_size_ = config_.buffer_size;
}

ProcessingResult EventProcessor::Impl::load_kernel_from_ir(const std::string& ir_code, const std::string& function_name) {
    try {
        if (ir_code.empty() || function_name.empty()) {
            return ProcessingResult::InvalidInput;
        }
        
        // Clean up existing kernel and program
        if (kernel_) {
            clReleaseKernel(kernel_);
            kernel_ = nullptr;
        }
        
        if (program_) {
            clReleaseProgram(program_);
            program_ = nullptr;
        }
        
        // Create program from source (for OpenCL, ir_code is actually OpenCL C source)
        cl_int err;
        const char* source_ptr = ir_code.c_str();
        size_t source_size = ir_code.size();
        
        program_ = clCreateProgramWithSource(context_, 1, &source_ptr, &source_size, &err);
        if (err != CL_SUCCESS) {
            return ProcessingResult::KernelError;
        }
        
        // Build program
        err = clBuildProgram(program_, 1, &device_, "", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            // Get build error log
            size_t log_size;
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            
            clReleaseProgram(program_);
            program_ = nullptr;
            return ProcessingResult::KernelError;
        }
        
        // Create kernel
        kernel_ = clCreateKernel(program_, function_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            clReleaseProgram(program_);
            program_ = nullptr;
            return ProcessingResult::KernelError;
        }
        
        return ProcessingResult::Success;
    } catch (const std::exception&) {
        return ProcessingResult::Error;
    }
}

ProcessingResult EventProcessor::Impl::load_kernel_from_file(const std::string& file_path, const std::string& function_name) {
    try {
        if (file_path.empty() || function_name.empty()) {
            return ProcessingResult::InvalidInput;
        }
        
        // Load source code from file
        std::ifstream file(file_path);
        if (!file.is_open()) {
            return ProcessingResult::InvalidInput;
        }
        
        std::string source_code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        // Use load_kernel_from_ir (which is actually OpenCL source code in this implementation)
        return load_kernel_from_ir(source_code, function_name);
    } catch (const std::exception&) {
        return ProcessingResult::Error;
    }
}

ProcessingResult EventProcessor::Impl::load_kernel_from_source(const std::string& source_code, const std::string& function_name,
                                                const std::vector<std::string>& include_paths,
                                                const std::vector<std::string>& compile_options) {
    // For OpenCL, we just pass through the source code directly
    return load_kernel_from_ir(source_code, function_name);
}

ProcessingResult EventProcessor::Impl::process_event(void* event_data, size_t event_size) {
    if (!is_ready()) {
        return ProcessingResult::KernelError;
    }
    
    if (!event_data || event_size == 0) {
        return ProcessingResult::InvalidInput;
    }
    
    cl_int err;
    
    // Create input buffer
    cl_mem input_buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, event_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        return ProcessingResult::MemoryError;
    }
    
    // Create output buffer
    cl_mem output_buffer = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, sizeof(uint32_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        return ProcessingResult::MemoryError;
    }
    
    // Copy input data to device
    err = clEnqueueWriteBuffer(command_queue_, input_buffer, CL_TRUE, 0, 
                              event_size, event_data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        return ProcessingResult::DeviceError;
    }
    
    // Set kernel arguments
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &input_buffer);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        return ProcessingResult::KernelError;
    }
    
    err = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &output_buffer);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        return ProcessingResult::KernelError;
    }
    
    uint32_t length = static_cast<uint32_t>(event_size);
    err = clSetKernelArg(kernel_, 2, sizeof(uint32_t), &length);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        return ProcessingResult::KernelError;
    }
    
    // Launch kernel
    size_t global_work_size = config_.block_size;
    size_t local_work_size = config_.block_size;
    
    err = clEnqueueNDRangeKernel(command_queue_, kernel_, 1, nullptr, 
                                &global_work_size, &local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        return ProcessingResult::KernelError;
    }
    
    // Wait for kernel to complete
    err = clFinish(command_queue_);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        return ProcessingResult::DeviceError;
    }
    
    // Read back results to original event data
    err = clEnqueueReadBuffer(command_queue_, input_buffer, CL_TRUE, 0,
                             event_size, event_data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        return ProcessingResult::DeviceError;
    }
    
    // Cleanup
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    
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
    
    // Ensure buffer is large enough
    if (buffer_size_ < buffer_size) {
        // Release old buffer
        if (device_buffer_) {
            clReleaseMemObject(device_buffer_);
        }
        
        // Create new buffer
        cl_int err;
        device_buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
        if (err != CL_SUCCESS) {
            device_buffer_ = nullptr;
            buffer_size_ = 0;
            return ProcessingResult::MemoryError;
        }
        buffer_size_ = buffer_size;
    }
    
    cl_int err;
    
    // Copy data to device
    err = clEnqueueWriteBuffer(command_queue_, device_buffer_, CL_TRUE, 0, 
                              buffer_size, events_buffer, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        return ProcessingResult::DeviceError;
    }
    
    // Create output buffer
    cl_mem result_buffer = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, event_count * sizeof(uint32_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        return ProcessingResult::MemoryError;
    }
    
    // Set kernel arguments
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &device_buffer_);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(result_buffer);
        return ProcessingResult::KernelError;
    }
    
    err = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &result_buffer);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(result_buffer);
        return ProcessingResult::KernelError;
    }
    
    uint32_t length = static_cast<uint32_t>(event_count);
    err = clSetKernelArg(kernel_, 2, sizeof(uint32_t), &length);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(result_buffer);
        return ProcessingResult::KernelError;
    }
    
    // Calculate work dimensions
    size_t global_work_size = ((event_count + config_.block_size - 1) / config_.block_size) * config_.block_size;
    size_t local_work_size = config_.block_size;
    
    // Launch kernel
    err = clEnqueueNDRangeKernel(command_queue_, kernel_, 1, nullptr, 
                                &global_work_size, &local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(result_buffer);
        return ProcessingResult::KernelError;
    }
    
    // Wait for kernel completion
    err = clFinish(command_queue_);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(result_buffer);
        return ProcessingResult::DeviceError;
    }
    
    // Read back results to original buffer
    err = clEnqueueReadBuffer(command_queue_, device_buffer_, CL_TRUE, 0,
                             buffer_size, events_buffer, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(result_buffer);
        return ProcessingResult::DeviceError;
    }
    
    // Cleanup
    clReleaseMemObject(result_buffer);
    
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
    return context_ && device_buffer_ && kernel_;
}

ProcessingResult EventProcessor::Impl::synchronize_async_operations() {
    // Just wait for the command queue to complete
    if (command_queue_) {
        cl_int err = clFinish(command_queue_);
        if (err != CL_SUCCESS) {
            return ProcessingResult::DeviceError;
        }
    }
    return ProcessingResult::Success;
}

// EventProcessor public interface implementation
EventProcessor::EventProcessor(const Config& config) 
    : pimpl_(std::make_unique<Impl>(config)) {}

EventProcessor::~EventProcessor() = default;

EventProcessor::EventProcessor(EventProcessor&&) noexcept = default;
EventProcessor& EventProcessor::operator=(EventProcessor&&) noexcept = default;

ProcessingResult EventProcessor::load_kernel_from_ir(const std::string& ir_code, const std::string& function_name) {
    return pimpl_->load_kernel_from_ir(ir_code, function_name);
}

ProcessingResult EventProcessor::load_kernel_from_file(const std::string& file_path, const std::string& function_name) {
    return pimpl_->load_kernel_from_file(file_path, function_name);
}

ProcessingResult EventProcessor::load_kernel_from_source(const std::string& source_code, const std::string& function_name,
                                         const std::vector<std::string>& include_paths,
                                         const std::vector<std::string>& compile_options) {
    return pimpl_->load_kernel_from_source(source_code, function_name, include_paths, compile_options);
}

ProcessingResult EventProcessor::process_event(void* event_data, size_t event_size) {
    return pimpl_->process_event(event_data, event_size);
}

ProcessingResult EventProcessor::process_events(void* events_buffer, size_t buffer_size, size_t event_count,
                                         bool is_async) {
    // Ignore is_async parameter in simplified implementation
    return pimpl_->process_events(events_buffer, buffer_size, event_count, false);
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

ProcessingResult EventProcessor::synchronize_async_operations() {
    return pimpl_->synchronize_async_operations();
}

// Utility functions for pinned memory management
void* EventProcessor::allocate_pinned_buffer(size_t size) {
    // OpenCL doesn't have an exact equivalent to CUDA pinned memory
    // Just use regular malloc for now
    return malloc(size);
}

void EventProcessor::free_pinned_buffer(void* pinned_ptr) {
    if (pinned_ptr) {
        free(pinned_ptr);
    }
}

// Utility functions for registering/unregistering existing host memory
ProcessingResult EventProcessor::register_host_buffer(void* ptr, size_t size, unsigned int flags) {
    // OpenCL doesn't have an exact equivalent to CUDA's host memory registration
    // Return success for API compatibility
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::unregister_host_buffer(void* ptr) {
    // OpenCL doesn't have an exact equivalent to CUDA's host memory registration
    // Return success for API compatibility
    return ProcessingResult::Success;
}

// Global utility functions
std::vector<GpuDeviceInfo> get_available_devices() {
    static GpuDeviceManager manager;
    return manager.get_all_devices();
}

int select_best_device(size_t min_memory) {
    static GpuDeviceManager manager;
    return manager.select_best_device();
}

bool validate_ptx_code(const std::string& ptx_code) {
    // For OpenCL, check for kernel functions
    return ptx_code.find("__kernel") != std::string::npos ||
           ptx_code.find("kernel") != std::string::npos;
}

} // namespace ebpf_gpu 
