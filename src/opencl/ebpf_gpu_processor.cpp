#include "ebpf_gpu_processor_impl.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <functional>
#include <thread>
#include <fstream>
#include <iterator>

namespace ebpf_gpu {

EventProcessor::Impl::Impl(const Config& config) 
    : config_(config), platform_(nullptr), device_(nullptr), context_(nullptr), command_queue_(nullptr),
      device_buffer_(nullptr), buffer_size_(0), program_(nullptr), kernel_(nullptr), 
      device_id_(-1), current_queue_idx_(0) {
    initialize_device();
}

EventProcessor::Impl::~Impl() {
    // Clean up OpenCL resources
    cleanup_opencl_resources();
}

void EventProcessor::Impl::cleanup_opencl_resources() {
    // Clean up multiple device buffers
    for (cl_mem buffer : device_buffers_) {
        if (buffer) {
            clReleaseMemObject(buffer);
        }
    }
    device_buffers_.clear();
    buffer_sizes_.clear();
    
    // Clean up original device buffer
    if (device_buffer_) {
        clReleaseMemObject(device_buffer_);
        device_buffer_ = nullptr;
    }
    
    // Clean up command queues
    cleanup_command_queues();
    
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
    
    if (config_.max_grid_size < 0) {
        throw std::runtime_error("Invalid max grid size: must be non-negative");
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
    // For simplicity in this implementation, just use the first device
    device_ = devices[0];
    
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
    
    // Allocate device buffer
    device_buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, config_.buffer_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(command_queue_);
        clReleaseContext(context_);
        throw std::runtime_error("Failed to allocate device memory");
    }
    buffer_size_ = config_.buffer_size;
    
    // Initialize command queues
    initialize_command_queues();
}

ProcessingResult EventProcessor::Impl::load_kernel_from_ir(const std::string& ir_code, const std::string& function_name) {
    try {
        if (ir_code.empty()) {
            return ProcessingResult::InvalidInput;
        }
        if (function_name.empty()) {
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
            
            // Print or log the error
            // std::cerr << "OpenCL program build failed: " << log.data() << std::endl;
            
            return ProcessingResult::KernelError;
        }
        
        // Create kernel
        kernel_ = clCreateKernel(program_, function_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            return ProcessingResult::KernelError;
        }
        
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
        
        // Load source code from file
        std::ifstream file(file_path);
        if (!file.is_open()) {
            return ProcessingResult::InvalidInput;
        }
        
        std::string source_code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        // Use load_kernel_from_ir (which is actually OpenCL source code in this implementation)
        return load_kernel_from_ir(source_code, function_name);
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
        
        // For OpenCL, we just pass through the source code directly
        // We don't perform any source translation from CUDA to OpenCL in this simple implementation
        return load_kernel_from_ir(source_code, function_name);
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
    
    // We don't assume any specific structure layout.
    // The caller is responsible for ensuring the event_data contains:
    // 1. A pointer to packet data
    // 2. The length of the packet
    // 3. A place to store the result
    
    // The OpenCL kernel expects 3 parameters:
    // 1. packet_data (buffer)
    // 2. packet_length (scalar)
    // 3. result (buffer)
    
    cl_int err;
    
    // Extract packet data and length from the event data
    // This should be done by the caller and passed in a format
    // that can be used directly by the kernel
    void* packet_data = event_data;
    size_t packet_length = event_size;
    
    // Create buffers for packet data and result
    cl_mem data_buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, packet_length, nullptr, &err);
    if (err != CL_SUCCESS) {
        return ProcessingResult::MemoryError;
    }
    
    // Create a result buffer
    uint32_t result_value = 0;
    cl_mem result_buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(data_buffer);
        return ProcessingResult::MemoryError;
    }
    
    // Copy packet data to device
    err = clEnqueueWriteBuffer(command_queue_, data_buffer, CL_TRUE, 0, 
                              packet_length, packet_data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(data_buffer);
        clReleaseMemObject(result_buffer);
        return ProcessingResult::DeviceError;
    }
    
    // Set kernel arguments
    // Arg 0: packet data buffer
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &data_buffer);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(data_buffer);
        clReleaseMemObject(result_buffer);
        return ProcessingResult::KernelError;
    }
    
    // Arg 1: packet length
    err = clSetKernelArg(kernel_, 1, sizeof(uint32_t), &packet_length);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(data_buffer);
        clReleaseMemObject(result_buffer);
        return ProcessingResult::KernelError;
    }
    
    // Arg 2: result buffer
    err = clSetKernelArg(kernel_, 2, sizeof(cl_mem), &result_buffer);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(data_buffer);
        clReleaseMemObject(result_buffer);
        return ProcessingResult::KernelError;
    }
    
    // Launch kernel for single event
    size_t global_work_size = config_.block_size;
    size_t local_work_size = config_.block_size;
    
    err = clEnqueueNDRangeKernel(command_queue_, kernel_, 1, nullptr, 
                                &global_work_size, &local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(data_buffer);
        clReleaseMemObject(result_buffer);
        return ProcessingResult::KernelError;
    }
    
    // Read back the result
    err = clEnqueueReadBuffer(command_queue_, result_buffer, CL_TRUE, 0, 
                             sizeof(uint32_t), &result_value, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(data_buffer);
        clReleaseMemObject(result_buffer);
        return ProcessingResult::DeviceError;
    }
    
    // Copy result back to original data if needed
    // This depends on the caller's structure, which we don't assume
    
    // Copy the result back to event_data if needed
    // We assume the last 4 bytes of event_data can store the result
    if (event_size >= sizeof(uint32_t)) {
        uint32_t* result_ptr = static_cast<uint32_t*>(event_data) + (event_size / sizeof(uint32_t) - 1);
        *result_ptr = result_value;
    }
    
    // Cleanup
    clReleaseMemObject(data_buffer);
    clReleaseMemObject(result_buffer);
    
    // Ensure all operations complete
    err = clFinish(command_queue_);
    if (err != CL_SUCCESS) {
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
    
    // Determine the optimal batch size for processing
    // If max_batch_size is 0 or greater than event_count, use event_count as batch_size
    size_t batch_size = (config_.max_batch_size > 0 && config_.max_batch_size < event_count) 
                      ? config_.max_batch_size : event_count;
    
    // Calculate the number of batches
    size_t num_batches = (event_count + batch_size - 1) / batch_size; // Ceiling division
    
    // For a single batch, optimize by avoiding overhead
    if (num_batches == 1) {
        return process_events_single_batch(events_buffer, buffer_size, event_count, is_async);
    } else {
        // Use multi-batch processing
        return process_events_multi_batch(events_buffer, buffer_size, event_count, is_async);
    }
}

ProcessingResult EventProcessor::Impl::process_events_single_batch(void* events_buffer, size_t buffer_size, 
                                                                  size_t event_count, bool is_async) {
    // Ensure the device buffer is large enough
    ProcessingResult buffer_result = ensure_buffer_size(buffer_size);
    if (buffer_result != ProcessingResult::Success) {
        return ProcessingResult::DeviceError;
    }
    
    // Get a command queue for this operation
    cl_command_queue queue = get_available_command_queue();
    cl_int err;
    
    // Transfer data to device
    err = clEnqueueWriteBuffer(queue, device_buffer_, CL_FALSE, 0, buffer_size, events_buffer, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        return ProcessingResult::DeviceError;
    }
    
    // Our packet_filter kernel expects three arguments:
    // 1. __global const unsigned char* packet_data (input buffer)
    // 2. const unsigned int packet_length (scalar)
    // 3. __global unsigned int* result (output buffer)
    
    // Allocate a separate buffer for results
    cl_mem result_buffer = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, event_count * sizeof(uint32_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        return ProcessingResult::MemoryError;
    }
    
    // Set kernel argument 0: packet data buffer
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &device_buffer_);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(result_buffer);
        return ProcessingResult::KernelError;
    }
    
    // Set kernel argument 1: packet length
    // We're assuming each event contains SimplePacket structure with a length field
    // For batch processing, we use a default packet length
    uint32_t packet_length = buffer_size / event_count;
    err = clSetKernelArg(kernel_, 1, sizeof(uint32_t), &packet_length);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(result_buffer);
        return ProcessingResult::KernelError;
    }
    
    // Set kernel argument 2: result buffer
    err = clSetKernelArg(kernel_, 2, sizeof(cl_mem), &result_buffer);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(result_buffer);
        return ProcessingResult::KernelError;
    }
    
    // Calculate work dimensions
    size_t global_work_size = ((event_count + config_.block_size - 1) / config_.block_size) * config_.block_size;
    size_t local_work_size = config_.block_size;
    
    // Apply max grid size limit if configured
    if (config_.max_grid_size > 0 && global_work_size > config_.max_grid_size * local_work_size) {
        global_work_size = config_.max_grid_size * local_work_size;
    }
    
    // Launch kernel
    err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, &global_work_size, 
                                &local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(result_buffer);
        return ProcessingResult::KernelError;
    }
    
    // Allocate memory for results on host
    std::vector<uint32_t> host_results(event_count, 0);
    
    // Copy results back from device
    err = clEnqueueReadBuffer(queue, result_buffer, CL_FALSE, 0, 
                             event_count * sizeof(uint32_t), host_results.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(result_buffer);
        return ProcessingResult::DeviceError;
    }
    
    // Copy original data back to host
    err = clEnqueueReadBuffer(queue, device_buffer_, CL_FALSE, 0, buffer_size, events_buffer, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(result_buffer);
        return ProcessingResult::DeviceError;
    }
    
    // For synchronous operation, wait for everything to complete
    if (!is_async) {
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(result_buffer);
            return ProcessingResult::DeviceError;
        }
        
        // If synchronous, we can copy results back to the events buffer
        // We assume the events buffer has space for results (usually at the end of each event structure)
        // For SimplePacket, we'd need to update this logic based on the actual structure
        
        // Release the result buffer
        clReleaseMemObject(result_buffer);
    }
    
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::Impl::process_events_multi_batch(void* events_buffer, size_t buffer_size, 
                                                               size_t event_count, bool is_async) {
    // Calculate batch parameters
    size_t batch_size = (config_.max_batch_size > 0 && config_.max_batch_size < event_count) 
                      ? config_.max_batch_size : event_count;
    size_t num_batches = (event_count + batch_size - 1) / batch_size;
    size_t event_size = buffer_size / event_count;
    
    // Assuming the event structure is:
    // struct {
    //   void* data;       // pointer to packet data
    //   uint32_t length;  // packet length 
    //   uint32_t result;  // result value
    // }
    struct EventLayout {
        void*    data_ptr;
        uint32_t length;
        uint32_t result;
    };
    
    // Verify that event size is at least as large as our expected layout
    if (event_size < sizeof(EventLayout)) {
        return ProcessingResult::InvalidInput;
    }
    
    // Initialize command queues if not already done
    if (command_queues_.empty()) {
        initialize_command_queues();
    }
    
    size_t num_queues = command_queues_.size();
    if (num_queues == 0) {
        return ProcessingResult::DeviceError;
    }
    
    // Process batches
    for (size_t batch = 0; batch < num_batches; batch++) {
        size_t offset = batch * batch_size;
        size_t current_batch_events = std::min(batch_size, event_count - offset);
        size_t current_batch_bytes = current_batch_events * event_size;
        char* batch_buffer = static_cast<char*>(events_buffer) + (offset * event_size);
        
        // Get command queue for this batch using round-robin
        size_t queue_idx = batch % num_queues;
        cl_command_queue current_queue = command_queues_[queue_idx];
        
        // Process each event in the batch
        for (size_t i = 0; i < current_batch_events; i++) {
            EventLayout* event = reinterpret_cast<EventLayout*>(batch_buffer + i * event_size);
            
            // Skip if data pointer is null
            if (!event->data_ptr || event->length == 0) {
                continue;
            }
            
            cl_int err;
            
            // Ensure buffer for packet data
            cl_mem data_buffer = clCreateBuffer(context_, CL_MEM_READ_ONLY, event->length, nullptr, &err);
            if (err != CL_SUCCESS) {
                continue; // Skip this event
            }
            
            // Create result buffer
            cl_mem result_buffer = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, sizeof(uint32_t), nullptr, &err);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(data_buffer);
                continue; // Skip this event
            }
            
            // Copy packet data to device
            err = clEnqueueWriteBuffer(current_queue, data_buffer, CL_FALSE, 0, 
                                      event->length, event->data_ptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(data_buffer);
                clReleaseMemObject(result_buffer);
                continue; // Skip this event
            }
            
            // Set kernel arguments
            err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &data_buffer);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(data_buffer);
                clReleaseMemObject(result_buffer);
                continue; // Skip this event
            }
            
            err = clSetKernelArg(kernel_, 1, sizeof(uint32_t), &event->length);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(data_buffer);
                clReleaseMemObject(result_buffer);
                continue; // Skip this event
            }
            
            err = clSetKernelArg(kernel_, 2, sizeof(cl_mem), &result_buffer);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(data_buffer);
                clReleaseMemObject(result_buffer);
                continue; // Skip this event
            }
            
            // Launch kernel for this event
            size_t global_work_size = config_.block_size;
            size_t local_work_size = config_.block_size;
            
            err = clEnqueueNDRangeKernel(current_queue, kernel_, 1, nullptr, 
                                        &global_work_size, &local_work_size, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(data_buffer);
                clReleaseMemObject(result_buffer);
                continue; // Skip this event
            }
            
            // Read result
            err = clEnqueueReadBuffer(current_queue, result_buffer, CL_FALSE, 0, 
                                     sizeof(uint32_t), &event->result, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(data_buffer);
                clReleaseMemObject(result_buffer);
                continue; // Skip this event
            }
            
            // Clean up resources
            clReleaseMemObject(data_buffer);
            clReleaseMemObject(result_buffer);
        }
    }
    
    // Final synchronization based on async flag
    if (!is_async) {
        // Wait for all command queues to complete
        for (auto& queue : command_queues_) {
            cl_int finish_err = clFinish(queue);
            if (finish_err != CL_SUCCESS) {
                return ProcessingResult::DeviceError;
            }
        }
    }
    
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::Impl::launch_kernel(cl_mem device_data, size_t event_count) {
    if (!kernel_) {
        return ProcessingResult::KernelError;
    }
    
    cl_int err;
    
    // Our packet_filter kernel expects three arguments:
    // 1. __global const unsigned char* packet_data (input buffer)
    // 2. const unsigned int packet_length (scalar)
    // 3. __global unsigned int* result (output buffer)
    
    // For simplicity, we'll use the same buffer for input and output
    // The first part contains packet data, and the later part will hold results
    
    // Set kernel argument 0: packet data buffer
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &device_data);
    if (err != CL_SUCCESS) {
        return ProcessingResult::KernelError;
    }
    
    // Set kernel argument 1: packet length (assuming fixed size for simplicity)
    // You may need to adjust this based on your actual packet structure
    uint32_t packet_length = 1500; // Maximum Ethernet frame size
    err = clSetKernelArg(kernel_, 1, sizeof(uint32_t), &packet_length);
    if (err != CL_SUCCESS) {
        return ProcessingResult::KernelError;
    }
    
    // Set kernel argument 2: result buffer (using the same device_data buffer)
    // In a real implementation, you might want to use a separate buffer
    err = clSetKernelArg(kernel_, 2, sizeof(cl_mem), &device_data);
    if (err != CL_SUCCESS) {
        return ProcessingResult::KernelError;
    }
    
    // Calculate work dimensions
    size_t global_work_size = ((event_count + config_.block_size - 1) / config_.block_size) * config_.block_size;
    size_t local_work_size = config_.block_size;
    
    // Apply max grid size limit if configured
    if (config_.max_grid_size > 0 && global_work_size > config_.max_grid_size * local_work_size) {
        global_work_size = config_.max_grid_size * local_work_size;
    }
    
    // Launch kernel
    err = clEnqueueNDRangeKernel(command_queue_, kernel_, 1, nullptr, &global_work_size, 
                                &local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        return ProcessingResult::KernelError;
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
    return context_ && device_buffer_ && kernel_;
}

ProcessingResult EventProcessor::Impl::ensure_buffer_size(size_t required_size) {
    if (!device_buffer_ || buffer_size_ < required_size) {
        if (device_buffer_) {
            clReleaseMemObject(device_buffer_);
            device_buffer_ = nullptr;
        }
        
        size_t new_size = std::max(required_size, config_.buffer_size);
        cl_int err;
        
        // Allocate a new buffer
        device_buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, new_size, nullptr, &err);
        if (err != CL_SUCCESS) {
            device_buffer_ = nullptr;
            buffer_size_ = 0;
            return ProcessingResult::MemoryError;
        }
        buffer_size_ = new_size;
    }
    return ProcessingResult::Success;
}

void EventProcessor::Impl::initialize_command_queues() {
    // Cleanup existing command queues first
    cleanup_command_queues();
    
    // Determine the number of command queues to create from config
    int num_queues = config_.max_stream_count;
    if (num_queues <= 0) {
        // Default to 2 queues if not specified
        num_queues = 2;
    }
    
    // Ensure we don't create too many queues
    if (num_queues > 16) {
        num_queues = 16;
    }
    
    // Create the command queues
    command_queues_.reserve(num_queues);
    device_buffers_.clear();
    buffer_sizes_.clear();
    
    for (int i = 0; i < num_queues; i++) {
        cl_int err;
        cl_command_queue queue = clCreateCommandQueue(context_, device_, 0, &err);
        if (err != CL_SUCCESS) {
            // If we can't create a queue, just break
            break;
        }
        
        command_queues_.push_back(queue);
    }
    
    // Ensure we have at least one command queue
    if (command_queues_.empty()) {
        // Create at least one command queue if none were created
        cl_int err;
        cl_command_queue queue = clCreateCommandQueue(context_, device_, 0, &err);
        if (err == CL_SUCCESS) {
            command_queues_.push_back(queue);
        } else {
            throw std::runtime_error("Failed to create any OpenCL command queues");
        }
    }
    
    // Reset current queue index
    current_queue_idx_ = 0;
    
    // Initialize device buffers for each command queue
    ensure_command_queue_buffers(config_.buffer_size);
}

void EventProcessor::Impl::cleanup_command_queues() {
    // Release all command queues
    for (auto& queue : command_queues_) {
        clFinish(queue);
        clReleaseCommandQueue(queue);
    }
    
    // Clear the vector
    command_queues_.clear();
    
    // Clean up device buffers associated with command queues
    for (cl_mem buffer : device_buffers_) {
        if (buffer) {
            clReleaseMemObject(buffer);
        }
    }
    device_buffers_.clear();
    buffer_sizes_.clear();
}

cl_command_queue EventProcessor::Impl::get_available_command_queue() {
    // Initialize command queues if they haven't been created yet
    if (command_queues_.empty()) {
        initialize_command_queues();
    }
    
    // Get the next command queue in a round-robin fashion
    cl_command_queue queue = command_queues_[current_queue_idx_];
    
    // Update the index for the next call
    current_queue_idx_ = (current_queue_idx_ + 1) % command_queues_.size();
    
    return queue;
}

ProcessingResult EventProcessor::Impl::ensure_command_queue_buffers(size_t required_buffer_size) {
    size_t num_queues = command_queues_.size();
    if (num_queues == 0) {
        return ProcessingResult::DeviceError;
    }
    
    // Check if we need to allocate or reallocate buffers
    if (device_buffers_.size() != num_queues) {
        // Clean up old buffers
        for (cl_mem buffer : device_buffers_) {
            if (buffer) {
                clReleaseMemObject(buffer);
            }
        }
        device_buffers_.clear();
        buffer_sizes_.clear();
        
        // Allocate new buffers for each command queue
        device_buffers_.resize(num_queues);
        buffer_sizes_.resize(num_queues);
        
        for (size_t i = 0; i < num_queues; i++) {
            cl_int err;
            device_buffers_[i] = clCreateBuffer(context_, CL_MEM_READ_WRITE, required_buffer_size, nullptr, &err);
            
            if (err != CL_SUCCESS) {
                // Clean up partially allocated buffers
                for (size_t j = 0; j < i; j++) {
                    clReleaseMemObject(device_buffers_[j]);
                }
                device_buffers_.clear();
                buffer_sizes_.clear();
                return ProcessingResult::MemoryError;
            }
            buffer_sizes_[i] = required_buffer_size;
        }
    } else {
        // Ensure existing buffers are large enough
        for (size_t i = 0; i < num_queues; i++) {
            if (buffer_sizes_[i] < required_buffer_size) {
                // Release old buffer
                clReleaseMemObject(device_buffers_[i]);
                
                // Create new buffer
                cl_int err;
                device_buffers_[i] = clCreateBuffer(context_, CL_MEM_READ_WRITE, required_buffer_size, nullptr, &err);
                
                if (err != CL_SUCCESS) {
                    return ProcessingResult::MemoryError;
                }
                buffer_sizes_[i] = required_buffer_size;
            }
        }
    }
    
    return ProcessingResult::Success;
}

ProcessingResult EventProcessor::Impl::synchronize_async_operations() {
    if (!is_ready()) {
        return ProcessingResult::KernelError;
    }
    
    // Synchronize all command queues
    for (auto& queue : command_queues_) {
        cl_int err = clFinish(queue);
        if (err != CL_SUCCESS) {
            return ProcessingResult::DeviceError;
        }
    }
    
    return ProcessingResult::Success;
}

size_t EventProcessor::Impl::get_total_device_buffer_memory() const {
    size_t total = buffer_size_;  // Original device buffer
    for (size_t size : buffer_sizes_) {
        total += size;  // Additional per-command queue buffers
    }
    return total;
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