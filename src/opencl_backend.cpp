#include "../include/gpu_backend.hpp"
#include <CL/cl.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <map>
#include <algorithm>

namespace ebpf_gpu {

class OpenCLBackend : public GpuBackend {
public:
    OpenCLBackend();
    ~OpenCLBackend() override;

    // Backend type
    BackendType get_type() const override { return BackendType::OpenCL; }
    
    // Device management
    void initialize_device(int device_id) override;
    void set_device(int device_id) override;
    
    // Memory management
    void* allocate_device_memory(size_t size) override;
    void free_device_memory(void* ptr) override;
    void copy_host_to_device(void* dst, const void* src, size_t size) override;
    void copy_device_to_host(void* dst, const void* src, size_t size) override;
    
    // Kernel management
    bool load_kernel_from_source(const std::string& source_code, 
                                const std::string& function_name,
                                const std::vector<std::string>& include_paths = {},
                                const std::vector<std::string>& compile_options = {}) override;
    bool load_kernel_from_binary(const std::string& binary_path, 
                                const std::string& function_name) override;
    
    // Kernel execution
    bool launch_kernel(void* data, size_t event_count, 
                      int block_size, size_t shared_memory_size, 
                      int max_grid_size) override;
    
    // Async execution support
    void* create_stream() override;
    void destroy_stream(void* stream) override;
    bool launch_kernel_async(void* data, size_t event_count, 
                            int block_size, size_t shared_memory_size, 
                            int max_grid_size, void* stream) override;
    bool synchronize_stream(void* stream) override;
    bool synchronize_device() override;
    
    // Memory management with streams
    void copy_host_to_device_async(void* dst, const void* src, size_t size, void* stream) override;
    void copy_device_to_host_async(void* dst, const void* src, size_t size, void* stream) override;
    
    // Pinned memory support
    void* allocate_pinned_host_memory(size_t size) override;
    void free_pinned_host_memory(void* ptr) override;
    bool register_host_memory(void* ptr, size_t size, unsigned int flags) override;
    bool unregister_host_memory(void* ptr) override;
    
    // Device information
    GpuDeviceInfo get_device_info(int device_id) const override;
    size_t get_available_memory(int device_id) const override;

private:
    struct CLMemWrapper {
        cl_mem mem;
        size_t size;
    };
    
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue command_queue_;
    cl_program program_;
    cl_kernel kernel_;
    int current_device_id_;
    
    // Keep track of allocated memory objects
    std::map<void*, CLMemWrapper> memory_objects_;
    std::map<void*, void*> pinned_memory_map_;
    
    // Device discovery
    std::vector<cl_device_id> get_available_devices() const;
    std::vector<cl_platform_id> get_platforms() const;
    
    // Helper methods for OpenCL error checking
    static void check_cl_error(cl_int error, const char* operation);
    
    // Helper to read binary file
    static std::vector<unsigned char> read_binary_file(const std::string& file_path);
    
    // Convert to compatible parameter
    void* to_cl_mem(void* ptr) const;
};

OpenCLBackend::OpenCLBackend() 
    : platform_(nullptr), device_(nullptr), context_(nullptr), 
      command_queue_(nullptr), program_(nullptr), kernel_(nullptr),
      current_device_id_(-1) {
    
    // Get platform
    auto platforms = get_platforms();
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found");
    }
    
    platform_ = platforms[0]; // Use the first platform by default
}

OpenCLBackend::~OpenCLBackend() {
    // Clean up allocated memory
    for (auto& [ptr, wrapper] : memory_objects_) {
        if (wrapper.mem) {
            clReleaseMemObject(wrapper.mem);
        }
    }
    memory_objects_.clear();
    
    // Clean up pinned memory
    for (auto& [host_ptr, mapped_ptr] : pinned_memory_map_) {
        if (mapped_ptr) {
            free_pinned_host_memory(host_ptr);
        }
    }
    pinned_memory_map_.clear();
    
    // Release OpenCL resources
    if (kernel_) {
        clReleaseKernel(kernel_);
    }
    
    if (program_) {
        clReleaseProgram(program_);
    }
    
    if (command_queue_) {
        clReleaseCommandQueue(command_queue_);
    }
    
    if (context_) {
        clReleaseContext(context_);
    }
}

std::vector<cl_platform_id> OpenCLBackend::get_platforms() const {
    cl_uint num_platforms = 0;
    cl_int error = clGetPlatformIDs(0, nullptr, &num_platforms);
    check_cl_error(error, "clGetPlatformIDs");
    
    if (num_platforms == 0) {
        return {};
    }
    
    std::vector<cl_platform_id> platforms(num_platforms);
    error = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    check_cl_error(error, "clGetPlatformIDs");
    
    return platforms;
}

std::vector<cl_device_id> OpenCLBackend::get_available_devices() const {
    if (!platform_) {
        return {};
    }
    
    cl_uint num_devices = 0;
    cl_int error = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    
    // If no GPU devices, try to get CPU devices
    if (error != CL_SUCCESS || num_devices == 0) {
        error = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices);
        if (error != CL_SUCCESS || num_devices == 0) {
            return {};
        }
    }
    
    std::vector<cl_device_id> devices(num_devices);
    error = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    
    // If getting GPU devices failed, try CPU devices
    if (error != CL_SUCCESS) {
        error = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, num_devices, devices.data(), nullptr);
        if (error != CL_SUCCESS) {
            return {};
        }
    }
    
    return devices;
}

void OpenCLBackend::initialize_device(int device_id) {
    // Clean up previous device resources
    if (command_queue_) {
        clReleaseCommandQueue(command_queue_);
        command_queue_ = nullptr;
    }
    
    if (context_) {
        clReleaseContext(context_);
        context_ = nullptr;
    }
    
    // Get all available devices
    auto devices = get_available_devices();
    if (devices.empty()) {
        throw std::runtime_error("No OpenCL devices available");
    }
    
    // Use the specified device or the first one
    if (device_id >= 0 && static_cast<size_t>(device_id) < devices.size()) {
        device_ = devices[device_id];
    } else {
        device_ = devices[0];
        device_id = 0;
    }
    
    current_device_id_ = device_id;
    
    // Create OpenCL context
    cl_int error;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &error);
    check_cl_error(error, "clCreateContext");
    
    // Create command queue
    command_queue_ = clCreateCommandQueue(context_, device_, 0, &error);
    check_cl_error(error, "clCreateCommandQueue");
}

void OpenCLBackend::set_device(int device_id) {
    if (device_id != current_device_id_) {
        initialize_device(device_id);
    }
}

void* OpenCLBackend::allocate_device_memory(size_t size) {
    if (!context_) {
        return nullptr;
    }
    
    cl_int error;
    cl_mem mem = clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, &error);
    if (error != CL_SUCCESS) {
        return nullptr;
    }
    
    // Create a unique virtual pointer to track this memory
    void* ptr = static_cast<void*>(mem);
    memory_objects_[ptr] = {mem, size};
    
    return ptr;
}

void OpenCLBackend::free_device_memory(void* ptr) {
    if (!ptr) {
        return;
    }
    
    auto it = memory_objects_.find(ptr);
    if (it != memory_objects_.end()) {
        clReleaseMemObject(it->second.mem);
        memory_objects_.erase(it);
    }
}

void* OpenCLBackend::to_cl_mem(void* ptr) const {
    auto it = memory_objects_.find(ptr);
    if (it != memory_objects_.end()) {
        return it->second.mem;
    }
    return ptr;
}

void OpenCLBackend::copy_host_to_device(void* dst, const void* src, size_t size) {
    if (!command_queue_ || !dst || !src) {
        return;
    }
    
    cl_mem cl_dst = static_cast<cl_mem>(to_cl_mem(dst));
    cl_int error = clEnqueueWriteBuffer(
        command_queue_, cl_dst, CL_TRUE, 0, size, src, 0, nullptr, nullptr);
    check_cl_error(error, "clEnqueueWriteBuffer");
}

void OpenCLBackend::copy_device_to_host(void* dst, const void* src, size_t size) {
    if (!command_queue_ || !dst || !src) {
        return;
    }
    
    cl_mem cl_src = static_cast<cl_mem>(to_cl_mem(const_cast<void*>(src)));
    cl_int error = clEnqueueReadBuffer(
        command_queue_, cl_src, CL_TRUE, 0, size, dst, 0, nullptr, nullptr);
    check_cl_error(error, "clEnqueueReadBuffer");
}

bool OpenCLBackend::load_kernel_from_source(const std::string& source_code,
                                           const std::string& function_name,
                                           const std::vector<std::string>& include_paths,
                                           const std::vector<std::string>& compile_options) {
    if (!context_ || source_code.empty() || function_name.empty()) {
        return false;
    }
    
    // Release old program and kernel if any
    if (kernel_) {
        clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
    
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }
    
    // Prepare sources
    const char* src = source_code.c_str();
    size_t length = source_code.size();
    
    // Create program
    cl_int error;
    program_ = clCreateProgramWithSource(context_, 1, &src, &length, &error);
    if (error != CL_SUCCESS) {
        return false;
    }
    
    // Build options
    std::string options;
    for (const auto& opt : compile_options) {
        options += opt + " ";
    }
    
    // Include paths
    for (const auto& path : include_paths) {
        options += "-I " + path + " ";
    }
    
    // Build program
    error = clBuildProgram(program_, 1, &device_, options.c_str(), nullptr, nullptr);
    if (error != CL_SUCCESS) {
        // Get build info to provide better error messages
        size_t log_size;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        
        return false;
    }
    
    // Create kernel
    kernel_ = clCreateKernel(program_, function_name.c_str(), &error);
    return error == CL_SUCCESS;
}

bool OpenCLBackend::load_kernel_from_binary(const std::string& binary_path,
                                           const std::string& function_name) {
    if (!context_ || binary_path.empty() || function_name.empty()) {
        return false;
    }
    
    // Release old program and kernel if any
    if (kernel_) {
        clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
    
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }
    
    try {
        // Read binary file
        std::vector<unsigned char> binary_data = read_binary_file(binary_path);
        if (binary_data.empty()) {
            return false;
        }
        
        // Create program from binary
        cl_int error;
        cl_int binary_status;
        const unsigned char* binary_ptr = binary_data.data();
        size_t binary_size = binary_data.size();
        
        program_ = clCreateProgramWithBinary(
            context_, 1, &device_, &binary_size, 
            &binary_ptr, &binary_status, &error);
        
        if (error != CL_SUCCESS || binary_status != CL_SUCCESS) {
            return false;
        }
        
        // Build program
        error = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
        if (error != CL_SUCCESS) {
            return false;
        }
        
        // Create kernel
        kernel_ = clCreateKernel(program_, function_name.c_str(), &error);
        return error == CL_SUCCESS;
    } catch (...) {
        return false;
    }
}

bool OpenCLBackend::launch_kernel(void* data, size_t event_count,
                                 int block_size, size_t shared_memory_size,
                                 int max_grid_size) {
    if (!kernel_ || !command_queue_ || !data) {
        return false;
    }
    
    try {
        // Set kernel arguments
        cl_mem cl_data = static_cast<cl_mem>(to_cl_mem(data));
        cl_int error;
        
        error = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &cl_data);
        if (error != CL_SUCCESS) return false;
        
        error = clSetKernelArg(kernel_, 1, sizeof(size_t), &event_count);
        if (error != CL_SUCCESS) return false;
        
        // Calculate work dimensions
        size_t global_work_size = ((event_count + block_size - 1) / block_size) * block_size;
        size_t local_work_size = block_size;
        
        // Apply max grid size if specified
        if (max_grid_size > 0 && global_work_size > static_cast<size_t>(max_grid_size) * block_size) {
            global_work_size = static_cast<size_t>(max_grid_size) * block_size;
        }
        
        // Launch kernel
        error = clEnqueueNDRangeKernel(
            command_queue_, kernel_, 1, nullptr,
            &global_work_size, &local_work_size,
            0, nullptr, nullptr);
        
        if (error != CL_SUCCESS) {
            return false;
        }
        
        // Wait for completion
        error = clFinish(command_queue_);
        return error == CL_SUCCESS;
    } catch (...) {
        return false;
    }
}

void* OpenCLBackend::create_stream() {
    if (!context_ || !device_) {
        return nullptr;
    }
    
    cl_int error;
    cl_command_queue queue = clCreateCommandQueue(context_, device_, 0, &error);
    if (error != CL_SUCCESS) {
        return nullptr;
    }
    
    return queue;
}

void OpenCLBackend::destroy_stream(void* stream) {
    if (stream) {
        clReleaseCommandQueue(static_cast<cl_command_queue>(stream));
    }
}

bool OpenCLBackend::launch_kernel_async(void* data, size_t event_count,
                                      int block_size, size_t shared_memory_size,
                                      int max_grid_size, void* stream) {
    if (!kernel_ || !stream || !data) {
        return false;
    }
    
    cl_command_queue queue = static_cast<cl_command_queue>(stream);
    
    try {
        // Set kernel arguments
        cl_mem cl_data = static_cast<cl_mem>(to_cl_mem(data));
        cl_int error;
        
        error = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &cl_data);
        if (error != CL_SUCCESS) return false;
        
        error = clSetKernelArg(kernel_, 1, sizeof(size_t), &event_count);
        if (error != CL_SUCCESS) return false;
        
        // Calculate work dimensions
        size_t global_work_size = ((event_count + block_size - 1) / block_size) * block_size;
        size_t local_work_size = block_size;
        
        // Apply max grid size if specified
        if (max_grid_size > 0 && global_work_size > static_cast<size_t>(max_grid_size) * block_size) {
            global_work_size = static_cast<size_t>(max_grid_size) * block_size;
        }
        
        // Launch kernel
        error = clEnqueueNDRangeKernel(
            queue, kernel_, 1, nullptr,
            &global_work_size, &local_work_size,
            0, nullptr, nullptr);
        
        return error == CL_SUCCESS;
    } catch (...) {
        return false;
    }
}

bool OpenCLBackend::synchronize_stream(void* stream) {
    if (!stream) {
        return false;
    }
    
    cl_int error = clFinish(static_cast<cl_command_queue>(stream));
    return error == CL_SUCCESS;
}

bool OpenCLBackend::synchronize_device() {
    if (!command_queue_) {
        return false;
    }
    
    cl_int error = clFinish(command_queue_);
    return error == CL_SUCCESS;
}

void OpenCLBackend::copy_host_to_device_async(void* dst, const void* src, size_t size, void* stream) {
    if (!stream || !dst || !src) {
        return;
    }
    
    cl_command_queue queue = static_cast<cl_command_queue>(stream);
    cl_mem cl_dst = static_cast<cl_mem>(to_cl_mem(dst));
    
    clEnqueueWriteBuffer(queue, cl_dst, CL_FALSE, 0, size, src, 0, nullptr, nullptr);
}

void OpenCLBackend::copy_device_to_host_async(void* dst, const void* src, size_t size, void* stream) {
    if (!stream || !dst || !src) {
        return;
    }
    
    cl_command_queue queue = static_cast<cl_command_queue>(stream);
    cl_mem cl_src = static_cast<cl_mem>(to_cl_mem(const_cast<void*>(src)));
    
    clEnqueueReadBuffer(queue, cl_src, CL_FALSE, 0, size, dst, 0, nullptr, nullptr);
}

void* OpenCLBackend::allocate_pinned_host_memory(size_t size) {
    if (!context_) {
        return nullptr;
    }
    
    cl_int error;
    cl_mem mem = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, nullptr, &error);
    if (error != CL_SUCCESS) {
        return nullptr;
    }
    
    void* ptr = clEnqueueMapBuffer(command_queue_, mem, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
                                  0, size, 0, nullptr, nullptr, &error);
    if (error != CL_SUCCESS) {
        clReleaseMemObject(mem);
        return nullptr;
    }
    
    // Store the mapping for cleanup
    pinned_memory_map_[ptr] = mem;
    
    return ptr;
}

void OpenCLBackend::free_pinned_host_memory(void* ptr) {
    if (!ptr) {
        return;
    }
    
    auto it = pinned_memory_map_.find(ptr);
    if (it != pinned_memory_map_.end()) {
        cl_mem mem = static_cast<cl_mem>(it->second);
        clEnqueueUnmapMemObject(command_queue_, mem, ptr, 0, nullptr, nullptr);
        clReleaseMemObject(mem);
        pinned_memory_map_.erase(it);
    }
}

bool OpenCLBackend::register_host_memory(void* ptr, size_t size, unsigned int flags) {
    // OpenCL doesn't have direct equivalent to CUDA's host memory registration
    // We'll have to emulate it by creating a buffer with CL_MEM_USE_HOST_PTR
    if (!ptr || size == 0 || !context_) {
        return false;
    }
    
    cl_int error;
    cl_mem mem = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, ptr, &error);
    if (error != CL_SUCCESS) {
        return false;
    }
    
    // Store in our memory objects map
    memory_objects_[ptr] = {mem, size};
    return true;
}

bool OpenCLBackend::unregister_host_memory(void* ptr) {
    if (!ptr) {
        return false;
    }
    
    auto it = memory_objects_.find(ptr);
    if (it != memory_objects_.end()) {
        clReleaseMemObject(it->second.mem);
        memory_objects_.erase(it);
        return true;
    }
    
    return false;
}

GpuDeviceInfo OpenCLBackend::get_device_info(int device_id) const {
    GpuDeviceInfo info;
    info.device_id = device_id;
    info.backend_type = BackendType::OpenCL;

    cl_int err;
    
    // Get device name
    char device_name[256];
    err = clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL device name");
    }
    info.name = device_name;
    
    // Get memory information
    cl_ulong global_mem_size;
    err = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL device memory");
    }
    info.total_memory = global_mem_size;
    info.available_memory = global_mem_size / 2; // Conservative estimate since OpenCL doesn't provide this directly
    
    // Get compute units
    cl_uint compute_units;
    err = clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL compute units");
    }
    info.num_cores = compute_units;
    
    // Get clock frequency
    cl_uint clock_freq;
    err = clGetDeviceInfo(device_, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_freq), &clock_freq, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL clock frequency");
    }
    info.clock_rate = clock_freq * 1000; // Convert MHz to kHz for consistency with CUDA
    
    // No compute capability for OpenCL devices
    info.compute_capability_major = 0;
    info.compute_capability_minor = 0;
    
    return info;
}

size_t OpenCLBackend::get_available_memory(int device_id) const {
    // Get available devices
    auto devices = get_available_devices();
    if (devices.empty() || static_cast<size_t>(device_id) >= devices.size()) {
        return 0;
    }
    
    cl_device_id device = devices[device_id];
    
    cl_ulong global_mem_size;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, nullptr);
    
    // OpenCL doesn't provide a direct way to query available memory
    // Return total memory as an approximation
    return global_mem_size;
}

void OpenCLBackend::check_cl_error(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        std::string error_message = std::string("OpenCL error in ") + operation + ": ";
        
        switch (error) {
            case CL_INVALID_CONTEXT: error_message += "CL_INVALID_CONTEXT"; break;
            case CL_INVALID_VALUE: error_message += "CL_INVALID_VALUE"; break;
            case CL_INVALID_BUFFER_SIZE: error_message += "CL_INVALID_BUFFER_SIZE"; break;
            case CL_INVALID_HOST_PTR: error_message += "CL_INVALID_HOST_PTR"; break;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE: error_message += "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
            case CL_OUT_OF_HOST_MEMORY: error_message += "CL_OUT_OF_HOST_MEMORY"; break;
            case CL_OUT_OF_RESOURCES: error_message += "CL_OUT_OF_RESOURCES"; break;
            default: error_message += "Unknown error: " + std::to_string(error);
        }
        
        throw std::runtime_error(error_message);
    }
}

std::vector<unsigned char> OpenCLBackend::read_binary_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return {};
    }
    
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<unsigned char> buffer(size);
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }
    
    return {};
}

#ifdef USE_OPENCL_BACKEND
// Register factory method
std::unique_ptr<GpuBackend> create_opencl_backend() {
    return std::make_unique<OpenCLBackend>();
}
#endif

} // namespace ebpf_gpu 