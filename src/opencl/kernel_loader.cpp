#include "../../include/kernel_loader.hpp"
#include <CL/cl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace ebpf_gpu {

// Helper function to get OpenCL device and context
static bool get_opencl_device_and_context(cl_device_id& device, cl_context& context) {
    cl_int err;
    
    // Get the platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "OpenCL: No platforms found" << std::endl;
        return false;
    }
    
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to get platform IDs: " << err << std::endl;
        return false;
    }
    
    // Use the first platform and get GPU devices
    cl_platform_id platform = platforms[0];
    cl_uint num_devices;
    
    // Try to get GPU devices first
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        // If no GPU devices, try to get CPU devices
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            std::cerr << "OpenCL: No GPU or CPU devices found" << std::endl;
            return false;
        }
    }
    
    // Get the device IDs
    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_devices, devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "OpenCL: Failed to get device IDs: " << err << std::endl;
            return false;
        }
    }
    
    // Use the first device
    device = devices[0];
    
    // Create an OpenCL context
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0
    };
    
    context = clCreateContext(properties, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to create context: " << err << std::endl;
        return false;
    }
    
    return true;
}

// Return the current backend type
BackendType get_backend_type() {
    return BackendType::OpenCL;
}

// GpuModule implementation for OpenCL
GpuModule::GpuModule(const std::string& ir_code) 
    : backend_type_(BackendType::OpenCL), module_handle_(nullptr), 
      context_handle_(nullptr), device_handle_(nullptr) {
    initialize_from_ir(ir_code);
}

GpuModule::GpuModule(const std::vector<char>& ir_data) 
    : backend_type_(BackendType::OpenCL), module_handle_(nullptr), 
      context_handle_(nullptr), device_handle_(nullptr) {
    if (ir_data.empty()) {
        throw std::runtime_error("Empty IR data");
    }
    
    std::string ir_code(ir_data.begin(), ir_data.end());
    initialize_from_ir(ir_code);
}

void GpuModule::initialize_from_ir(const std::string& ir_code) {
    cl_int err;
    
    if (ir_code.empty()) {
        throw std::runtime_error("Empty OpenCL code");
    }
    
    // Get device and context
    cl_device_id device;
    cl_context context;
    if (!get_opencl_device_and_context(device, context)) {
        throw std::runtime_error("Failed to get OpenCL device and context");
    }
    
    // Store handles for cleanup
    device_handle_ = device;
    context_handle_ = context;
    
    // Create program from source
    const char* source_ptr = ir_code.c_str();
    size_t source_len = ir_code.length();
    
    cl_program program = clCreateProgramWithSource(context, 1, &source_ptr, &source_len, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to create program from source: " << err << std::endl;
        clReleaseContext(context);
        throw std::runtime_error("Failed to create OpenCL program");
    }
    
    // Build the program
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // If there was a build error, get the build log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        
        std::vector<char> log(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        log[log_size] = '\0';
        
        std::cerr << "OpenCL: Program build failed: " << log.data() << std::endl;
        clReleaseProgram(program);
        clReleaseContext(context);
        throw std::runtime_error("Failed to build OpenCL program");
    }
    
    module_handle_ = program;
}

GpuModule::~GpuModule() {
    cleanup();
}

void GpuModule::cleanup() {
    // Release all kernels in the cache
    for (auto& pair : function_cache_) {
        if (pair.second) {
            clReleaseKernel(static_cast<cl_kernel>(pair.second));
        }
    }
    function_cache_.clear();
    
    // Release program
    if (module_handle_) {
        clReleaseProgram(static_cast<cl_program>(module_handle_));
        module_handle_ = nullptr;
    }
    
    // Release context
    if (context_handle_) {
        clReleaseContext(static_cast<cl_context>(context_handle_));
        context_handle_ = nullptr;
    }
    
    // No need to release device, it's managed by OpenCL runtime
    device_handle_ = nullptr;
}

GpuModule::GpuModule(GpuModule&& other) noexcept
    : backend_type_(other.backend_type_),
      module_handle_(other.module_handle_),
      context_handle_(other.context_handle_),
      device_handle_(other.device_handle_),
      function_cache_(std::move(other.function_cache_)) {
    other.module_handle_ = nullptr;
    other.context_handle_ = nullptr;
    other.device_handle_ = nullptr;
}

GpuModule& GpuModule::operator=(GpuModule&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        backend_type_ = other.backend_type_;
        module_handle_ = other.module_handle_;
        context_handle_ = other.context_handle_;
        device_handle_ = other.device_handle_;
        function_cache_ = std::move(other.function_cache_);
        
        other.module_handle_ = nullptr;
        other.context_handle_ = nullptr;
        other.device_handle_ = nullptr;
    }
    return *this;
}

GenericHandle GpuModule::get() const noexcept {
    return module_handle_;
}

bool GpuModule::is_valid() const noexcept {
    return module_handle_ != nullptr;
}

FunctionHandle GpuModule::get_function(const std::string& function_name) const {
    if (!is_valid()) {
        throw std::runtime_error("Invalid OpenCL program");
    }
    
    // Check cache first
    auto it = function_cache_.find(function_name);
    if (it != function_cache_.end()) {
        return it->second;
    }
    
    cl_int err;
    cl_kernel kernel = clCreateKernel(static_cast<cl_program>(module_handle_), function_name.c_str(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to create kernel '" << function_name << "': " << err << std::endl;
        return nullptr;
    }
    
    // Store the kernel in cache
    function_cache_[function_name] = kernel;
    
    return kernel;
}

// KernelLoader implementation for OpenCL
std::unique_ptr<GpuModule> KernelLoader::load_from_ir(const std::string& ir_code) const {
    try {
        return std::make_unique<GpuModule>(ir_code);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load IR: " << e.what() << std::endl;
        return nullptr;
    }
}

std::unique_ptr<GpuModule> KernelLoader::load_from_file(const std::string& file_path) const {
    try {
        std::vector<char> ir_data = read_ir_file(file_path);
        return std::make_unique<GpuModule>(ir_data);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load file " << file_path << ": " << e.what() << std::endl;
        return nullptr;
    }
}

std::unique_ptr<GpuModule> KernelLoader::load_from_source(
    const std::string& source_code,
    const std::vector<std::string>& include_paths,
    const std::vector<std::string>& compile_options) const {
    try {
        // For OpenCL, we can directly use the source code as IR
        return load_from_ir(source_code);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load source: " << e.what() << std::endl;
        return nullptr;
    }
}

std::vector<char> KernelLoader::read_ir_file(const std::string& file_path) const {
    return read_file(file_path);
}

std::vector<char> KernelLoader::read_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read file: " + file_path);
    }
    
    return buffer;
}

std::string KernelLoader::compile_source_to_ir(
    const std::string& source_code,
    const std::vector<std::string>& include_paths,
    const std::vector<std::string>& compile_options) {
    // For OpenCL, we can use the source directly
    return source_code;
}

bool KernelLoader::validate_ir(const std::string& ir_code) {
    // Basic validation - check if it contains some OpenCL keywords
    return !ir_code.empty() && 
           (ir_code.find("__kernel") != std::string::npos || 
            ir_code.find("kernel") != std::string::npos);
}

} // namespace ebpf_gpu 