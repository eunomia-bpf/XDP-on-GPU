#ifdef USE_OPENCL_BACKEND

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
    : program_(nullptr), context_(nullptr), device_(nullptr) {
    cl_int err;
    
    // Get device and context
    if (!get_opencl_device_and_context(device_, context_)) {
        throw std::runtime_error("Failed to get OpenCL device and context");
    }
    
    // Create program from source
    const char* source_ptr = ir_code.c_str();
    size_t source_len = ir_code.length();
    
    program_ = clCreateProgramWithSource(context_, 1, &source_ptr, &source_len, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to create program from source: " << err << std::endl;
        throw std::runtime_error("Failed to create OpenCL program");
    }
    
    // Build the program
    err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // If there was a build error, get the build log
        size_t log_size;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        
        std::vector<char> log(log_size + 1);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        log[log_size] = '\0';
        
        std::cerr << "OpenCL: Program build failed: " << log.data() << std::endl;
        clReleaseProgram(program_);
        program_ = nullptr;
        throw std::runtime_error("Failed to build OpenCL program");
    }
}

GpuModule::GpuModule(const std::vector<char>& ir_data) 
    : program_(nullptr), context_(nullptr), device_(nullptr) {
    if (ir_data.empty()) {
        throw std::runtime_error("Empty IR data");
    }
    
    std::string ir_code(ir_data.begin(), ir_data.end());
    *this = GpuModule(ir_code);
}

GpuModule::~GpuModule() {
    for (auto kernel : kernels_) {
        if (kernel) {
            clReleaseKernel(kernel);
        }
    }
    
    if (program_) {
        clReleaseProgram(program_);
    }
    
    if (context_) {
        clReleaseContext(context_);
    }
}

GpuModule::GpuModule(GpuModule&& other) noexcept
    : program_(other.program_), context_(other.context_), device_(other.device_), 
      kernels_(std::move(other.kernels_)) {
    other.program_ = nullptr;
    other.context_ = nullptr;
    other.device_ = nullptr;
}

GpuModule& GpuModule::operator=(GpuModule&& other) noexcept {
    if (this != &other) {
        // Clean up existing resources
        for (auto kernel : kernels_) {
            if (kernel) {
                clReleaseKernel(kernel);
            }
        }
        
        if (program_) {
            clReleaseProgram(program_);
        }
        
        if (context_) {
            clReleaseContext(context_);
        }
        
        // Move resources from other
        program_ = other.program_;
        context_ = other.context_;
        device_ = other.device_;
        kernels_ = std::move(other.kernels_);
        
        // Reset other
        other.program_ = nullptr;
        other.context_ = nullptr;
        other.device_ = nullptr;
    }
    return *this;
}

cl_kernel GpuModule::get_function(const std::string& function_name) const {
    if (!is_valid()) {
        throw std::runtime_error("Invalid OpenCL program");
    }
    
    cl_int err;
    cl_kernel kernel = clCreateKernel(program_, function_name.c_str(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to create kernel '" << function_name << "': " << err << std::endl;
        return nullptr;
    }
    
    // Store the kernel for cleanup
    kernels_.push_back(kernel);
    
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

#endif // USE_OPENCL_BACKEND 