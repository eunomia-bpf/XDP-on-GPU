#include "../../include/kernel_loader.hpp"
#include <CL/cl.h>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace ebpf_gpu {

// CudaModule implementation for OpenCL
CudaModule::CudaModule(const std::string& ptx_code) : module_(nullptr) {
    // In OpenCL, we'll use the ptx_code parameter as OpenCL C source code
    cl_int err;
    cl_platform_id platform;
    
    // Get OpenCL platform
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL platform");
    }
    
    // Get device
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL GPU device");
    }
    
    // Create context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL context");
    }
    
    // Create program from source
    const char* source_ptr = ptx_code.c_str();
    size_t source_size = ptx_code.size();
    cl_program program = clCreateProgramWithSource(context, 1, &source_ptr, &source_size, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context);
        throw std::runtime_error("Failed to create OpenCL program");
    }
    
    // Build program
    err = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build error log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        
        std::string error_msg = "OpenCL program build failed: ";
        error_msg.append(log.data(), log_size);
        
        clReleaseProgram(program);
        clReleaseContext(context);
        throw std::runtime_error(error_msg);
    }
    
    // Store the program, context, and device in a custom struct
    struct OpenCLModule {
        cl_program program;
        cl_context context;
        cl_device_id device;
    };
    
    OpenCLModule* ocl_module = new OpenCLModule{program, context, device};
    module_ = reinterpret_cast<CUmodule>(ocl_module);
}

CudaModule::CudaModule(const std::vector<char>& ptx_data) : module_(nullptr) {
    // Convert vector<char> to string and delegate to the string constructor
    std::string source_code(ptx_data.begin(), ptx_data.end());
    *this = CudaModule(source_code); // Use string constructor
}

CudaModule::~CudaModule() {
    if (module_) {
        struct OpenCLModule {
            cl_program program;
            cl_context context;
            cl_device_id device;
        };
        
        auto* ocl_module = reinterpret_cast<OpenCLModule*>(module_);
        clReleaseProgram(ocl_module->program);
        clReleaseContext(ocl_module->context);
        delete ocl_module;
    }
}

CudaModule::CudaModule(CudaModule&& other) noexcept : module_(other.module_) {
    other.module_ = nullptr;
}

CudaModule& CudaModule::operator=(CudaModule&& other) noexcept {
    if (this != &other) {
        if (module_) {
            struct OpenCLModule {
                cl_program program;
                cl_context context;
                cl_device_id device;
            };
            
            auto* ocl_module = reinterpret_cast<OpenCLModule*>(module_);
            clReleaseProgram(ocl_module->program);
            clReleaseContext(ocl_module->context);
            delete ocl_module;
        }
        module_ = other.module_;
        other.module_ = nullptr;
    }
    return *this;
}

CUfunction CudaModule::get_function(const std::string& function_name) const {
    if (!module_) {
        throw std::runtime_error("Module is not loaded");
    }
    
    struct OpenCLModule {
        cl_program program;
        cl_context context;
        cl_device_id device;
    };
    
    auto* ocl_module = reinterpret_cast<OpenCLModule*>(module_);
    
    cl_int err;
    cl_kernel kernel = clCreateKernel(ocl_module->program, function_name.c_str(), &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL kernel: " + function_name);
    }
    
    // For OpenCL, we'll return the kernel as the function pointer
    return reinterpret_cast<CUfunction>(kernel);
}

// KernelLoader implementation
std::unique_ptr<CudaModule> KernelLoader::load_from_ptx(const std::string& ptx_code) const {
    if (ptx_code.empty()) {
        throw std::invalid_argument("OpenCL source code cannot be empty");
    }
    
    return std::make_unique<CudaModule>(ptx_code);
}

std::unique_ptr<CudaModule> KernelLoader::load_from_file(const std::string& file_path) const {
    auto source_data = read_file(file_path);
    return std::make_unique<CudaModule>(source_data);
}

std::unique_ptr<CudaModule> KernelLoader::load_from_cuda_source(
    const std::string& cuda_source,
    const std::vector<std::string>& include_paths,
    const std::vector<std::string>& compile_options) const {
    
    // For OpenCL, we'll just pass through the source code directly
    // We don't perform any source translation from CUDA to OpenCL in this simple implementation
    return load_from_ptx(cuda_source);
}

std::vector<char> KernelLoader::read_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size + 1);  // +1 for null terminator
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read file: " + file_path);
    }
    
    buffer[size] = '\0';  // Null terminate for string handling
    return buffer;
}

std::string KernelLoader::compile_cuda_to_ptx(
    const std::string& cuda_source,
    const std::vector<std::string>& include_paths,
    const std::vector<std::string>& compile_options) {
    
    // For OpenCL, we just return the source code directly
    // No CUDA-to-PTX compilation is performed
    return cuda_source;
}

bool KernelLoader::validate_ptx(const std::string& ptx_code) {
    if (ptx_code.empty()) {
        return false;
    }
    
    // Basic validation for OpenCL source
    // Check for OpenCL kernel function declaration
    return ptx_code.find("__kernel") != std::string::npos ||
           ptx_code.find("kernel") != std::string::npos;
}

std::vector<char> KernelLoader::read_ptx_file(const std::string& file_path) const {
    return read_file(file_path);
}

} // namespace ebpf_gpu 