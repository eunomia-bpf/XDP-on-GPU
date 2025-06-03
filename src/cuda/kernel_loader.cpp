#ifdef USE_CUDA_BACKEND

#include "kernel_loader.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ebpf_gpu {

// Return the current backend type
BackendType get_backend_type() {
    return BackendType::CUDA;
}

// GpuModule implementation for CUDA
GpuModule::GpuModule(const std::string& ir_code) : module_(nullptr) {
    if (ir_code.empty()) {
        throw std::runtime_error("Empty PTX code");
    }

    CUresult result = cuModuleLoadData(&module_, ir_code.c_str());
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        throw std::runtime_error("Failed to load CUDA module: " + std::string(error_str));
    }
}

GpuModule::GpuModule(const std::vector<char>& ir_data) : module_(nullptr) {
    if (ir_data.empty()) {
        throw std::runtime_error("Empty PTX data");
    }

    CUresult result = cuModuleLoadData(&module_, ir_data.data());
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        throw std::runtime_error("Failed to load CUDA module: " + std::string(error_str));
    }
}

GpuModule::~GpuModule() {
    if (module_) {
        cuModuleUnload(module_);
    }
}

GpuModule::GpuModule(GpuModule&& other) noexcept : module_(other.module_) {
    other.module_ = nullptr;
}

GpuModule& GpuModule::operator=(GpuModule&& other) noexcept {
    if (this != &other) {
        if (module_) {
            cuModuleUnload(module_);
        }
        module_ = other.module_;
        other.module_ = nullptr;
    }
    return *this;
}

CUfunction GpuModule::get_function(const std::string& function_name) const {
    if (!module_) {
        throw std::runtime_error("Module is not loaded");
    }

    CUfunction function;
    CUresult result = cuModuleGetFunction(&function, module_, function_name.c_str());
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        throw std::runtime_error("Failed to get function '" + function_name + "': " + error_str);
    }

    return function;
}

// KernelLoader implementation for CUDA
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
        std::string ptx = compile_source_to_ir(source_code, include_paths, compile_options);
        return load_from_ir(ptx);
    } catch (const std::exception& e) {
        std::cerr << "Failed to compile source: " << e.what() << std::endl;
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
    
    std::vector<char> buffer(size + 1);  // +1 for null terminator
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read file: " + file_path);
    }
    
    buffer[size] = '\0';  // Null terminate for string handling
    return buffer;
}

std::string KernelLoader::compile_source_to_ir(
    const std::string& source_code,
    const std::vector<std::string>& include_paths,
    const std::vector<std::string>& compile_options) {
    // TODO: Implement NVRTC compilation from CUDA source to PTX
    // For now, we'll assume the source is already PTX
    std::cerr << "Warning: CUDA source to PTX compilation not implemented. Assuming input is already PTX." << std::endl;
    return source_code;
}

bool KernelLoader::validate_ir(const std::string& ir_code) {
    // Basic validation for PTX code
    return !ir_code.empty() && 
           (ir_code.find(".version") != std::string::npos || 
            ir_code.find(".target") != std::string::npos);
}

} // namespace ebpf_gpu

#endif // USE_CUDA_BACKEND 