#include "kernel_loader.hpp"
#include "error_handling.hpp"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <stdexcept>

namespace ebpf_gpu {

// CudaModule implementation
CudaModule::CudaModule(const std::string& ptx_code) : module_(nullptr) {
    check_cuda_driver(cuModuleLoadData(&module_, ptx_code.c_str()), "cuModuleLoadData");
}

CudaModule::CudaModule(const std::vector<char>& ptx_data) : module_(nullptr) {
    check_cuda_driver(cuModuleLoadData(&module_, ptx_data.data()), "cuModuleLoadData");
}

CudaModule::~CudaModule() {
    if (module_) {
        cuModuleUnload(module_);
    }
}

CudaModule::CudaModule(CudaModule&& other) noexcept : module_(other.module_) {
    other.module_ = nullptr;
}

CudaModule& CudaModule::operator=(CudaModule&& other) noexcept {
    if (this != &other) {
        if (module_) {
            cuModuleUnload(module_);
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
    
    CUfunction function;
    check_cuda_driver(cuModuleGetFunction(&function, module_, function_name.c_str()),
                     "cuModuleGetFunction for " + function_name);
    return function;
}

// KernelLoader implementation
std::unique_ptr<CudaModule> KernelLoader::load_from_ptx(const std::string& ptx_code) const {
    if (ptx_code.empty()) {
        throw std::invalid_argument("PTX code cannot be empty");
    }
    
    return std::make_unique<CudaModule>(ptx_code);
}

std::unique_ptr<CudaModule> KernelLoader::load_from_file(const std::string& file_path) const {
    auto ptx_data = read_ptx_file(file_path);
    return std::make_unique<CudaModule>(ptx_data);
}

std::unique_ptr<CudaModule> KernelLoader::load_from_cuda_source(
    const std::string& cuda_source,
    const std::vector<std::string>& include_paths,
    const std::vector<std::string>& compile_options) const {
    
    std::string ptx_code = compile_cuda_to_ptx(cuda_source, include_paths, compile_options);
    return load_from_ptx(ptx_code);
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
    
    buffer[size] = '\0';  // Null terminate for PTX
    return buffer;
}

std::string KernelLoader::compile_cuda_to_ptx(
    const std::string& cuda_source,
    const std::vector<std::string>& include_paths,
    const std::vector<std::string>& compile_options) {
    
    // Create temporary files
    std::string temp_cu_file = "/tmp/temp_kernel_" + std::to_string(std::rand()) + ".cu";
    std::string temp_ptx_file = "/tmp/temp_kernel_" + std::to_string(std::rand()) + ".ptx";
    
    try {
        // Write CUDA source to temporary file
        std::ofstream cu_file(temp_cu_file);
        if (!cu_file.is_open()) {
            throw std::runtime_error("Cannot create temporary CUDA file");
        }
        cu_file << cuda_source;
        cu_file.close();
        
        // Build nvcc command
        std::ostringstream cmd;
        cmd << "nvcc -ptx";
        
        // Add include paths
        for (const auto& include_path : include_paths) {
            cmd << " -I" << include_path;
        }
        
        // Add compile options
        for (const auto& option : compile_options) {
            cmd << " " << option;
        }
        
        cmd << " " << temp_cu_file << " -o " << temp_ptx_file;
        
        // Execute compilation
        int result = std::system(cmd.str().c_str());
        if (result != 0) {
            throw std::runtime_error("CUDA compilation failed");
        }
        
        // Read PTX result
        auto ptx_data = read_file(temp_ptx_file);
        std::string ptx_code(ptx_data.begin(), ptx_data.end());
        
        // Cleanup temporary files
        std::remove(temp_cu_file.c_str());
        std::remove(temp_ptx_file.c_str());
        
        return ptx_code;
        
    } catch (...) {
        // Cleanup on error
        std::remove(temp_cu_file.c_str());
        std::remove(temp_ptx_file.c_str());
        throw;
    }
}

bool KernelLoader::validate_ptx(const std::string& ptx_code) {
    if (ptx_code.empty()) {
        return false;
    }
    
    // Basic PTX validation - check for PTX header
    return ptx_code.find(".version") != std::string::npos &&
           ptx_code.find(".target") != std::string::npos;
}

std::vector<char> KernelLoader::read_ptx_file(const std::string& file_path) const {
    return read_file(file_path);
}

} // namespace ebpf_gpu 