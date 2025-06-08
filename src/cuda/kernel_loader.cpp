#include "../../include/kernel_loader.hpp"
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <filesystem>
#include <memory>

namespace ebpf_gpu {

// Return the current backend type
BackendType get_backend_type() {
    return BackendType::CUDA;
}

// GpuModule implementation for CUDA
GpuModule::GpuModule(const std::string& ir_code) : backend_type_(BackendType::CUDA), module_handle_(nullptr) {
    initialize_from_ir(ir_code);
}

GpuModule::GpuModule(const std::vector<char>& ir_data) : backend_type_(BackendType::CUDA), module_handle_(nullptr) {
    if (ir_data.empty()) {
        throw std::runtime_error("Empty PTX data");
    }

    std::string ir_code(ir_data.begin(), ir_data.end());
    initialize_from_ir(ir_code);
}

void GpuModule::initialize_from_ir(const std::string& ir_code) {
    if (ir_code.empty()) {
        throw std::runtime_error("Empty PTX code");
    }

    CUresult result = cuModuleLoadData(reinterpret_cast<CUmodule*>(&module_handle_), ir_code.c_str());
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        throw std::runtime_error("Failed to load CUDA module: " + std::string(error_str));
    }
}

GpuModule::~GpuModule() {
    cleanup();
}

void GpuModule::cleanup() {
    if (module_handle_) {
        cuModuleUnload(static_cast<CUmodule>(module_handle_));
        module_handle_ = nullptr;
    }
    
    // Clear function cache
    for (auto& pair : function_cache_) {
        pair.second = nullptr;
    }
    function_cache_.clear();
}

GpuModule::GpuModule(GpuModule&& other) noexcept 
    : backend_type_(other.backend_type_),
      module_handle_(other.module_handle_), 
      function_cache_(std::move(other.function_cache_)) {
    other.module_handle_ = nullptr;
}

GpuModule& GpuModule::operator=(GpuModule&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        backend_type_ = other.backend_type_;
        module_handle_ = other.module_handle_;
        function_cache_ = std::move(other.function_cache_);
        
        other.module_handle_ = nullptr;
    }
    return *this;
}

GenericHandle GpuModule::get() const noexcept {
    return module_handle_;
}

FunctionHandle GpuModule::get_function(const std::string& function_name) const {
    if (!is_valid()) {
        throw std::runtime_error("Module is not loaded");
    }
    
    // Check cache first
    auto it = function_cache_.find(function_name);
    if (it != function_cache_.end()) {
        return it->second;
    }
    
    CUfunction function;
    CUresult result = cuModuleGetFunction(&function, static_cast<CUmodule>(module_handle_), function_name.c_str());
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        throw std::runtime_error("Failed to get function '" + function_name + "': " + error_str);
    }

    // Cache the function
    function_cache_[function_name] = function;
    return function;
}

bool GpuModule::is_valid() const noexcept {
    return module_handle_ != nullptr;
}

// KernelLoader implementation for CUDA
std::unique_ptr<GpuModule> KernelLoader::load_from_ir(const std::string& ir_code) const {
    if (ir_code.empty()) {
        throw std::runtime_error("Empty PTX code");
    }
    
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
    
    if (source_code.empty()) {
        throw std::runtime_error("Empty source code provided");
    }
    
    // Create temporary files
    std::string temp_dir = "/tmp";
    std::string temp_cu_file = temp_dir + "/temp_kernel_" + std::to_string(std::rand()) + ".cu";
    std::string temp_ptx_file = temp_dir + "/temp_kernel_" + std::to_string(std::rand()) + ".ptx";
    
    try {
        // Write source code to temporary .cu file
        std::ofstream cu_file(temp_cu_file);
        if (!cu_file.is_open()) {
            throw std::runtime_error("Failed to create temporary source file: " + temp_cu_file);
        }
        cu_file << source_code;
        cu_file.close();
        
        // Build nvcc command
        std::stringstream cmd;
        cmd << "nvcc --ptx";
        
        // Add include paths
        for (const auto& include_path : include_paths) {
            cmd << " -I\"" << include_path << "\"";
        }
        
        // Add custom compile options
        for (const auto& option : compile_options) {
            cmd << " " << option;
        }
        
        // Add default options for better compatibility
        cmd << " --gpu-architecture=compute_50";
        cmd << " --std=c++14";
        
        // Input and output files
        cmd << " \"" << temp_cu_file << "\"";
        cmd << " -o \"" << temp_ptx_file << "\"";
        
        // Redirect stderr to stdout for better error capture
        cmd << " 2>&1";
        
        std::string command = cmd.str();
        std::cout << "Executing: " << command << std::endl;
        
        // Execute nvcc command
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("Failed to execute nvcc command");
        }
        
        // Capture command output
        std::string output;
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
            output += buffer;
        }
        
        // Check if nvcc succeeded
        int exit_code = pclose(pipe.release());
        if (exit_code != 0) {
            // Clean up temp files before throwing
            std::remove(temp_cu_file.c_str());
            std::remove(temp_ptx_file.c_str());
            throw std::runtime_error("nvcc compilation failed with exit code " + std::to_string(exit_code) + ":\n" + output);
        }
        
        // Print any warnings/info from nvcc
        if (!output.empty()) {
            std::cout << "nvcc output:\n" << output << std::endl;
        }
        
        // Read the generated PTX file
        std::ifstream ptx_file(temp_ptx_file, std::ios::binary | std::ios::ate);
        if (!ptx_file.is_open()) {
            // Clean up temp files before throwing
            std::remove(temp_cu_file.c_str());
            std::remove(temp_ptx_file.c_str());
            throw std::runtime_error("Failed to open generated PTX file: " + temp_ptx_file);
        }
        
        std::streamsize ptx_size = ptx_file.tellg();
        ptx_file.seekg(0, std::ios::beg);
        
        std::string ptx_code(ptx_size, '\0');
        if (!ptx_file.read(&ptx_code[0], ptx_size)) {
            ptx_file.close();
            std::remove(temp_cu_file.c_str());
            std::remove(temp_ptx_file.c_str());
            throw std::runtime_error("Failed to read PTX file");
        }
        ptx_file.close();
        
        // Clean up temporary files
        std::remove(temp_cu_file.c_str());
        std::remove(temp_ptx_file.c_str());
        
        return ptx_code;
        
    } catch (const std::exception& e) {
        // Clean up temporary files in case of any error
        std::remove(temp_cu_file.c_str());
        std::remove(temp_ptx_file.c_str());
        throw;
    }
}

bool KernelLoader::validate_ir(const std::string& ir_code) {
    // Basic validation for PTX code
    return !ir_code.empty() && 
           (ir_code.find(".version") != std::string::npos || 
            ir_code.find(".target") != std::string::npos);
}

} // namespace ebpf_gpu 