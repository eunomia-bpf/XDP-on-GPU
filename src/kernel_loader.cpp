#include "../include/kernel_loader.hpp"
#include <fstream>
#include <vector>
#include <regex>
#include <string>
#include <stdexcept>

#ifdef USE_CUDA_BACKEND
#include <cuda.h>
#endif

namespace ebpf_gpu {

// Helper function to check if string ends with a suffix
bool ends_with(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

// Helper class for CUDA module management
// Only defined if CUDA backend is enabled
#ifdef USE_CUDA_BACKEND
class CudaModule {
public:
    CudaModule(const std::string& ptx) {
        // Load PTX code into CUDA module
        CUresult result = cuModuleLoadData(&module_, ptx.c_str());
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to load PTX code into CUDA module");
        }
    }
    
    CudaModule(const std::vector<char>& ptx) {
        // Load PTX code into CUDA module
        CUresult result = cuModuleLoadData(&module_, ptx.data());
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to load PTX code into CUDA module");
        }
    }
    
    ~CudaModule() {
        if (module_) {
            cuModuleUnload(module_);
        }
    }
    
    // Non-copyable
    CudaModule(const CudaModule&) = delete;
    CudaModule& operator=(const CudaModule&) = delete;
    
    // Movable
    CudaModule(CudaModule&& other) noexcept : module_(other.module_) {
        other.module_ = nullptr;
    }
    
    CudaModule& operator=(CudaModule&& other) noexcept {
        if (this != &other) {
            if (module_) {
                cuModuleUnload(module_);
            }
            module_ = other.module_;
            other.module_ = nullptr;
        }
        return *this;
    }
    
    CUfunction get_function(const std::string& name) const {
        CUfunction function;
        CUresult result = cuModuleGetFunction(&function, module_, name.c_str());
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get function from CUDA module: " + name);
        }
        return function;
    }
    
    operator bool() const {
        return module_ != nullptr;
    }
    
private:
    CUmodule module_ = nullptr;
};
#else
// Stub CudaModule for when CUDA is not available
class CudaModule {
public:
    CudaModule(const std::string&) {
        throw std::runtime_error("CUDA backend not available");
    }
    
    CudaModule(const std::vector<char>&) {
        throw std::runtime_error("CUDA backend not available");
    }
    
    void* get_function(const std::string&) const {
        throw std::runtime_error("CUDA backend not available");
        return nullptr;
    }
    
    operator bool() const {
        return false;
    }
};
#endif

// KernelLoader implementation

KernelLoader::KernelLoader() = default;
KernelLoader::~KernelLoader() = default;

bool KernelLoader::validate_ptx(const std::string& ptx) {
    // Basic validation: Check for .version, .target, and .entry in the PTX code
    if (ptx.empty()) {
        return false;
    }
    
    std::regex version_pattern(R"(\.version\s+\d+\.\d+)");
    std::regex target_pattern(R"(\.target\s+\w+)");
    std::regex entry_pattern(R"(\.entry\s+\w+)");
    
    return std::regex_search(ptx, version_pattern) &&
           std::regex_search(ptx, target_pattern) &&
           std::regex_search(ptx, entry_pattern);
}

std::vector<char> KernelLoader::read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read file: " + path);
    }
    
    return buffer;
}

void KernelLoader::load_from_ptx(const std::string& ptx) {
    if (!validate_ptx(ptx)) {
        throw std::invalid_argument("Invalid PTX code");
    }
    
    // Only try to create a CUDA module if CUDA backend is enabled
#ifdef USE_CUDA_BACKEND
    try {
        module_ = std::make_unique<CudaModule>(ptx);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load PTX: " + std::string(e.what()));
    }
#else
    throw std::runtime_error("CUDA backend not available");
#endif
}

void KernelLoader::load_from_file(const std::string& path) {
    auto content = read_file(path);
    
    // Check file extension to determine type
    if (ends_with(path, ".ptx")) {
        std::string ptx_code(content.begin(), content.end());
        load_from_ptx(ptx_code);
    } else {
        throw std::runtime_error("Unsupported file format: " + path);
    }
}

void* KernelLoader::get_kernel(const std::string& name) const {
#ifdef USE_CUDA_BACKEND
    if (module_) {
        try {
            return (void*)module_->get_function(name);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to get kernel function: " + std::string(e.what()));
        }
    }
#endif

    throw std::runtime_error("No kernel module loaded or CUDA backend not available");
}

} // namespace ebpf_gpu 