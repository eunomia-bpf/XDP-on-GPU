#pragma once

#include <string>
#include <memory>
#include <vector>

// Include appropriate headers based on backend
#ifdef USE_CUDA_BACKEND
#include <cuda.h>
#endif

#ifdef USE_OPENCL_BACKEND
#include <CL/cl.h>
#endif

namespace ebpf_gpu {

// Define backend type
enum class BackendType {
    Unknown = 0,
    CUDA = 1,
    OpenCL = 2
};

// Get the current backend type
BackendType get_backend_type();

// Forward declarations
class GpuModule;

#ifdef USE_CUDA_BACKEND
// CUDA-specific implementation
class GpuModule {
public:
    explicit GpuModule(const std::string& ir_code);
    explicit GpuModule(const std::vector<char>& ir_data);
    ~GpuModule();
    
    // Non-copyable, movable
    GpuModule(const GpuModule&) = delete;
    GpuModule& operator=(const GpuModule&) = delete;
    GpuModule(GpuModule&& other) noexcept;
    GpuModule& operator=(GpuModule&& other) noexcept;
    
    CUmodule get() const noexcept { return module_; }
    CUfunction get_function(const std::string& function_name) const;
    
    bool is_valid() const noexcept { return module_ != nullptr; }

private:
    CUmodule module_;
};
#elif defined(USE_OPENCL_BACKEND)
// OpenCL-specific implementation
class GpuModule {
public:
    explicit GpuModule(const std::string& ir_code);
    explicit GpuModule(const std::vector<char>& ir_data);
    ~GpuModule();
    
    // Non-copyable, movable
    GpuModule(const GpuModule&) = delete;
    GpuModule& operator=(const GpuModule&) = delete;
    GpuModule(GpuModule&& other) noexcept;
    GpuModule& operator=(GpuModule&& other) noexcept;
    
    cl_program get() const noexcept { return program_; }
    cl_kernel get_function(const std::string& function_name) const;
    
    bool is_valid() const noexcept { return program_ != nullptr; }

private:
    cl_program program_;
    cl_context context_;
    cl_device_id device_;
    mutable std::vector<cl_kernel> kernels_; // Cache for created kernels
};
#else
// Fallback implementation for when no backend is defined
class GpuModule {
public:
    explicit GpuModule(const std::string& ir_code) {}
    explicit GpuModule(const std::vector<char>& ir_data) {}
    ~GpuModule() {}
    
    GpuModule(const GpuModule&) = delete;
    GpuModule& operator=(const GpuModule&) = delete;
    GpuModule(GpuModule&& other) noexcept = default;
    GpuModule& operator=(GpuModule&& other) noexcept = default;
    
    void* get_function(const std::string& function_name) const { return nullptr; }
    bool is_valid() const noexcept { return false; }
};
#endif

class KernelLoader {
public:
    KernelLoader() = default;
    ~KernelLoader() = default;
    
    // Non-copyable, movable
    KernelLoader(const KernelLoader&) = delete;
    KernelLoader& operator=(const KernelLoader&) = delete;
    KernelLoader(KernelLoader&&) = default;
    KernelLoader& operator=(KernelLoader&&) = default;
    
    // Get the backend type
    BackendType get_backend() const { return get_backend_type(); }
    
    // Load from IR string (PTX for CUDA, SPIR/OpenCL C for OpenCL)
    std::unique_ptr<GpuModule> load_from_ir(const std::string& ir_code) const;
    
    // Load from IR file
    std::unique_ptr<GpuModule> load_from_file(const std::string& file_path) const;
    
    // Load from source (CUDA or OpenCL C)
    std::unique_ptr<GpuModule> load_from_source(
        const std::string& source_code,
        const std::vector<std::string>& include_paths = {},
        const std::vector<std::string>& compile_options = {}) const;
    
    // Utility functions
    static std::vector<char> read_file(const std::string& file_path);
    
    // Compile source to IR
    static std::string compile_source_to_ir(
        const std::string& source_code,
        const std::vector<std::string>& include_paths = {},
        const std::vector<std::string>& compile_options = {});
    
    // Validate IR code
    static bool validate_ir(const std::string& ir_code);
    
private:
    std::vector<char> read_ir_file(const std::string& file_path) const;
};

// Compatibility typedefs for backward compatibility
#ifdef USE_CUDA_BACKEND
using CudaModule = GpuModule;
#endif

} // namespace ebpf_gpu 