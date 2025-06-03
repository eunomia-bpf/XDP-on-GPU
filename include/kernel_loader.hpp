#pragma once

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>

namespace ebpf_gpu {

// Define backend type
enum class BackendType {
    Unknown = 0,
    CUDA = 1,
    OpenCL = 2
};

// Get the current backend type
BackendType get_backend_type();

// Forward declaration for backend-specific handles
using GenericHandle = void*;
using FunctionHandle = void*;

// Unified GPU Module class
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
    
    GenericHandle get() const noexcept;
    FunctionHandle get_function(const std::string& function_name) const;
    
    bool is_valid() const noexcept;

private:
    BackendType backend_type_;
    GenericHandle module_handle_ = nullptr;
    mutable std::unordered_map<std::string, FunctionHandle> function_cache_;
    
    // Context handles for OpenCL
    GenericHandle context_handle_ = nullptr;
    GenericHandle device_handle_ = nullptr;
    
    void cleanup();
    void initialize_from_ir(const std::string& ir_code);
};

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

} // namespace ebpf_gpu 