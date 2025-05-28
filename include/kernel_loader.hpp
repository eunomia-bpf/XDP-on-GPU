#pragma once

#include <string>
#include <memory>
#include <vector>
#include <cuda.h>

namespace ebpf_gpu {

class CudaModule {
public:
    explicit CudaModule(const std::string& ptx_code);
    explicit CudaModule(const std::vector<char>& ptx_data);
    ~CudaModule();
    
    // Non-copyable, movable
    CudaModule(const CudaModule&) = delete;
    CudaModule& operator=(const CudaModule&) = delete;
    CudaModule(CudaModule&& other) noexcept;
    CudaModule& operator=(CudaModule&& other) noexcept;
    
    CUmodule get() const noexcept { return module_; }
    CUfunction get_function(const std::string& function_name) const;
    
    bool is_valid() const noexcept { return module_ != nullptr; }

private:
    CUmodule module_;
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
    
    // Load from PTX string
    std::unique_ptr<CudaModule> load_from_ptx(const std::string& ptx_code) const;
    
    // Load from PTX file
    std::unique_ptr<CudaModule> load_from_file(const std::string& file_path) const;
    
    // Load from CUDA source (compile to PTX first)
    std::unique_ptr<CudaModule> load_from_cuda_source(
        const std::string& cuda_source,
        const std::vector<std::string>& include_paths = {},
        const std::vector<std::string>& compile_options = {}) const;
    
    // Utility functions
    static std::vector<char> read_file(const std::string& file_path);
    static std::string compile_cuda_to_ptx(
        const std::string& cuda_source,
        const std::vector<std::string>& include_paths = {},
        const std::vector<std::string>& compile_options = {});
    
    // Validate PTX code
    static bool validate_ptx(const std::string& ptx_code);
    
private:
    std::vector<char> read_ptx_file(const std::string& file_path) const;
};

} // namespace ebpf_gpu 