#pragma once

#include <string>
#include <vector>
#include <memory>

namespace ebpf_gpu {

// Forward declaration
class CudaModule;

/**
 * Utility class for loading and managing GPU kernels
 */
class KernelLoader {
public:
    KernelLoader();
    ~KernelLoader();
    
    // Non-copyable
    KernelLoader(const KernelLoader&) = delete;
    KernelLoader& operator=(const KernelLoader&) = delete;
    
    // Movable
    KernelLoader(KernelLoader&&) = default;
    KernelLoader& operator=(KernelLoader&&) = default;
    
    /**
     * Load kernel from PTX code string
     * @param ptx PTX code string
     * @throws std::runtime_error if loading fails
     */
    void load_from_ptx(const std::string& ptx);
    
    /**
     * Load kernel from file (.ptx, .cl, etc.)
     * @param path File path
     * @throws std::runtime_error if loading fails
     */
    void load_from_file(const std::string& path);
    
    /**
     * Get a kernel function by name
     * @param name Kernel function name
     * @return Pointer to kernel function (must be cast to appropriate type)
     * @throws std::runtime_error if kernel not found
     */
    void* get_kernel(const std::string& name) const;
    
    /**
     * Static helper to validate PTX code
     * @param ptx PTX code string
     * @return true if valid, false otherwise
     */
    static bool validate_ptx(const std::string& ptx);
    
    /**
     * Static helper to read a file into a vector of chars
     * @param path File path
     * @return Vector containing file contents
     * @throws std::runtime_error if file cannot be read
     */
    static std::vector<char> read_file(const std::string& path);
    
private:
    std::unique_ptr<CudaModule> module_;
};

} // namespace ebpf_gpu 