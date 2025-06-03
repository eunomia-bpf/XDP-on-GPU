#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <memory>  // For std::unique_ptr

namespace ebpf_gpu {

/**
 * @brief Type of GPU backend
 */
enum class BackendType {
    CUDA,
    OpenCL
};

/**
 * @brief Information about a GPU device
 */
struct GpuDeviceInfo {
    int device_id = -1;
    BackendType backend_type = BackendType::CUDA;
    std::string name;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int clock_rate = 0;          // Clock rate in kHz
    int num_cores = 0;           // Number of cores/multiprocessors
    size_t total_memory = 0;     // Total global memory in bytes
    size_t available_memory = 0; // Available memory in bytes
};

/**
 * @brief Abstract interface for GPU backends (CUDA, OpenCL, etc.)
 */
class GpuBackend {
public:
    virtual ~GpuBackend() = default;
    
    // Device management
    virtual void initialize_device(int device_id) = 0;
    virtual void set_device(int device_id) = 0;
    
    // Memory management
    virtual void* allocate_device_memory(size_t size) = 0;
    virtual void free_device_memory(void* ptr) = 0;
    virtual void copy_host_to_device(void* dst, const void* src, size_t size) = 0;
    virtual void copy_device_to_host(void* dst, const void* src, size_t size) = 0;
    
    // Kernel management
    virtual bool load_kernel_from_source(const std::string& source, const std::string& kernel_name,
                                       const std::vector<std::string>& compile_options = {},
                                       const std::vector<std::string>& include_paths = {}) = 0;
    virtual bool load_kernel_from_binary(const std::string& binary_path, const std::string& kernel_name) = 0;
    
    // Kernel execution
    virtual bool launch_kernel(void* kernel, size_t num_elements, int device_id, size_t block_size, int shared_memory_size = 0) = 0;
    
    // Asynchronous operations
    virtual void* create_stream() = 0;
    virtual void destroy_stream(void* stream) = 0;
    virtual bool launch_kernel_async(void* kernel, size_t num_elements, int device_id, size_t block_size, 
                                   int shared_memory_size, void* stream) = 0;
    virtual bool synchronize_stream(void* stream) = 0;
    virtual bool synchronize_device() = 0;
    
    // Asynchronous memory operations
    virtual void copy_host_to_device_async(void* dst, const void* src, size_t size, void* stream) = 0;
    virtual void copy_device_to_host_async(void* dst, const void* src, size_t size, void* stream) = 0;
    
    // Pinned memory operations
    virtual void* allocate_pinned_host_memory(size_t size) = 0;
    virtual void free_pinned_host_memory(void* ptr) = 0;
    
    // Page-locked memory operations
    virtual bool register_host_memory(void* ptr, size_t size, unsigned int flags) = 0;
    virtual bool unregister_host_memory(void* ptr) = 0;
    
    // Device information
    virtual GpuDeviceInfo get_device_info(int device_id) const = 0;
    virtual size_t get_available_memory(int device_id) const = 0;
    
    // Backend type
    virtual BackendType get_type() const = 0;
};

// Factory function to create a specific backend
std::unique_ptr<GpuBackend> create_backend(BackendType type);

} // namespace ebpf_gpu 