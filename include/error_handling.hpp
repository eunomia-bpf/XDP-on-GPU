#pragma once

#include <stdexcept>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

namespace ebpf_gpu {

class CudaException : public std::runtime_error {
public:
    explicit CudaException(const std::string& message) 
        : std::runtime_error(message) {}
};

class CudaRuntimeException : public CudaException {
public:
    CudaRuntimeException(cudaError_t error, const std::string& operation)
        : CudaException(operation + " failed: " + cudaGetErrorString(error))
        , error_code_(error) {}
    
    cudaError_t error_code() const noexcept { return error_code_; }

private:
    cudaError_t error_code_;
};

class CudaDriverException : public CudaException {
public:
    CudaDriverException(CUresult error, const std::string& operation);
    
    CUresult error_code() const noexcept { return error_code_; }

private:
    CUresult error_code_;
};

// RAII wrapper for CUDA context
class CudaContext {
public:
    explicit CudaContext(int device_id);
    ~CudaContext();
    
    // Non-copyable, movable
    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;
    CudaContext(CudaContext&& other) noexcept;
    CudaContext& operator=(CudaContext&& other) noexcept;
    
    CUcontext get() const noexcept { return context_; }
    void set_current() const;

private:
    CUcontext context_;
    bool owns_context_;
};

// RAII wrapper for device memory
class DeviceMemory {
public:
    explicit DeviceMemory(size_t size);
    ~DeviceMemory();
    
    // Non-copyable, movable
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    DeviceMemory(DeviceMemory&& other) noexcept;
    DeviceMemory& operator=(DeviceMemory&& other) noexcept;
    
    void* get() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
    
    void copy_from_host(const void* host_ptr, size_t size);
    void copy_to_host(void* host_ptr, size_t size) const;

private:
    void* ptr_;
    size_t size_;
};

// Utility functions
void check_cuda_runtime(cudaError_t result, const std::string& operation);
void check_cuda_driver(CUresult result, const std::string& operation);

} // namespace ebpf_gpu 