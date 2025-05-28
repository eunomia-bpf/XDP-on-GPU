#include "error_handling.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

namespace ebpf_gpu {

CudaDriverException::CudaDriverException(CUresult error, const std::string& operation)
    : CudaException([&]() {
        const char* error_name = nullptr;
        const char* error_string = nullptr;
        cuGetErrorName(error, &error_name);
        cuGetErrorString(error, &error_string);
        return operation + " failed: " + (error_name ? error_name : "unknown") + 
               " (" + (error_string ? error_string : "no description") + ")";
    }())
    , error_code_(error) {}

// CudaContext implementation
CudaContext::CudaContext(int device_id) : context_(nullptr), owns_context_(true) {
    check_cuda_driver(cuInit(0), "cuInit");
    
    CUdevice device;
    check_cuda_driver(cuDeviceGet(&device, device_id), "cuDeviceGet");
    check_cuda_driver(cuCtxCreate(&context_, 0, device), "cuCtxCreate");
}

CudaContext::~CudaContext() {
    if (owns_context_ && context_) {
        cuCtxDestroy(context_);
    }
}

CudaContext::CudaContext(CudaContext&& other) noexcept 
    : context_(other.context_), owns_context_(other.owns_context_) {
    other.context_ = nullptr;
    other.owns_context_ = false;
}

CudaContext& CudaContext::operator=(CudaContext&& other) noexcept {
    if (this != &other) {
        if (owns_context_ && context_) {
            cuCtxDestroy(context_);
        }
        context_ = other.context_;
        owns_context_ = other.owns_context_;
        other.context_ = nullptr;
        other.owns_context_ = false;
    }
    return *this;
}

void CudaContext::set_current() const {
    check_cuda_driver(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
}

// DeviceMemory implementation
DeviceMemory::DeviceMemory(size_t size) : ptr_(nullptr), size_(size) {
    check_cuda_runtime(cudaMalloc(&ptr_, size), "cudaMalloc");
}

DeviceMemory::~DeviceMemory() {
    if (ptr_) {
        cudaFree(ptr_);
    }
}

DeviceMemory::DeviceMemory(DeviceMemory&& other) noexcept 
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

DeviceMemory& DeviceMemory::operator=(DeviceMemory&& other) noexcept {
    if (this != &other) {
        if (ptr_) {
            cudaFree(ptr_);
        }
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void DeviceMemory::copy_from_host(const void* host_ptr, size_t size) {
    if (size > size_) {
        throw std::invalid_argument("Copy size exceeds device memory size");
    }
    check_cuda_runtime(cudaMemcpy(ptr_, host_ptr, size, cudaMemcpyHostToDevice), 
                      "cudaMemcpy (host to device)");
}

void DeviceMemory::copy_to_host(void* host_ptr, size_t size) const {
    if (size > size_) {
        throw std::invalid_argument("Copy size exceeds device memory size");
    }
    check_cuda_runtime(cudaMemcpy(host_ptr, ptr_, size, cudaMemcpyDeviceToHost), 
                      "cudaMemcpy (device to host)");
}

// Utility functions
void check_cuda_runtime(cudaError_t result, const std::string& operation) {
    if (result != cudaSuccess) {
        throw CudaRuntimeException(result, operation);
    }
}

void check_cuda_driver(CUresult result, const std::string& operation) {
    if (result != CUDA_SUCCESS) {
        throw CudaDriverException(result, operation);
    }
}

} // namespace ebpf_gpu 