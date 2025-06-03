#include "../include/gpu_backend.hpp"
#include "../include/kernel_loader.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <memory>

namespace ebpf_gpu {

class CudaBackend : public GpuBackend {
public:
    CudaBackend();
    ~CudaBackend() override;

    // Backend type
    BackendType get_type() const override { return BackendType::CUDA; }
    
    // Device management
    void initialize_device(int device_id) override;
    void set_device(int device_id) override;
    
    // Memory management
    void* allocate_device_memory(size_t size) override;
    void free_device_memory(void* ptr) override;
    void copy_host_to_device(void* dst, const void* src, size_t size) override;
    void copy_device_to_host(void* dst, const void* src, size_t size) override;
    
    // Kernel management
    bool load_kernel_from_source(const std::string& source_code, 
                                const std::string& function_name,
                                const std::vector<std::string>& include_paths = {},
                                const std::vector<std::string>& compile_options = {}) override;
    bool load_kernel_from_binary(const std::string& binary_path, 
                                const std::string& function_name) override;
    
    // Kernel execution
    bool launch_kernel(void* data, size_t event_count, 
                      int block_size, size_t shared_memory_size, 
                      int max_grid_size) override;
    
    // Async execution support
    void* create_stream() override;
    void destroy_stream(void* stream) override;
    bool launch_kernel_async(void* data, size_t event_count, 
                            int block_size, size_t shared_memory_size, 
                            int max_grid_size, void* stream) override;
    bool synchronize_stream(void* stream) override;
    bool synchronize_device() override;
    
    // Memory management with streams
    void copy_host_to_device_async(void* dst, const void* src, size_t size, void* stream) override;
    void copy_device_to_host_async(void* dst, const void* src, size_t size, void* stream) override;
    
    // Pinned memory support
    void* allocate_pinned_host_memory(size_t size) override;
    void free_pinned_host_memory(void* ptr) override;
    bool register_host_memory(void* ptr, size_t size, unsigned int flags) override;
    bool unregister_host_memory(void* ptr) override;
    
    // Device information
    GpuDeviceInfo get_device_info(int device_id) const override;
    size_t get_available_memory(int device_id) const override;

private:
    CUcontext context_;
    std::unique_ptr<CudaModule> module_;
    CUfunction kernel_function_;
    int current_device_id_;
    GpuDeviceManager device_manager_;
    
    // Ensure context is current
    bool ensure_context_current();
};

CudaBackend::CudaBackend() : context_(nullptr), kernel_function_(nullptr), current_device_id_(-1) {
    // Initialize CUDA driver if not already done
    CUresult cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to initialize CUDA driver");
    }
}

CudaBackend::~CudaBackend() {
    if (context_) {
        cuCtxDestroy(context_);
    }
}

void CudaBackend::initialize_device(int device_id) {
    if (context_) {
        cuCtxDestroy(context_);
        context_ = nullptr;
    }
    
    current_device_id_ = device_id;
    
    // Get device and create context
    CUdevice device;
    CUresult result = cuDeviceGet(&device, device_id);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to get CUDA device");
    }
    
    result = cuCtxCreate(&context_, 0, device);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to create CUDA context");
    }
    
    // Set this context as current immediately after creation
    result = cuCtxSetCurrent(context_);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to set CUDA context as current");
    }
}

void CudaBackend::set_device(int device_id) {
    current_device_id_ = device_id;
    cudaSetDevice(device_id);
}

void* CudaBackend::allocate_device_memory(size_t size) {
    void* ptr = nullptr;
    cudaError_t result = cudaMalloc(&ptr, size);
    if (result != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

void CudaBackend::free_device_memory(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void CudaBackend::copy_host_to_device(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void CudaBackend::copy_device_to_host(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

bool CudaBackend::load_kernel_from_source(const std::string& source_code,
                                         const std::string& function_name,
                                         const std::vector<std::string>& include_paths,
                                         const std::vector<std::string>& compile_options) {
    try {
        if (source_code.empty() || function_name.empty()) {
            return false;
        }
        
        KernelLoader loader;
        module_ = loader.load_from_cuda_source(source_code, include_paths, compile_options);
        if (!module_ || !module_->is_valid()) {
            return false;
        }
        
        kernel_function_ = module_->get_function(function_name);
        return kernel_function_ != nullptr;
    } catch (...) {
        return false;
    }
}

bool CudaBackend::load_kernel_from_binary(const std::string& binary_path,
                                         const std::string& function_name) {
    try {
        if (binary_path.empty() || function_name.empty()) {
            return false;
        }
        
        KernelLoader loader;
        module_ = loader.load_from_file(binary_path);
        if (!module_ || !module_->is_valid()) {
            return false;
        }
        
        kernel_function_ = module_->get_function(function_name);
        return kernel_function_ != nullptr;
    } catch (...) {
        return false;
    }
}

bool CudaBackend::launch_kernel(void* data, size_t event_count,
                               int block_size, size_t shared_memory_size,
                               int max_grid_size) {
    if (!kernel_function_ || !ensure_context_current()) {
        return false;
    }
    
    try {
        // Calculate grid size based on event count and block size
        int grid_size = static_cast<int>((event_count + block_size - 1) / block_size);
        if (max_grid_size > 0 && grid_size > max_grid_size) {
            grid_size = max_grid_size;
        }
        
        // Set up launch parameters
        void* args[] = { &data, &event_count };
        
        // Launch the kernel
        CUresult result = cuLaunchKernel(
            kernel_function_,
            grid_size, 1, 1,          // Grid dimensions
            block_size, 1, 1,         // Block dimensions
            shared_memory_size,       // Shared memory size
            nullptr,                  // Stream (using default stream)
            args,                     // Kernel arguments
            nullptr                   // Extra (unused)
        );
        
        if (result != CUDA_SUCCESS) {
            return false;
        }
        
        // Synchronize to wait for kernel completion
        return synchronize_device();
    } catch (...) {
        return false;
    }
}

void* CudaBackend::create_stream() {
    cudaStream_t stream;
    cudaError_t result = cudaStreamCreate(&stream);
    if (result != cudaSuccess) {
        return nullptr;
    }
    return reinterpret_cast<void*>(stream);
}

void CudaBackend::destroy_stream(void* stream) {
    if (stream) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    }
}

bool CudaBackend::launch_kernel_async(void* data, size_t event_count,
                                     int block_size, size_t shared_memory_size,
                                     int max_grid_size, void* stream) {
    if (!kernel_function_ || !ensure_context_current()) {
        return false;
    }
    
    try {
        // Calculate grid size based on event count and block size
        int grid_size = static_cast<int>((event_count + block_size - 1) / block_size);
        if (max_grid_size > 0 && grid_size > max_grid_size) {
            grid_size = max_grid_size;
        }
        
        // Set up launch parameters
        void* args[] = { &data, &event_count };
        
        // Launch the kernel
        CUresult result = cuLaunchKernel(
            kernel_function_,
            grid_size, 1, 1,          // Grid dimensions
            block_size, 1, 1,         // Block dimensions
            shared_memory_size,       // Shared memory size
            static_cast<cudaStream_t>(stream), // Stream
            args,                     // Kernel arguments
            nullptr                   // Extra (unused)
        );
        
        return result == CUDA_SUCCESS;
    } catch (...) {
        return false;
    }
}

bool CudaBackend::synchronize_stream(void* stream) {
    if (!stream) {
        return false;
    }
    
    cudaError_t result = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    return result == cudaSuccess;
}

bool CudaBackend::synchronize_device() {
    cudaError_t result = cudaDeviceSynchronize();
    return result == cudaSuccess;
}

void CudaBackend::copy_host_to_device_async(void* dst, const void* src, size_t size, void* stream) {
    cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
}

void CudaBackend::copy_device_to_host_async(void* dst, const void* src, size_t size, void* stream) {
    cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
}

void* CudaBackend::allocate_pinned_host_memory(size_t size) {
    void* ptr = nullptr;
    cudaError_t result = cudaMallocHost(&ptr, size);
    if (result != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

void CudaBackend::free_pinned_host_memory(void* ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

bool CudaBackend::register_host_memory(void* ptr, size_t size, unsigned int flags) {
    if (!ptr || size == 0) {
        return false;
    }
    
    cudaError_t result = cudaHostRegister(ptr, size, flags);
    return result == cudaSuccess;
}

bool CudaBackend::unregister_host_memory(void* ptr) {
    if (!ptr) {
        return false;
    }
    
    cudaError_t result = cudaHostUnregister(ptr);
    return result == cudaSuccess;
}

GpuDeviceInfo CudaBackend::get_device_info(int device_id) const {
    return device_manager_.get_device_info(device_id);
}

size_t CudaBackend::get_available_memory(int device_id) const {
    return device_manager_.get_available_memory(device_id);
}

bool CudaBackend::ensure_context_current() {
    if (!context_) {
        return false;
    }
    
    CUcontext current;
    CUresult result = cuCtxGetCurrent(&current);
    if (result != CUDA_SUCCESS) {
        return false;
    }
    
    if (current != context_) {
        result = cuCtxSetCurrent(context_);
        if (result != CUDA_SUCCESS) {
            return false;
        }
    }
    
    return true;
}

#ifdef USE_CUDA_BACKEND
// Register factory method
std::unique_ptr<GpuBackend> create_cuda_backend() {
    return std::make_unique<CudaBackend>();
}
#endif

} // namespace ebpf_gpu 