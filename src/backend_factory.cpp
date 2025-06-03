#include "../include/gpu_backend.hpp"
#include <stdexcept>
#include <memory>
#include <iostream>

namespace ebpf_gpu {

// Forward declarations for backend creation functions
#ifdef USE_CUDA_BACKEND
std::unique_ptr<GpuBackend> create_cuda_backend();
#endif

#ifdef USE_OPENCL_BACKEND
std::unique_ptr<GpuBackend> create_opencl_backend();
#endif

// Fallback/stub backend when no GPU backends are available
class StubBackend : public GpuBackend {
public:
    BackendType get_type() const override { return BackendType::CUDA; } // Default to CUDA type
    
    void initialize_device(int) override {}
    void set_device(int) override {}
    
    void* allocate_device_memory(size_t) override { return nullptr; }
    void free_device_memory(void*) override {}
    void copy_host_to_device(void*, const void*, size_t) override {}
    void copy_device_to_host(void*, const void*, size_t) override {}
    
    bool load_kernel_from_source(const std::string&, const std::string&, 
                                const std::vector<std::string>&, 
                                const std::vector<std::string>&) override { return false; }
    bool load_kernel_from_binary(const std::string&, const std::string&) override { return false; }
    
    bool launch_kernel(void*, size_t, int, size_t, int) override { return false; }
    
    void* create_stream() override { return nullptr; }
    void destroy_stream(void*) override {}
    bool launch_kernel_async(void*, size_t, int, size_t, int, void*) override { return false; }
    bool synchronize_stream(void*) override { return false; }
    bool synchronize_device() override { return false; }
    
    void copy_host_to_device_async(void*, const void*, size_t, void*) override {}
    void copy_device_to_host_async(void*, const void*, size_t, void*) override {}
    
    void* allocate_pinned_host_memory(size_t) override { return nullptr; }
    void free_pinned_host_memory(void*) override {}
    bool register_host_memory(void*, size_t, unsigned int) override { return false; }
    bool unregister_host_memory(void*) override { return false; }
    
    GpuDeviceInfo get_device_info(int) const override { return GpuDeviceInfo{}; }
    size_t get_available_memory(int) const override { return 0; }
};

std::unique_ptr<GpuBackend> create_stub_backend() {
    return std::make_unique<StubBackend>();
}

std::unique_ptr<GpuBackend> create_backend(BackendType type) {
    try {
        switch (type) {
            case BackendType::CUDA:
            #ifdef USE_CUDA_BACKEND
                return create_cuda_backend();
            #elif defined(USE_OPENCL_BACKEND)
                // Fallback to OpenCL if CUDA not available but OpenCL is
                std::cout << "CUDA backend requested but not available, falling back to OpenCL" << std::endl;
                return create_opencl_backend();
            #else
                std::cout << "No GPU backends available, using stub backend" << std::endl;
                return create_stub_backend();
            #endif
            
            case BackendType::OpenCL:
            #ifdef USE_OPENCL_BACKEND
                return create_opencl_backend();
            #elif defined(USE_CUDA_BACKEND)
                // Fallback to CUDA if OpenCL not available but CUDA is
                std::cout << "OpenCL backend requested but not available, falling back to CUDA" << std::endl;
                return create_cuda_backend();
            #else
                std::cout << "No GPU backends available, using stub backend" << std::endl;
                return create_stub_backend();
            #endif
            
            default:
                throw std::invalid_argument("Unknown backend type");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error creating backend: " << e.what() << std::endl;
        std::cout << "Falling back to stub backend" << std::endl;
        return create_stub_backend();
    }
}

} // namespace ebpf_gpu 