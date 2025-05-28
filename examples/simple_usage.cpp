#include "gpu_device_manager.hpp"
#include "kernel_loader.hpp"
#include "error_handling.hpp"
#include <iostream>
#include <vector>

using namespace ebpf_gpu;

int main() {
    try {
        std::cout << "eBPF GPU Processor - Simple Usage Example\n";
        std::cout << "==========================================\n\n";
        
        // 1. Initialize GPU device manager
        std::cout << "1. Detecting GPU devices...\n";
        GpuDeviceManager device_manager;
        
        int device_count = device_manager.get_device_count();
        std::cout << "   Found " << device_count << " CUDA device(s)\n";
        
        if (device_count == 0) {
            std::cout << "   No CUDA devices available. Exiting.\n";
            return 1;
        }
        
        // 2. Select best device
        std::cout << "\n2. Selecting best device...\n";
        int best_device = device_manager.select_best_device();
        auto device_info = device_manager.get_device_info(best_device);
        
        std::cout << "   Selected device " << best_device << ": " << device_info.name << "\n";
        std::cout << "   Compute capability: " << device_info.compute_capability_major 
                  << "." << device_info.compute_capability_minor << "\n";
        std::cout << "   Total memory: " << device_info.total_memory / (1024*1024) << " MB\n";
        std::cout << "   Free memory: " << device_info.free_memory / (1024*1024) << " MB\n";
        
        // 3. Load a simple kernel
        std::cout << "\n3. Loading kernel...\n";
        KernelLoader loader;
        
        // Simple PTX kernel that just returns
        std::string simple_ptx = R"(
.version 7.0
.target sm_61
.address_size 64

.visible .entry simple_example_kernel(
    .param .u64 simple_example_kernel_param_0,
    .param .u64 simple_example_kernel_param_1
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<3>;
    
    ld.param.u64 %rd1, [simple_example_kernel_param_0];
    ld.param.u64 %rd2, [simple_example_kernel_param_1];
    
    ret;
}
)";
        
        // Validate PTX
        if (!KernelLoader::validate_ptx(simple_ptx)) {
            std::cout << "   PTX validation failed!\n";
            return 1;
        }
        std::cout << "   PTX validation passed\n";
        
        // Load the kernel
        auto module = loader.load_from_ptx(simple_ptx);
        std::cout << "   Kernel loaded successfully\n";
        std::cout << "   Module valid: " << (module->is_valid() ? "Yes" : "No") << "\n";
        
        // 4. Get kernel function
        std::cout << "\n4. Getting kernel function...\n";
        auto function = module->get_function("simple_example_kernel");
        std::cout << "   Function retrieved successfully\n";
        
        std::cout << "\n✓ Example completed successfully!\n";
        std::cout << "\nThis demonstrates:\n";
        std::cout << "  - GPU device detection and selection\n";
        std::cout << "  - PTX kernel validation and loading\n";
        std::cout << "  - Kernel function retrieval\n";
        std::cout << "  - Proper error handling with exceptions\n";
        
        return 0;
        
    } catch (const CudaException& e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 