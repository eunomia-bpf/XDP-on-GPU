#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include "gpu_device_manager.hpp"
#include "kernel_loader.hpp"
#include <vector>
#include <chrono>

using namespace ebpf_gpu;

TEST_CASE("Performance - GPU Device Detection", "[performance]") {
    BENCHMARK("Device manager initialization") {
        GpuDeviceManager manager;
        return manager.get_device_count();
    };
    
    BENCHMARK("Device info query") {
        GpuDeviceManager manager;
        if (manager.get_device_count() > 0) {
            return manager.get_device_info(0);
        }
        return GpuDeviceInfo{};
    };
}

TEST_CASE("Performance - Kernel Loading", "[performance]") {
    // Initialize GPU device manager first to set up CUDA context
    GpuDeviceManager device_manager;
    if (device_manager.get_device_count() == 0) {
        SKIP("No CUDA devices available for performance testing");
    }
    
    KernelLoader loader;
    
    // Create PTX with auto-detected architecture
    std::string simple_ptx = 
        ".version 7.0\n"
        ".target sm_" + std::to_string(CUDA_ARCH_SM) + "\n"
        ".address_size 64\n"
        "\n"
        ".visible .entry simple_kernel(\n"
        "    .param .u64 simple_kernel_param_0\n"
        ")\n"
        "{\n"
        "    ret;\n"
        "}\n";
    
    BENCHMARK("PTX validation") {
        return KernelLoader::validate_ptx(simple_ptx);
    };
    
    BENCHMARK("PTX loading") {
        return loader.load_from_ptx(simple_ptx);
    };
}

TEST_CASE("Performance - Memory Operations", "[performance]") {
    SECTION("Vector operations") {
        BENCHMARK("Vector creation (1K elements)") {
            std::vector<int> vec(1000);
            return vec.size();
        };
        
        BENCHMARK("Vector creation (1M elements)") {
            std::vector<int> vec(1000000);
            return vec.size();
        };
    }
} 