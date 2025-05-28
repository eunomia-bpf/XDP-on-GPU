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
    
    std::string simple_ptx = R"(
.version 7.0
.target sm_61
.address_size 64

.visible .entry simple_kernel(
    .param .u64 simple_kernel_param_0
)
{
    ret;
}
)";
    
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