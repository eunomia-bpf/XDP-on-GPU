#include <catch2/catch_test_macros.hpp>
#include "gpu_device_manager.hpp"
#include "kernel_loader.hpp"
#include "error_handling.hpp"
#include <vector>

using namespace ebpf_gpu;

TEST_CASE("Integration - Complete Workflow", "[integration]") {
    SECTION("Device detection and kernel loading") {
        // Test the complete workflow from device detection to kernel loading
        GpuDeviceManager device_manager;
        
        if (device_manager.get_device_count() == 0) {
            SKIP("No CUDA devices available for integration testing");
        }
        
        // Select best device
        int device_id = device_manager.select_best_device();
        REQUIRE(device_id >= 0);
        
        auto device_info = device_manager.get_device_info(device_id);
        REQUIRE(!device_info.name.empty());
        
        // Load a simple kernel
        KernelLoader loader;
        // Create PTX with auto-detected architecture
        std::string simple_ptx = 
            ".version 7.0\n"
            ".target sm_" + std::to_string(CUDA_ARCH_SM) + "\n"
            ".address_size 64\n"
            "\n"
            ".visible .entry integration_test_kernel(\n"
            "    .param .u64 integration_test_kernel_param_0\n"
            ")\n"
            "{\n"
            "    ret;\n"
            "}\n";
        
        REQUIRE(KernelLoader::validate_ptx(simple_ptx));
        
        auto module = loader.load_from_ptx(simple_ptx);
        REQUIRE(module != nullptr);
        REQUIRE(module->is_valid());
    }
}

TEST_CASE("Integration - Error Handling", "[integration]") {
    SECTION("Exception propagation") {
        // Initialize GPU device manager first
        GpuDeviceManager device_manager;
        if (device_manager.get_device_count() == 0) {
            SKIP("No CUDA devices available for error handling testing");
        }
        
        // Test that exceptions are properly propagated through the system
        KernelLoader loader;
        
        REQUIRE_THROWS_AS(loader.load_from_ptx(""), std::invalid_argument);
        REQUIRE_THROWS_AS(loader.load_from_file("/non/existent/file.ptx"), std::runtime_error);
    }
}

TEST_CASE("Integration - Resource Management", "[integration]") {
    SECTION("RAII behavior") {
        // Test that resources are properly managed
        {
            GpuDeviceManager manager;
            KernelLoader loader;
            
            // Objects should be properly destroyed when going out of scope
            REQUIRE(manager.get_device_count() >= 0);
        }
        
        // Test multiple instances
        for (int i = 0; i < 5; ++i) {
            GpuDeviceManager manager;
            REQUIRE(manager.get_device_count() >= 0);
        }
    }
} 