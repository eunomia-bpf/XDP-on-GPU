#include <catch2/catch_test_macros.hpp>
#include "gpu_device_manager.hpp"
#include "error_handling.hpp"

using namespace ebpf_gpu;

TEST_CASE("GpuDeviceManager - Basic Functionality", "[device_manager]") {
    SECTION("Device count") {
        GpuDeviceManager manager;
        int device_count = manager.get_device_count();
        REQUIRE(device_count >= 0);
        
        if (device_count > 0) {
            INFO("Found " << device_count << " CUDA devices");
        } else {
            WARN("No CUDA devices found - some tests will be skipped");
        }
    }
    
    SECTION("Device information") {
        GpuDeviceManager manager;
        int device_count = manager.get_device_count();
        
        if (device_count > 0) {
            auto devices = manager.get_all_devices();
            REQUIRE(devices.size() == static_cast<size_t>(device_count));
            
            for (const auto& device : devices) {
                REQUIRE(device.device_id >= 0);
                REQUIRE(!device.name.empty());
                REQUIRE(device.total_memory > 0);
                REQUIRE(device.compute_capability_major >= 3);
                REQUIRE(device.compute_capability_minor >= 0);
                REQUIRE(device.multiprocessor_count > 0);
                REQUIRE(device.max_threads_per_block > 0);
                
                INFO("Device " << device.device_id << ": " << device.name);
                INFO("  Compute capability: " << device.compute_capability_major 
                     << "." << device.compute_capability_minor);
                INFO("  Total memory: " << device.total_memory / (1024*1024) << " MB");
                INFO("  Free memory: " << device.free_memory / (1024*1024) << " MB");
                INFO("  Multiprocessors: " << device.multiprocessor_count);
            }
        }
    }
}

TEST_CASE("GpuDeviceManager - Device Selection", "[device_manager]") {
    GpuDeviceManager manager;
    int device_count = manager.get_device_count();
    
    if (device_count == 0) {
        SKIP("No CUDA devices available");
    }
    
    SECTION("Best device selection") {
        int best_device = manager.select_best_device();
        REQUIRE(best_device >= 0);
        REQUIRE(best_device < device_count);
        
        auto device_info = manager.get_device_info(best_device);
        INFO("Selected best device: " << device_info.name);
    }
    
    SECTION("Device suitability") {
        for (int i = 0; i < device_count; ++i) {
            bool suitable = manager.is_device_suitable(i);
            auto device_info = manager.get_device_info(i);
            
            INFO("Device " << i << " (" << device_info.name << ") suitable: " << suitable);
            
            // Check with memory requirement
            size_t min_memory = 100 * 1024 * 1024; // 100MB
            bool suitable_with_memory = manager.is_device_suitable(i, min_memory);
            
            if (device_info.free_memory >= min_memory) {
                REQUIRE(suitable_with_memory);
            }
        }
    }
}

TEST_CASE("GpuDeviceManager - Capabilities", "[device_manager]") {
    GpuDeviceManager manager;
    int device_count = manager.get_device_count();
    
    if (device_count == 0) {
        SKIP("No CUDA devices available");
    }
    
    SECTION("Unified memory support") {
        for (int i = 0; i < device_count; ++i) {
            bool supports_unified = manager.supports_unified_memory(i);
            auto device_info = manager.get_device_info(i);
            
            REQUIRE(supports_unified == device_info.unified_addressing);
            INFO("Device " << i << " unified memory: " << supports_unified);
        }
    }
    
    SECTION("Compute capability check") {
        for (int i = 0; i < device_count; ++i) {
            auto device_info = manager.get_device_info(i);
            
            // Check if device supports at least compute capability 3.5
            bool supports_35 = manager.supports_compute_capability(i, 3, 5);
            bool expected_35 = (device_info.compute_capability_major > 3) ||
                              (device_info.compute_capability_major == 3 && 
                               device_info.compute_capability_minor >= 5);
            
            REQUIRE(supports_35 == expected_35);
            
            // Check current capability
            bool supports_current = manager.supports_compute_capability(
                i, device_info.compute_capability_major, device_info.compute_capability_minor);
            REQUIRE(supports_current);
        }
    }
}

TEST_CASE("GpuDeviceManager - Memory Information", "[device_manager]") {
    GpuDeviceManager manager;
    int device_count = manager.get_device_count();
    
    if (device_count == 0) {
        SKIP("No CUDA devices available");
    }
    
    SECTION("Memory queries") {
        for (int i = 0; i < device_count; ++i) {
            size_t total_memory = manager.get_total_memory(i);
            size_t available_memory = manager.get_available_memory(i);
            
            REQUIRE(total_memory > 0);
            REQUIRE(available_memory <= total_memory);
            
            INFO("Device " << i << " memory: " << available_memory / (1024*1024) 
                 << " MB available / " << total_memory / (1024*1024) << " MB total");
        }
    }
}

TEST_CASE("GpuDeviceManager - Error Handling", "[device_manager]") {
    GpuDeviceManager manager;
    
    SECTION("Invalid device ID") {
        REQUIRE_THROWS_AS(manager.get_device_info(-1), std::out_of_range);
        REQUIRE_THROWS_AS(manager.get_device_info(1000), std::out_of_range);
        
        REQUIRE_FALSE(manager.is_device_suitable(-1));
        REQUIRE_FALSE(manager.is_device_suitable(1000));
        
        REQUIRE_FALSE(manager.supports_unified_memory(-1));
        REQUIRE_FALSE(manager.supports_compute_capability(-1, 3, 5));
        
        REQUIRE(manager.get_available_memory(-1) == 0);
        REQUIRE(manager.get_total_memory(-1) == 0);
    }
} 