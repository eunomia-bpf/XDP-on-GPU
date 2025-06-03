#include <catch2/catch_test_macros.hpp>
#include "../include/gpu_device_manager.hpp"
#include <iostream>

TEST_CASE("Device Manager Initialization", "[device][manager]") {
    SECTION("Create Device Manager") {
        ebpf_gpu::GpuDeviceManager manager;
        
        // Test with valid backend type
        int cuda_count = manager.get_device_count(ebpf_gpu::BackendType::CUDA);
        int opencl_count = manager.get_device_count(ebpf_gpu::BackendType::OpenCL);
        
        // Just verify that these don't crash
        SUCCEED("Got device counts: CUDA=" + std::to_string(cuda_count) 
               + ", OpenCL=" + std::to_string(opencl_count));
        
        // If we have devices, try to get info about them
        if (cuda_count > 0 || opencl_count > 0) {
            ebpf_gpu::BackendType backend = (cuda_count > 0) ? 
                ebpf_gpu::BackendType::CUDA : ebpf_gpu::BackendType::OpenCL;
                
            int best_device = manager.get_best_device(backend);
            REQUIRE(best_device >= 0);
            
            auto device_info = manager.query_device_info(best_device);
            REQUIRE(!device_info.name.empty());
        }
    }
    
    SECTION("Multiple Device Manager Instances") {
        ebpf_gpu::GpuDeviceManager manager1;
        ebpf_gpu::GpuDeviceManager manager2;
        
        // Test with valid backend type
        int cuda_count1 = manager1.get_device_count(ebpf_gpu::BackendType::CUDA);
        int cuda_count2 = manager2.get_device_count(ebpf_gpu::BackendType::CUDA);
        
        REQUIRE(cuda_count1 == cuda_count2);
        
        // Not checking device features since they may not be available on test systems
    }
}

TEST_CASE("Device Selection", "[device][selection]") {
    ebpf_gpu::GpuDeviceManager manager;
    
    // Test with valid backend type
    int cuda_count = manager.get_device_count(ebpf_gpu::BackendType::CUDA);
    int opencl_count = manager.get_device_count(ebpf_gpu::BackendType::OpenCL);
    
    if (cuda_count > 0 || opencl_count > 0) {
        ebpf_gpu::BackendType backend = (cuda_count > 0) ? 
            ebpf_gpu::BackendType::CUDA : ebpf_gpu::BackendType::OpenCL;
            
        int best_device = manager.get_best_device(backend);
        REQUIRE(best_device >= 0);
        
        // Get device info
        auto device_info = manager.query_device_info(best_device);
        REQUIRE(!device_info.name.empty());
    } else {
        SUCCEED("No devices available to test selection");
    }
}

TEST_CASE("Device Memory", "[device][memory]") {
    ebpf_gpu::GpuDeviceManager manager;
    
    // Test with valid backend type
    int cuda_count = manager.get_device_count(ebpf_gpu::BackendType::CUDA);
    int opencl_count = manager.get_device_count(ebpf_gpu::BackendType::OpenCL);
    
    if (cuda_count > 0 || opencl_count > 0) {
        ebpf_gpu::BackendType backend = (cuda_count > 0) ? 
            ebpf_gpu::BackendType::CUDA : ebpf_gpu::BackendType::OpenCL;
            
        int device_id = manager.get_best_device(backend);
        
        // Get device memory
        size_t mem = manager.get_available_memory(device_id);
        
        REQUIRE(mem > 0);
        
        // Get device info to verify name 
        auto device_info = manager.query_device_info(device_id);
        REQUIRE(!device_info.name.empty());
    } else {
        SUCCEED("No devices available to test memory");
    }
}

TEST_CASE("Error Handling", "[device][error]") {
    ebpf_gpu::GpuDeviceManager manager;
    
    // Test with invalid device IDs
    // Note: We can't be certain that -1 or 1000 are invalid, but it's a reasonable assumption
    
    // Available memory for invalid devices should be 0
    REQUIRE(manager.get_available_memory(-1) == 0);
} 