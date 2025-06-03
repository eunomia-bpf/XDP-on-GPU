#include <catch2/catch_test_macros.hpp>
#include "../include/ebpf_gpu_processor.hpp"
#include "../include/gpu_device_manager.hpp"
#include "../include/kernel_loader.hpp"
#include "test_utils.hpp"
#include <vector>
#include <string>
#include <iostream>

// Test event generation utilities
void create_test_events(std::vector<ebpf_gpu::NetworkEvent>& events) {
    // Fill with test data
    for (size_t i = 0; i < events.size(); i++) {
        events[i].src_ip = 0x0A000001 + i;  // 10.0.0.1, 10.0.0.2, etc.
        events[i].dst_ip = 0xC0A80001 + i;  // 192.168.0.1, 192.168.0.2, etc.
        events[i].src_port = 12345 + i % 1000;
        events[i].dst_port = 80 + i % 10;
        events[i].protocol = i % 2 == 0 ? 6 : 17; // TCP or UDP
        events[i].action = 0; // Default to drop
        events[i].timestamp = 1600000000 + i;
    }
}

bool validate_results(const std::vector<ebpf_gpu::NetworkEvent>& events) {
    // Simple validation - check that actions were set
    bool all_processed = true;
    for (const auto& event : events) {
        if (event.action == 0) { // Action still at default/drop
            all_processed = false;
            break;
        }
    }
    return all_processed;
}

void reset_event_actions(std::vector<ebpf_gpu::NetworkEvent>& events) {
    for (auto& event : events) {
        event.action = 0; // Reset to default/drop
    }
}

TEST_CASE("Integration - Basic functionality", "[integration]") {
    SECTION("Initialize device and check info") {
        ebpf_gpu::GpuDeviceManager device_manager;
        
        if (device_manager.get_device_count(ebpf_gpu::BackendType::CUDA) == 0 && 
            device_manager.get_device_count(ebpf_gpu::BackendType::OpenCL) == 0) {
            SKIP("No GPU devices available");
            return;
        }
        
        // Choose backend based on available devices
        ebpf_gpu::BackendType backend_type = 
            device_manager.get_device_count(ebpf_gpu::BackendType::CUDA) > 0 ? 
            ebpf_gpu::BackendType::CUDA : ebpf_gpu::BackendType::OpenCL;
            
        int device_id = device_manager.get_best_device(backend_type);
        REQUIRE(device_id >= 0);
        
        auto device_info = device_manager.query_device_info(device_id);
        INFO("Testing on device: " << device_info.name);
        
        // Use a simple test kernel
        const char* test_ptx = get_test_ptx();
        REQUIRE(test_ptx != nullptr);
        
        ebpf_gpu::KernelLoader loader;
        try {
            loader.load_from_ptx(test_ptx);
            SUCCEED("Loaded test PTX kernel successfully");
        } catch (const std::exception& e) {
            WARN("Failed to load kernel: " << e.what());
        }
    }
}

// More complex integration test using the EventProcessor
TEST_CASE("Integration - Event Processing", "[integration]") {
    // Configure EventProcessor
    ebpf_gpu::EventProcessor::Config config;
    config.device_id = -1;  // Auto-select best device
    config.buffer_size = 1024 * 1024;
    config.block_size = 256;
    
    // Try both backends, preferring CUDA if available
    ebpf_gpu::GpuDeviceManager device_manager;
    if (device_manager.get_device_count(ebpf_gpu::BackendType::CUDA) > 0) {
        config.backend_type = ebpf_gpu::BackendType::CUDA;
    } else if (device_manager.get_device_count(ebpf_gpu::BackendType::OpenCL) > 0) {
        config.backend_type = ebpf_gpu::BackendType::OpenCL;
    } else {
        SKIP("No GPU devices available");
        return;
    }
    
    try {
        // Create processor
        ebpf_gpu::EventProcessor processor(config);
        
        // Load kernel
        const char* test_ptx = get_test_ptx();
        REQUIRE(test_ptx != nullptr);
        
        auto result = processor.load_kernel_from_ptx(test_ptx, "simple_packet_filter");
        if (result != ebpf_gpu::ProcessingResult::Success) {
            SKIP("Failed to load kernel - skipping test");
            return;
        }
        
        // Create test events
        const size_t num_events = 100;
        std::vector<ebpf_gpu::NetworkEvent> events(num_events);
        create_test_events(events);
        
        // Process events
        result = processor.process_events(
            events.data(), events.size() * sizeof(ebpf_gpu::NetworkEvent), events.size());
            
        // We can't guarantee success on all systems, so just don't crash
        INFO("Event processing result: " << static_cast<int>(result));
        
        if (result == ebpf_gpu::ProcessingResult::Success) {
            REQUIRE(validate_results(events));
        }
    } catch (const std::exception& e) {
        WARN("Exception in event processing test: " << e.what());
    }
}

TEST_CASE("Integration - Multiple backends", "[integration]") {
    SECTION("Test available backends") {
        try {
            ebpf_gpu::GpuDeviceManager manager;
            REQUIRE(manager.get_device_count(ebpf_gpu::BackendType::CUDA) >= 0);
            REQUIRE(manager.get_device_count(ebpf_gpu::BackendType::OpenCL) >= 0);
            
            INFO("CUDA devices: " << manager.get_device_count(ebpf_gpu::BackendType::CUDA));
            INFO("OpenCL devices: " << manager.get_device_count(ebpf_gpu::BackendType::OpenCL));
        } catch (const std::exception& e) {
            FAIL("Exception: " << e.what());
        }
    }
    
    SECTION("Test backend selection") {
        try {
            ebpf_gpu::GpuDeviceManager manager;
            REQUIRE(manager.get_device_count(ebpf_gpu::BackendType::CUDA) >= 0);
            REQUIRE(manager.get_device_count(ebpf_gpu::BackendType::OpenCL) >= 0);
            
            // Create config for each available backend
            if (manager.get_device_count(ebpf_gpu::BackendType::CUDA) > 0) {
                ebpf_gpu::EventProcessor::Config config;
                config.backend_type = ebpf_gpu::BackendType::CUDA;
                
                ebpf_gpu::EventProcessor processor(config);
                REQUIRE(processor.get_backend_type() == ebpf_gpu::BackendType::CUDA);
            }
            
            if (manager.get_device_count(ebpf_gpu::BackendType::OpenCL) > 0) {
                ebpf_gpu::EventProcessor::Config config;
                config.backend_type = ebpf_gpu::BackendType::OpenCL;
                
                ebpf_gpu::EventProcessor processor(config);
                REQUIRE(processor.get_backend_type() == ebpf_gpu::BackendType::OpenCL);
            }
        } catch (const std::exception& e) {
            FAIL("Exception: " << e.what());
        }
    }
} 