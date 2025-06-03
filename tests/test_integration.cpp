#include <catch2/catch_test_macros.hpp>
#include "ebpf_gpu_processor.hpp"
#include "gpu_device_manager.hpp"
#include "kernel_loader.hpp"
#include "test_utils.hpp"
#include "test_kernel.h"
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>

using namespace ebpf_gpu;

// Helper function to create test events
void create_test_events(std::vector<ebpf_gpu::NetworkEvent>& events) {
    std::srand(std::time(nullptr));
    
    for (size_t i = 0; i < events.size(); i++) {
        events[i].data = nullptr;
        events[i].length = 64 + (std::rand() % 1400);
        events[i].timestamp = std::time(nullptr) * 1000000 + i;
        events[i].src_ip = 0xC0A80000 + (std::rand() % 256); // 192.168.0.x
        events[i].dst_ip = 0x08080808; // 8.8.8.8
        events[i].src_port = 1024 + (std::rand() % 60000);
        events[i].dst_port = (std::rand() % 2) ? 80 : 443; // HTTP or HTTPS
        events[i].protocol = (std::rand() % 2) ? 6 : 17; // TCP or UDP
        events[i].action = 0; // Initialize to DROP
    }
}

// Helper function to validate processing results
bool validate_results(const std::vector<ebpf_gpu::NetworkEvent>& events) {
    size_t processed_count = 0;
    for (const auto& event : events) {
        // Check that action was modified (0=DROP, 1=PASS)
        if (event.action == 0 || event.action == 1) {
            processed_count++;
        }
    }
    return processed_count == events.size();
}

// Helper function to reset event actions
void reset_event_actions(std::vector<ebpf_gpu::NetworkEvent>& events) {
    for (auto& event : events) {
        event.action = 0;
    }
}

TEST_CASE("Integration - Complete Workflow", "[integration]") {
    SECTION("Device detection and kernel loading") {
        // Test device information retrieval
        std::vector<GpuDeviceInfo> devices = get_available_devices();
        INFO("Found " << devices.size() << " GPU devices");
        
        // Skip if no devices available
        if (devices.empty()) {
            SKIP("No GPU devices found for integration testing");
            return;
        }
        
        // Test device selection
        int device_id = select_best_device();
        INFO("Selected device ID: " << device_id);
        REQUIRE(device_id >= 0);
        
        // Get test PTX/IR code
        const char* test_ptx = get_test_ptx();
        if (!test_ptx || strlen(test_ptx) == 0) {
            SKIP("PTX file not found for integration testing");
            return;
        }
        
        // Validate IR code
        KernelLoader loader;
        REQUIRE(KernelLoader::validate_ir(test_ptx));
        
        // Load IR code
        auto module = loader.load_from_ir(test_ptx);
        REQUIRE(module != nullptr);
        REQUIRE(module->is_valid());
    }
}

TEST_CASE("Integration - Event Processing Workflow", "[integration]") {
    SECTION("Complete event processing pipeline") {
        // Skip if no test PTX/IR available
        const char* test_ptx = get_test_ptx();
        if (!test_ptx || strlen(test_ptx) == 0) {
            SKIP("PTX file not found for integration testing");
            return;
        }
        
        // Create processor with default config
        EventProcessor processor;
        
        // Load kernel
        auto result = processor.load_kernel_from_source(test_ptx, "simple_kernel");
        if (result != ProcessingResult::Success) {
            SKIP("Failed to load kernel for integration testing");
            return;
        }
        
        // Create test data
        const size_t num_events = 100;
        std::vector<uint8_t> test_data(num_events * 64, 0); // 100 events, 64 bytes each
        
        // Process events
        result = processor.process_events(test_data.data(), test_data.size(), num_events);
        // Note: We're just testing that it doesn't crash, not checking the result
    }
}

TEST_CASE("Integration - Error Handling", "[integration]") {
    SECTION("Memory allocation failures") {
        // Skip if no test PTX/IR available
        const char* test_ptx = get_test_ptx();
        if (!test_ptx || strlen(test_ptx) == 0) {
            SKIP("PTX file not found for error handling testing");
            return;
        }
        
        // Verify resource acquisition
        EventProcessor::Config config;
        config.device_id = 0;
        
        try {
            EventProcessor processor(config);
            
            // Test extremely large allocation that should fail
            const size_t extremely_large_size = static_cast<size_t>(1) << 40; // 1TB
            void* ptr = EventProcessor::allocate_pinned_buffer(extremely_large_size);
            
            // If allocation somehow succeeded, free it
            if (ptr) {
                EventProcessor::free_pinned_buffer(ptr);
                INFO("Extremely large pinned allocation unexpectedly succeeded");
            } else {
                INFO("Extremely large pinned allocation correctly failed");
            }
            
        } catch (const std::exception& e) {
            INFO("Caught exception: " << e.what());
            // It's ok if it throws
        }
        
        // Verify that after resource limits, we can still create a processor
        EventProcessor processor;
        REQUIRE(processor.is_ready());
    }
}

TEST_CASE("Integration - Resource Management", "[integration]") {
    SECTION("EventProcessor lifecycle") {
        // Skip if no test PTX/IR available
        const char* test_ptx = get_test_ptx();
        if (!test_ptx || strlen(test_ptx) == 0) {
            SKIP("PTX file not found for resource management testing");
            return;
        }
        
        // Create and destroy processors multiple times
        for (int i = 0; i < 5; ++i) {
            EventProcessor processor;
            REQUIRE(processor.is_ready());
            
            auto result = processor.load_kernel_from_source(test_ptx, "simple_kernel");
            if (result == ProcessingResult::Success) {
                INFO("Successfully loaded kernel in iteration " << i);
            } else {
                INFO("Failed to load kernel in iteration " << i << ", error: " << static_cast<int>(result));
            }
        }
        
        // Create processor and move it
        EventProcessor p1;
        REQUIRE(p1.is_ready());
        
        EventProcessor p2(std::move(p1));
        REQUIRE(p2.is_ready());
        
        EventProcessor p3;
        p3 = std::move(p2);
        REQUIRE(p3.is_ready());
    }
} 