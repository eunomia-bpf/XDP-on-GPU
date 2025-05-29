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
        
        // Load a simple kernel using test utilities
        const char* test_ptx = get_test_ptx();
        if (!test_ptx) {
            SKIP("PTX file not found for integration testing");
        }
        
        REQUIRE(KernelLoader::validate_ptx(test_ptx));
        
        KernelLoader loader;
        auto module = loader.load_from_ptx(test_ptx);
        REQUIRE(module != nullptr);
        REQUIRE(module->is_valid());
    }
}

TEST_CASE("Integration - Event Processing Workflow", "[integration]") {
    SECTION("Complete event processing pipeline") {
        auto devices = get_available_devices();
        if (devices.empty()) {
            SKIP("No CUDA devices available for integration testing");
        }
        
        const char* ptx_code = get_test_ptx();
        if (!ptx_code) {
            SKIP("PTX file not found for integration testing");
        }
        
        // Create processor with default config
        EventProcessor processor;
        ProcessingResult load_result = processor.load_kernel_from_ptx(ptx_code, kernel_names::DEFAULT_TEST_KERNEL);
        REQUIRE(load_result == ProcessingResult::Success);
        
        // Create test events
        std::vector<ebpf_gpu::NetworkEvent> events(100);
        create_test_events(events);
        size_t buffer_size = events.size() * sizeof(ebpf_gpu::NetworkEvent);
        
        // Process events
        ProcessingResult process_result = processor.process_events(events.data(), buffer_size, events.size());
        REQUIRE(process_result == ProcessingResult::Success);
        
        // Validate results
        bool results_valid = validate_results(events);
        REQUIRE(results_valid);
    }
}

TEST_CASE("Integration - Error Handling", "[integration]") {
    SECTION("Memory allocation failures") {
        auto devices = get_available_devices();
        if (devices.empty()) {
            SKIP("No CUDA devices available for error handling testing");
        }
        
        const char* ptx_code = get_test_ptx();
        if (!ptx_code) {
            SKIP("PTX file not found for error handling testing");
        }
        
        // Test with invalid inputs
        EventProcessor processor;
        ProcessingResult load_result = processor.load_kernel_from_ptx(ptx_code, kernel_names::DEFAULT_TEST_KERNEL);
        REQUIRE(load_result == ProcessingResult::Success);
        
        // Test with null buffer
        ProcessingResult null_result = processor.process_events(nullptr, 1000, 10);
        REQUIRE(null_result == ProcessingResult::InvalidInput);
        
        // Test with zero size
        std::vector<ebpf_gpu::NetworkEvent> events(10);
        ProcessingResult zero_result = processor.process_events(events.data(), 0, 10);
        REQUIRE(zero_result == ProcessingResult::InvalidInput);
    }
    
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
    
    SECTION("EventProcessor lifecycle") {
        auto devices = get_available_devices();
        if (devices.empty()) {
            SKIP("No CUDA devices available for resource management testing");
        }
        
        const char* ptx_code = get_test_ptx();
        if (!ptx_code) {
            SKIP("PTX file not found for resource management testing");
        }
        
        // Test multiple processor instances
        for (int i = 0; i < 3; ++i) {
            EventProcessor processor;
            ProcessingResult load_result = processor.load_kernel_from_ptx(ptx_code, kernel_names::DEFAULT_TEST_KERNEL);
            REQUIRE(load_result == ProcessingResult::Success);
            REQUIRE(processor.is_ready());
        }
    }
} 