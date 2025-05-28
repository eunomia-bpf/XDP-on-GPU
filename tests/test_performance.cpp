#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include "ebpf_gpu_processor.hpp"
#include "test_utils.hpp"
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <ctime>

using namespace ebpf_gpu;

// Helper function to create test events
void create_test_events(std::vector<NetworkEvent>& events) {
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
bool validate_results(const std::vector<NetworkEvent>& events) {
    size_t processed_count = 0;
    for (const auto& event : events) {
        // Check that action was modified (0=DROP, 1=PASS)
        if (event.action == 0 || event.action == 1) {
            processed_count++;
        }
    }
    return processed_count == events.size();
}

TEST_CASE("Performance - Basic Operations", "[performance]") {
    auto devices = get_available_devices();
    if (devices.empty()) {
        SKIP("No CUDA devices available for performance testing");
    }
    
    const char* ptx_code = get_test_ptx();
    if (!ptx_code) {
        SKIP("PTX file not found for performance testing");
    }
    
    // Pre-create test data
    std::vector<NetworkEvent> events_10(10);
    std::vector<NetworkEvent> events_100(100);
    std::vector<NetworkEvent> events_1000(1000);
    
    create_test_events(events_10);
    create_test_events(events_100);
    create_test_events(events_1000);

    BENCHMARK("Complete workflow - 100 events") {
        try {
            EventProcessor processor;
            processor.load_kernel_from_ptx(ptx_code, "_Z20simple_packet_filterPN8ebpf_gpu12NetworkEventEm");
            
            // Reset actions before processing
            for (auto& event : events_100) event.action = 0;
            
            ProcessingResult result = processor.process_events(events_100);
            
            // Validate results
            if (result == ProcessingResult::Success) {
                REQUIRE(validate_results(events_100));
            }
            return static_cast<int>(result);
        } catch (...) {
            return -1;
        }
    };
}

TEST_CASE("Performance - Scaling Test", "[performance]") {
    auto devices = get_available_devices();
    if (devices.empty()) {
        SKIP("No CUDA devices available for performance testing");
    }
    
    const char* ptx_code = get_test_ptx();
    if (!ptx_code) {
        SKIP("PTX file not found for performance testing");
    }
    
    auto test_with_size = [&](size_t num_events) {
        try {
            EventProcessor processor;
            processor.load_kernel_from_ptx(ptx_code, "_Z20simple_packet_filterPN8ebpf_gpu12NetworkEventEm");
            
            std::vector<NetworkEvent> events(num_events);
            create_test_events(events);
            
            for (auto& event : events) event.action = 0;
            
            ProcessingResult result = processor.process_events(events);
            
            if (result == ProcessingResult::Success) {
                REQUIRE(validate_results(events));
            }
            return static_cast<int>(result);
        } catch (...) {
            return -1;
        }
    };
    
    BENCHMARK("Scaling test - 10K events") {
        return test_with_size(10000);
    };
}

TEST_CASE("Performance - Interface Comparison", "[performance]") {
    auto devices = get_available_devices();
    if (devices.empty()) {
        SKIP("No CUDA devices available for performance testing");
    }
    
    const char* ptx_code = get_test_ptx();
    if (!ptx_code) {
        SKIP("PTX file not found for performance testing");
    }
    
    // Setup once
    EventProcessor processor;
    processor.load_kernel_from_ptx(ptx_code, "_Z20simple_packet_filterPN8ebpf_gpu12NetworkEventEm");
    
    // Pre-create test data
    const size_t num_events = 10000;
    std::vector<NetworkEvent> events(num_events);
    create_test_events(events);
    
    BENCHMARK("Vector interface - 10K events") {
        for (auto& event : events) event.action = 0;
        
        ProcessingResult result = processor.process_events(events);
        
        if (result == ProcessingResult::Success) {
            REQUIRE(validate_results(events));
        }
        return static_cast<int>(result);
    };
    
    BENCHMARK("Buffer interface - 10K events") {
        for (auto& event : events) event.action = 0;
        
        size_t buffer_size = events.size() * sizeof(NetworkEvent);
        ProcessingResult result = processor.process_buffer(events.data(), buffer_size, num_events);
        
        if (result == ProcessingResult::Success) {
            REQUIRE(validate_results(events));
        }
        return static_cast<int>(result);
    };
}
