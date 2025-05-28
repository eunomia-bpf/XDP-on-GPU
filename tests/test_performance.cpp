#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include "ebpf_gpu_processor.hpp"
#include <vector>
#include <chrono>
#include <random>
#include <cstring>

using namespace ebpf_gpu;

// Helper function to create test events
void create_test_events(std::vector<NetworkEvent>& events) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducible results
    std::uniform_int_distribution<uint32_t> ip_dist(0xC0A80000, 0xC0A800FF); // 192.168.0.x
    std::uniform_int_distribution<uint16_t> port_dist(1024, 65535);
    std::uniform_int_distribution<uint32_t> len_dist(64, 1500);
    std::uniform_int_distribution<uint8_t> proto_dist(0, 1); // TCP or UDP
    
    for (size_t i = 0; i < events.size(); i++) {
        events[i].data = nullptr;
        events[i].length = len_dist(gen);
        events[i].timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        events[i].src_ip = ip_dist(gen);
        events[i].dst_ip = 0x08080808; // 8.8.8.8
        events[i].src_port = port_dist(gen);
        events[i].dst_port = (i % 2) ? 80 : 443;
        events[i].protocol = proto_dist(gen) ? 6 : 17; // TCP or UDP
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

// Helper function to get PTX code
const char* get_test_ptx() {
    // This would normally load from a file, but for testing we'll use a simple stub
    static const char* ptx_stub = R"(
.version 7.0
.address_size 64

.visible .entry _Z20simple_packet_filterPN8ebpf_gpu12NetworkEventEm(
    .param .u64 _Z20simple_packet_filterPN8ebpf_gpu12NetworkEventEm_param_0,
    .param .u64 _Z20simple_packet_filterPN8ebpf_gpu12NetworkEventEm_param_1
)
{
    .reg .u64 %rd<3>;
    .reg .u32 %r<3>;
    
    ld.param.u64 %rd1, [_Z20simple_packet_filterPN8ebpf_gpu12NetworkEventEm_param_0];
    ld.param.u64 %rd2, [_Z20simple_packet_filterPN8ebpf_gpu12NetworkEventEm_param_1];
    
    // Simple processing: set action to 1 (PASS)
    mov.u32 %r1, 1;
    st.global.u8 [%rd1+25], %r1; // action field offset
    
    ret;
}
)";
    return ptx_stub;
}

TEST_CASE("Performance - Basic Operations", "[performance]") {
    auto devices = get_available_devices();
    if (devices.empty()) {
        SKIP("No CUDA devices available for performance testing");
    }
    
    // Pre-create test data
    std::vector<NetworkEvent> events_10(10);
    std::vector<NetworkEvent> events_100(100);
    std::vector<NetworkEvent> events_1000(1000);
    
    create_test_events(events_10);
    create_test_events(events_100);
    create_test_events(events_1000);
    
    const char* ptx_code = get_test_ptx();
    
    BENCHMARK("Complete workflow - 10 events") {
        try {
            EventProcessor processor;
            processor.load_kernel_from_ptx(ptx_code, "_Z20simple_packet_filterPN8ebpf_gpu12NetworkEventEm");
            
            // Reset actions before processing
            for (auto& event : events_10) event.action = 0;
            
            ProcessingResult result = processor.process_events(events_10);
            
            // Validate results
            if (result == ProcessingResult::Success) {
                REQUIRE(validate_results(events_10));
            }
            return static_cast<int>(result);
        } catch (...) {
            return -1;
        }
    };
    
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
    
    BENCHMARK("Complete workflow - 1000 events") {
        try {
            EventProcessor processor;
            processor.load_kernel_from_ptx(ptx_code, "_Z20simple_packet_filterPN8ebpf_gpu12NetworkEventEm");
            
            // Reset actions before processing
            for (auto& event : events_1000) event.action = 0;
            
            ProcessingResult result = processor.process_events(events_1000);
            
            // Validate results
            if (result == ProcessingResult::Success) {
                REQUIRE(validate_results(events_1000));
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
    
    // Setup once
    EventProcessor processor;
    const char* ptx_code = get_test_ptx();
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
