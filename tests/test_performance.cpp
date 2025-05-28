#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include "ebpf_gpu_processor.hpp"
#include "test_utils.hpp"
#include "test_kernel.h"
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

// Helper function to reset event actions
void reset_event_actions(std::vector<NetworkEvent>& events) {
    for (auto& event : events) {
        event.action = 0;
    }
}

// Helper function to copy events for comparison
std::vector<NetworkEvent> copy_events(const std::vector<NetworkEvent>& events) {
    return events; // Simple copy
}

TEST_CASE("Performance - Basic Operations", "[performance][benchmark]") {
    auto devices = get_available_devices();
    if (devices.empty()) {
        SKIP("No CUDA devices available for performance testing");
    }
    
    const char* ptx_code = get_test_ptx();
    if (!ptx_code) {
        SKIP("PTX file not found for performance testing");
    }
    
    // Pre-create and setup test data
    std::vector<NetworkEvent> events_1000(1000);
    create_test_events(events_1000);
    
    // Setup processor once outside benchmark
    EventProcessor processor;
    processor.load_kernel_from_ptx(ptx_code, kernel_names::DEFAULT_TEST_KERNEL);
    
    // Warm up GPU (first run is often slower)
    reset_event_actions(events_1000);
    size_t buffer_size = events_1000.size() * sizeof(NetworkEvent);
    processor.process_events(events_1000.data(), buffer_size, events_1000.size());

    BENCHMARK_ADVANCED("Event processing - 1000 events")(Catch::Benchmark::Chronometer meter) {
        // Reset data state before measurement
        reset_event_actions(events_1000);
        size_t buffer_size = events_1000.size() * sizeof(NetworkEvent);
        
        // Measure only the GPU processing
        meter.measure([&] {
            return processor.process_events(events_1000.data(), buffer_size, events_1000.size());
        });
    };
    
    // Validate results after benchmark (not during timing)
    reset_event_actions(events_1000);
    ProcessingResult final_result = processor.process_events(events_1000.data(), buffer_size, events_1000.size());
    REQUIRE(final_result == ProcessingResult::Success);
    REQUIRE(validate_results(events_1000));
}

TEST_CASE("Performance - Scaling Test", "[performance][benchmark]") {
    auto devices = get_available_devices();
    if (devices.empty()) {
        SKIP("No CUDA devices available for performance testing");
    }
    
    const char* ptx_code = get_test_ptx();
    if (!ptx_code) {
        SKIP("PTX file not found for performance testing");
    }
    
    // Setup processor once
    EventProcessor processor;
    processor.load_kernel_from_ptx(ptx_code, kernel_names::DEFAULT_TEST_KERNEL);
    
    // Test 1K events
    {
        std::vector<NetworkEvent> events(1000);
        create_test_events(events);
        size_t buffer_size = events.size() * sizeof(NetworkEvent);
        
        // Warm up
        reset_event_actions(events);
        processor.process_events(events.data(), buffer_size, events.size());
        
        BENCHMARK_ADVANCED("Scaling - 1K events")(Catch::Benchmark::Chronometer meter) {
            reset_event_actions(events);
            meter.measure([&] {
                return processor.process_events(events.data(), buffer_size, events.size());
            });
        };
        
        // Validate after benchmark
        reset_event_actions(events);
        ProcessingResult final_result = processor.process_events(events.data(), buffer_size, events.size());
        REQUIRE(final_result == ProcessingResult::Success);
        REQUIRE(validate_results(events));
    }
    
    // Test 10K events
    {
        std::vector<NetworkEvent> events(10000);
        create_test_events(events);
        size_t buffer_size = events.size() * sizeof(NetworkEvent);
        
        // Warm up
        reset_event_actions(events);
        processor.process_events(events.data(), buffer_size, events.size());
        
        BENCHMARK_ADVANCED("Scaling - 10K events")(Catch::Benchmark::Chronometer meter) {
            reset_event_actions(events);
            meter.measure([&] {
                return processor.process_events(events.data(), buffer_size, events.size());
            });
        };
        
        // Validate after benchmark
        reset_event_actions(events);
        ProcessingResult final_result = processor.process_events(events.data(), buffer_size, events.size());
        REQUIRE(final_result == ProcessingResult::Success);
        REQUIRE(validate_results(events));
    }
    
    // Test 100K events
    {
        std::vector<NetworkEvent> events(100000);
        create_test_events(events);
        size_t buffer_size = events.size() * sizeof(NetworkEvent);
        
        // Warm up
        reset_event_actions(events);
        processor.process_events(events.data(), buffer_size, events.size());
        
        BENCHMARK_ADVANCED("Scaling - 100K events")(Catch::Benchmark::Chronometer meter) {
            reset_event_actions(events);
            meter.measure([&] {
                return processor.process_events(events.data(), buffer_size, events.size());
            });
        };
        
        // Validate after benchmark
        reset_event_actions(events);
        ProcessingResult final_result = processor.process_events(events.data(), buffer_size, events.size());
        REQUIRE(final_result == ProcessingResult::Success);
        REQUIRE(validate_results(events));
    }

    // Test 1M events
    {
        std::vector<NetworkEvent> events(1000000);
        create_test_events(events);
        size_t buffer_size = events.size() * sizeof(NetworkEvent);
        
        // Warm up
        reset_event_actions(events);
        processor.process_events(events.data(), buffer_size, events.size());

        BENCHMARK_ADVANCED("Scaling - 1M events")(Catch::Benchmark::Chronometer meter) {
            reset_event_actions(events);
            meter.measure([&] {
                return processor.process_events(events.data(), buffer_size, events.size());
            });
        };

        // Validate after benchmark
        reset_event_actions(events);
        ProcessingResult final_result = processor.process_events(events.data(), buffer_size, events.size());
        REQUIRE(final_result == ProcessingResult::Success);
        REQUIRE(validate_results(events));
    }
}

TEST_CASE("Performance - Single vs Multiple Events", "[performance][benchmark]") {
    auto devices = get_available_devices();
    if (devices.empty()) {
        SKIP("No CUDA devices available for performance testing");
    }
    
    const char* ptx_code = get_test_ptx();
    if (!ptx_code) {
        SKIP("PTX file not found for performance testing");
    }
    
    // Setup processor once
    EventProcessor processor;
    processor.load_kernel_from_ptx(ptx_code, kernel_names::DEFAULT_TEST_KERNEL);
    
    // Pre-create test data
    const size_t num_events = 1000;
    std::vector<NetworkEvent> events(num_events);
    create_test_events(events);
    
    // Warm up
    reset_event_actions(events);
    size_t buffer_size = events.size() * sizeof(NetworkEvent);
    processor.process_events(events.data(), buffer_size, events.size());
    
    BENCHMARK_ADVANCED("Multiple events - 1K events batch")(Catch::Benchmark::Chronometer meter) {
        reset_event_actions(events);
        size_t buffer_size = events.size() * sizeof(NetworkEvent);
        
        meter.measure([&] {
            return processor.process_events(events.data(), buffer_size, events.size());
        });
    };
    
    BENCHMARK_ADVANCED("Single events - 1K individual calls")(Catch::Benchmark::Chronometer meter) {
        NetworkEvent single_event = events[0];
        
        meter.measure([&] {
            int total_result = 0;
            for (size_t i = 0; i < num_events; i++) {
                single_event.action = 0;
                ProcessingResult result = processor.process_event(&single_event, sizeof(NetworkEvent));
                total_result += static_cast<int>(result);
            }
            return total_result;
        });
    };
    
    // Validate results after benchmarks
    reset_event_actions(events);
    ProcessingResult final_result = processor.process_events(events.data(), buffer_size, events.size());
    REQUIRE(final_result == ProcessingResult::Success);
    REQUIRE(validate_results(events));
    
    // Validate single event processing
    NetworkEvent single_event = events[0];
    single_event.action = 0;
    ProcessingResult single_result = processor.process_event(&single_event, sizeof(NetworkEvent));
    REQUIRE(single_result == ProcessingResult::Success);
    REQUIRE((single_event.action == 0 || single_event.action == 1));
}

TEST_CASE("Performance - Memory Transfer vs Compute", "[performance][benchmark]") {
    auto devices = get_available_devices();
    if (devices.empty()) {
        SKIP("No CUDA devices available for performance testing");
    }
    
    const char* ptx_code = get_test_ptx();
    if (!ptx_code) {
        SKIP("PTX file not found for performance testing");
    }
    
    EventProcessor processor;
    processor.load_kernel_from_ptx(ptx_code, kernel_names::DEFAULT_TEST_KERNEL);
    
    // Test small events (64B each)
    {
        const size_t event_count = 100;
        const size_t event_size_multiplier = 1;
        
        std::vector<std::vector<uint8_t>> event_buffers;
        event_buffers.reserve(event_count);
        
        for (size_t i = 0; i < event_count; i++) {
            size_t padded_size = sizeof(NetworkEvent) + (event_size_multiplier * 64);
            event_buffers.emplace_back(padded_size, 0);
            
            // Place NetworkEvent at the beginning
            NetworkEvent* event = reinterpret_cast<NetworkEvent*>(event_buffers[i].data());
            event->data = nullptr;
            event->length = 64;
            event->timestamp = i;
            event->src_ip = 0xC0A80001;
            event->dst_ip = 0x08080808;
            event->src_port = 1024;
            event->dst_port = 80;
            event->protocol = 6;
            event->action = 0;
        }
        
        // Warm up
        for (auto& buffer : event_buffers) {
            processor.process_event(buffer.data(), buffer.size());
        }
        
        BENCHMARK_ADVANCED("Small events - 100 events (64B each)")(Catch::Benchmark::Chronometer meter) {
            meter.measure([&] {
                int total_result = 0;
                for (auto& buffer : event_buffers) {
                    // Reset action
                    reinterpret_cast<NetworkEvent*>(buffer.data())->action = 0;
                    ProcessingResult result = processor.process_event(buffer.data(), buffer.size());
                    total_result += static_cast<int>(result);
                }
                return total_result;
            });
        };
    }
    
    // Test large events (1KB each)
    {
        const size_t event_count = 100;
        const size_t event_size_multiplier = 16;
        
        std::vector<std::vector<uint8_t>> event_buffers;
        event_buffers.reserve(event_count);
        
        for (size_t i = 0; i < event_count; i++) {
            size_t padded_size = sizeof(NetworkEvent) + (event_size_multiplier * 64);
            event_buffers.emplace_back(padded_size, 0);
            
            // Place NetworkEvent at the beginning
            NetworkEvent* event = reinterpret_cast<NetworkEvent*>(event_buffers[i].data());
            event->data = nullptr;
            event->length = 64;
            event->timestamp = i;
            event->src_ip = 0xC0A80001;
            event->dst_ip = 0x08080808;
            event->src_port = 1024;
            event->dst_port = 80;
            event->protocol = 6;
            event->action = 0;
        }
        
        // Warm up
        for (auto& buffer : event_buffers) {
            processor.process_event(buffer.data(), buffer.size());
        }
        
        BENCHMARK_ADVANCED("Large events - 100 events (1KB each)")(Catch::Benchmark::Chronometer meter) {
            meter.measure([&] {
                int total_result = 0;
                for (auto& buffer : event_buffers) {
                    // Reset action
                    reinterpret_cast<NetworkEvent*>(buffer.data())->action = 0;
                    ProcessingResult result = processor.process_event(buffer.data(), buffer.size());
                    total_result += static_cast<int>(result);
                }
                return total_result;
            });
        };
    }
}
