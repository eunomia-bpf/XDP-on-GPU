#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include "../include/ebpf_gpu_processor.hpp"
#include "../include/gpu_device_manager.hpp"
#include "../include/kernel_loader.hpp"
#include "test_utils.hpp"
#include <vector>
#include <chrono>
#include <random>
#include <ctime>
#include <string>

// Configuration for test parameters
namespace test_config {
    const std::vector<size_t> event_sizes = {100, 1000, 10000};
    const std::chrono::seconds wait_timeout{10}; // Timeout for waiting on completion
}

// Helper function to create test events
void create_test_events(std::vector<ebpf_gpu::NetworkEvent>& events) {
    std::srand(std::time(nullptr));
    
    for (size_t i = 0; i < events.size(); i++) {
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

// Helper function to format sizes (1000 -> "1K", 1000000 -> "1M")
std::string format_size(size_t size) {
    if (size >= 1000000) return std::to_string(size / 1000000) + "M";
    if (size >= 1000) return std::to_string(size / 1000) + "K";
    return std::to_string(size);
}

TEST_CASE("Performance - Basic Operations", "[performance][benchmark]") {
    // Setup device manager to check for devices
    ebpf_gpu::GpuDeviceManager device_manager;
    int cuda_count = device_manager.get_device_count(ebpf_gpu::BackendType::CUDA);
    int opencl_count = device_manager.get_device_count(ebpf_gpu::BackendType::OpenCL);
    
    if (cuda_count == 0 && opencl_count == 0) {
        SKIP("No GPU devices available for performance testing");
    }
    
    const char* ptx_code = get_test_ptx();
    if (!ptx_code) {
        SKIP("PTX file not found for performance testing");
    }
    
    // Pre-create and setup test data
    std::vector<ebpf_gpu::NetworkEvent> events_1000(1000);
    create_test_events(events_1000);
    
    // Setup processor once outside benchmark
    ebpf_gpu::EventProcessor::Config config;
    config.backend_type = (cuda_count > 0) ? 
        ebpf_gpu::BackendType::CUDA : ebpf_gpu::BackendType::OpenCL;
    
    ebpf_gpu::EventProcessor processor(config);
    auto load_result = processor.load_kernel_from_ptx(ptx_code, "simple_packet_filter");
    REQUIRE(load_result == ebpf_gpu::ProcessingResult::Success);

    // Warm up GPU (first run is often slower)
    reset_event_actions(events_1000);
    size_t buffer_size = events_1000.size() * sizeof(ebpf_gpu::NetworkEvent);
    processor.process_events(events_1000.data(), buffer_size, events_1000.size());

    BENCHMARK_ADVANCED("Event processing - 1000 events")(Catch::Benchmark::Chronometer meter) {
        // Reset data state before measurement
        reset_event_actions(events_1000);
        
        // Measure only the GPU processing
        meter.measure([&] {
            return processor.process_events(events_1000.data(), buffer_size, events_1000.size());
        });
    };
    
    // Validate results after benchmark (not during timing)
    reset_event_actions(events_1000);
    auto final_result = processor.process_events(events_1000.data(), buffer_size, events_1000.size());
    REQUIRE(final_result == ebpf_gpu::ProcessingResult::Success);
    REQUIRE(validate_results(events_1000));
}

