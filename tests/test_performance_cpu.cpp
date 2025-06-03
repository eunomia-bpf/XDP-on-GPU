#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include "../include/ebpf_gpu_processor.hpp"
#include "../include/gpu_device_manager.hpp"
#include "test_utils.hpp"
#include "test_kernel.h"
#include <vector>
#include <chrono>
#include <ctime>

// Helper function to create test events
void create_test_events_cpu(std::vector<ebpf_gpu::NetworkEvent>& events) {
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

// Helper function to reset event actions
void reset_event_actions_cpu(std::vector<ebpf_gpu::NetworkEvent>& events) {
    for (auto& event : events) {
        event.action = 0;
    }
}

TEST_CASE("Performance - CPU vs GPU Comparison", "[performance][comparison][benchmark]") {
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
    
    // Setup CPU and GPU processing
    ebpf_gpu::EventProcessor::Config config;
    config.backend_type = (cuda_count > 0) ? 
        ebpf_gpu::BackendType::CUDA : ebpf_gpu::BackendType::OpenCL;
    
    ebpf_gpu::EventProcessor processor(config);
    auto load_result = processor.load_kernel_from_ptx(ptx_code, ebpf_gpu::kernel_names::SIMPLE_PACKET_FILTER);
    REQUIRE(load_result == ebpf_gpu::ProcessingResult::Success);
    
    // Test with a smaller event count
    const size_t event_count = 1000;
    
    std::vector<ebpf_gpu::NetworkEvent> gpu_events(event_count);
    std::vector<ebpf_gpu::NetworkEvent> cpu_events(event_count);
    size_t buffer_size = gpu_events.size() * sizeof(ebpf_gpu::NetworkEvent);
    
    create_test_events_cpu(gpu_events);
    cpu_events = gpu_events; // Ensure same input data
    
    // Warm up GPU
    reset_event_actions_cpu(gpu_events);
    processor.process_events(gpu_events.data(), buffer_size, gpu_events.size());
    
    BENCHMARK_ADVANCED("GPU processing - 1000 events")(Catch::Benchmark::Chronometer meter) {
        reset_event_actions_cpu(gpu_events);
        meter.measure([&] {
            return processor.process_events(gpu_events.data(), buffer_size, gpu_events.size());
        });
    };
    
    BENCHMARK_ADVANCED("CPU processing - 1000 events")(Catch::Benchmark::Chronometer meter) {
        reset_event_actions_cpu(cpu_events);
        meter.measure([&] {
            ebpf_gpu::cpu::simple_packet_filter(cpu_events.data(), cpu_events.size());
            return cpu_events.size();
        });
    };
    
    // Validate both produce same results
    reset_event_actions_cpu(gpu_events);
    reset_event_actions_cpu(cpu_events);
    
    processor.process_events(gpu_events.data(), buffer_size, gpu_events.size());
    ebpf_gpu::cpu::simple_packet_filter(cpu_events.data(), cpu_events.size());
    
    // Compare results
    bool results_match = true;
    for (size_t i = 0; i < event_count; i++) {
        if (gpu_events[i].action != cpu_events[i].action) {
            results_match = false;
            break;
        }
    }
    REQUIRE(results_match);
}
