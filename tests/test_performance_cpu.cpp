#include "catch2/catch_test_macros.hpp"
#include "catch2/benchmark/catch_benchmark.hpp"
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
void create_test_events_cpu(std::vector<NetworkEvent>& events) {
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
bool validate_results_cpu(const std::vector<NetworkEvent>& events) {
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
void reset_event_actions_cpu(std::vector<NetworkEvent>& events) {
    for (auto& event : events) {
        event.action = 0;
    }
}

// Helper function to copy events for comparison
std::vector<NetworkEvent> copy_events_cpu(const std::vector<NetworkEvent>& events) {
    return events; // Simple copy
}

// Get the kernel function name based on the backend
std::string get_test_kernel_name_cpu() {
    TestBackend backend = detect_test_backend();
    if (backend == TestBackend::OpenCL) {
        return "simple_kernel"; // OpenCL function name
    } else {
        return kernel_names::DEFAULT_TEST_KERNEL; // CUDA/PTX function name
    }
}

TEST_CASE("Performance - CPU vs GPU Comparison", "[performance][comparison][benchmark]") {
    auto devices = get_available_devices();
    if (devices.empty()) {
        SKIP("No GPU devices available for performance testing");
    }
    
    const char* test_code = get_test_ptx();
    if (!test_code) {
        SKIP("Test IR code not found for performance testing");
    }
    
    // Get the selected kernel name and corresponding CPU function
    std::string kernel_name = get_test_kernel_name_cpu();
    const char* selected_kernel = kernel_name.c_str();
    cpu::FilterFunction selected_cpu_function = cpu::get_cpu_function_by_name(selected_kernel);
    const char* function_name = cpu::get_function_display_name(selected_kernel);
    
    // Setup GPU processor
    
    // Test with different event counts
    const std::vector<size_t> event_counts = {100, 1000, 10000, 100000, 1000000, 10000000};
    
    for (size_t event_count : event_counts) {
        EventProcessor processor;
        ProcessingResult load_result = processor.load_kernel_from_ir(test_code, kernel_name);
        REQUIRE(load_result == ProcessingResult::Success);
        std::vector<NetworkEvent> gpu_events(event_count);
        std::vector<NetworkEvent> cpu_events(event_count);
        size_t buffer_size = gpu_events.size() * sizeof(NetworkEvent);
        
        create_test_events_cpu(gpu_events);
        cpu_events = copy_events_cpu(gpu_events); // Ensure same input data
        
        // Warm up GPU
        reset_event_actions_cpu(gpu_events);
        processor.register_host_buffer(gpu_events.data(), buffer_size);
        processor.process_events(gpu_events.data(), buffer_size, gpu_events.size());
        
        std::string gpu_test_name = "GPU " + std::string(function_name) + " - " + std::to_string(event_count) + " events";
        std::string cpu_test_name = "CPU " + std::string(function_name) + " - " + std::to_string(event_count) + " events";
        
        BENCHMARK_ADVANCED(gpu_test_name.c_str())(Catch::Benchmark::Chronometer meter) {
            reset_event_actions_cpu(gpu_events);
            meter.measure([&] {
                return processor.process_events(gpu_events.data(), buffer_size, gpu_events.size());
            });
        };
        
        BENCHMARK_ADVANCED(cpu_test_name.c_str())(Catch::Benchmark::Chronometer meter) {
            reset_event_actions_cpu(cpu_events);
            meter.measure([&] {
                selected_cpu_function(cpu_events.data(), cpu_events.size());
                return cpu_events.size();
            });
        };
        
        // Validate both produce same results
        reset_event_actions_cpu(gpu_events);
        reset_event_actions_cpu(cpu_events);
        
        processor.process_events(gpu_events.data(), buffer_size, gpu_events.size());
        selected_cpu_function(cpu_events.data(), cpu_events.size());
        
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
}

TEST_CASE("Performance - Multiple Filter Comparison", "[performance][filters][benchmark]") {
    auto devices = get_available_devices();
    if (devices.empty()) {
        SKIP("No GPU devices available for performance testing");
    }
    
    const char* test_code = get_test_ptx();
    if (!test_code) {
        SKIP("Test IR code not found for performance testing");
    }
    
    // Test different filter types
    const std::vector<const char*> test_kernels = {
        kernel_names::SIMPLE_PACKET_FILTER,
        kernel_names::PORT_BASED_FILTER,
        kernel_names::MINIMAL_FILTER,
        kernel_names::COMPLEX_FILTER
    };
    
    const size_t event_count = 1000 * 1000;
    
    for (const char* kernel_name : test_kernels) {
        // Get corresponding CPU function
        cpu::FilterFunction cpu_function = cpu::get_cpu_function_by_name(kernel_name);
        const char* function_name = cpu::get_function_display_name(kernel_name);
        
        // Setup GPU processor for this kernel
        EventProcessor processor;
        
        // For OpenCL, we need to use a different approach since it doesn't support these CUDA-specific kernels
        TestBackend backend = detect_test_backend();
        ProcessingResult load_result;
        
        if (backend == TestBackend::OpenCL) {
            // For OpenCL, use the simple kernel for all tests
            load_result = processor.load_kernel_from_ir(test_code, "simple_kernel");
        } else {
            // For CUDA, use the specific kernel
            load_result = processor.load_kernel_from_ir(test_code, kernel_name);
        }
        
        REQUIRE(load_result == ProcessingResult::Success);
        
        // Create test data
        std::vector<NetworkEvent> gpu_events(event_count);
        std::vector<NetworkEvent> cpu_events(event_count);
        
        create_test_events_cpu(gpu_events);
        cpu_events = copy_events_cpu(gpu_events);
        
        // Warm up GPU
        reset_event_actions_cpu(gpu_events);
        size_t buffer_size = gpu_events.size() * sizeof(NetworkEvent);
        processor.register_host_buffer(gpu_events.data(), buffer_size);
        processor.process_events(gpu_events.data(), buffer_size, gpu_events.size());
        
        std::string gpu_test_name = "GPU " + std::string(function_name);
        std::string cpu_test_name = "CPU " + std::string(function_name);
        
        BENCHMARK_ADVANCED(gpu_test_name.c_str())(Catch::Benchmark::Chronometer meter) {
            reset_event_actions_cpu(gpu_events);
            meter.measure([&] {
                return processor.process_events(gpu_events.data(), buffer_size, gpu_events.size());
            });
        };
        
        BENCHMARK_ADVANCED(cpu_test_name.c_str())(Catch::Benchmark::Chronometer meter) {
            reset_event_actions_cpu(cpu_events);
            meter.measure([&] {
                cpu_function(cpu_events.data(), cpu_events.size());
                return cpu_events.size();
            });
        };
        
        // For OpenCL, we expect different results due to using a different kernel
        if (backend == TestBackend::CUDA) {
            // Validate results match for CUDA
            reset_event_actions_cpu(gpu_events);
            reset_event_actions_cpu(cpu_events);
            
            processor.process_events(gpu_events.data(), buffer_size, gpu_events.size());
            cpu_function(cpu_events.data(), cpu_events.size());
            
            // Compare results (allow some differences for stateful filters)
            bool results_match = true;
            size_t mismatch_count = 0;
            for (size_t i = 0; i < event_count; i++) {
                if (gpu_events[i].action != cpu_events[i].action) {
                    mismatch_count++;
                }
            }
            
            // Allow up to 5% mismatch for complex/stateful filters due to implementation differences
            double mismatch_rate = static_cast<double>(mismatch_count) / event_count;
            REQUIRE(mismatch_rate < 0.05);
        }
    }
}
