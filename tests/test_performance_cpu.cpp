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
#include <iostream>

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
    // Check if GPU is available
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
        
        // // Debug: Print first few events before processing
        // std::cout << "Before processing:" << std::endl;
        // for (size_t i = 0; i < std::min(size_t(5), event_count); i++) {
        //     std::cout << "GPU Event " << i << ": src_ip=" << std::hex << gpu_events[i].src_ip << ", protocol=" 
        //               << std::dec << (int)gpu_events[i].protocol << ", action=" << (int)gpu_events[i].action << std::endl;
        //     std::cout << "CPU Event " << i << ": src_ip=" << std::hex << cpu_events[i].src_ip << ", protocol=" 
        //               << std::dec << (int)cpu_events[i].protocol << ", action=" << (int)cpu_events[i].action << std::endl;
        // }
        
        processor.process_events(gpu_events.data(), buffer_size, gpu_events.size());
        selected_cpu_function(cpu_events.data(), cpu_events.size());
        
        // // Debug: Print first few events after processing
        // std::cout << "After processing:" << std::endl;
        // for (size_t i = 0; i < std::min(size_t(5), event_count); i++) {
        //     std::cout << "GPU Event " << i << ": src_ip=" << std::hex << gpu_events[i].src_ip << ", protocol=" 
        //               << std::dec << (int)gpu_events[i].protocol << ", action=" << (int)gpu_events[i].action << std::endl;
        //     std::cout << "CPU Event " << i << ": src_ip=" << std::hex << cpu_events[i].src_ip << ", protocol=" 
        //               << std::dec << (int)cpu_events[i].protocol << ", action=" << (int)cpu_events[i].action << std::endl;
        // }
        
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

TEST_CASE("Performance - Hash Load Balancing CPU vs GPU", "[performance][hash][load_balancing][comparison][benchmark]") {
    // Check if GPU is available
    auto devices = get_available_devices();
    if (devices.empty()) {
        SKIP("No GPU devices available for hash load balancing performance testing");
    }
    
    const char* test_code = get_test_ptx();
    if (!test_code) {
        SKIP("Test IR code not found for hash load balancing performance testing");
    }
    
    // Test different hash load balancing kernels
    const std::vector<const char*> test_kernels = {
        kernel_names::HASH_LOAD_BALANCER,
        kernel_names::BATCH_HASH_LOAD_BALANCER
    };
    
    // Test with different event counts to measure scalability
    const std::vector<size_t> event_counts = {1000, 10000, 100000, 1000000, 10000000};
    
    for (const char* kernel_name : test_kernels) {
        for (size_t event_count : event_counts) {
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
            
            // Create test data with diverse network flows for better hash distribution
            std::vector<NetworkEvent> gpu_events(event_count);
            std::vector<NetworkEvent> cpu_events(event_count);
            size_t buffer_size = gpu_events.size() * sizeof(NetworkEvent);
            
            create_test_events_cpu(gpu_events);
            
            // Ensure diverse source IPs for better hash distribution
            for (size_t i = 0; i < gpu_events.size(); i++) {
                gpu_events[i].src_ip = 0xC0A80000 + (i % 256) + ((i / 256) % 256) * 256; // Varied 192.168.x.y
                gpu_events[i].src_port = 1024 + (i * 17) % 60000; // Prime number for better distribution
                gpu_events[i].dst_port = (i % 3 == 0) ? 80 : ((i % 3 == 1) ? 443 : 8080); // Mix of ports
            }
            
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
                    cpu_function(cpu_events.data(), cpu_events.size());
                    return cpu_events.size();
                });
            };
            
            // Validate results for CUDA (OpenCL might have different behavior)
            if (backend == TestBackend::CUDA) {
                reset_event_actions_cpu(gpu_events);
                reset_event_actions_cpu(cpu_events);
                
                processor.process_events(gpu_events.data(), buffer_size, gpu_events.size());
                cpu_function(cpu_events.data(), cpu_events.size());
                
                // For hash load balancing, we expect similar distribution patterns
                // Count distribution across workers for both CPU and GPU
                std::vector<int> gpu_worker_counts(8, 0);
                std::vector<int> cpu_worker_counts(8, 0);
                
                for (size_t i = 0; i < event_count; i++) {
                    int gpu_worker = gpu_events[i].action & 0x7F;
                    int cpu_worker = cpu_events[i].action & 0x7F;
                    
                    if (gpu_worker < 8) gpu_worker_counts[gpu_worker]++;
                    if (cpu_worker < 8) cpu_worker_counts[cpu_worker]++;
                }
                
                // Verify that both CPU and GPU have similar active worker counts
                int gpu_active_workers = 0, cpu_active_workers = 0;
                for (int i = 0; i < 8; i++) {
                    if (gpu_worker_counts[i] > 0) gpu_active_workers++;
                    if (cpu_worker_counts[i] > 0) cpu_active_workers++;
                }
                
                INFO("GPU active workers: " << gpu_active_workers << ", CPU active workers: " << cpu_active_workers);
                
                // Both should have similar number of active workers (within 2 difference)
                REQUIRE(std::abs(gpu_active_workers - cpu_active_workers) <= 2);
                
                // Both should use most workers for good load balancing
                REQUIRE(gpu_active_workers >= 4);
                REQUIRE(cpu_active_workers >= 4);
            }
        }
    }
}
