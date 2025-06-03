#include "../include/ebpf_gpu_processor.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <future>

// A minimal OpenCL kernel for testing
const char* test_kernel = R"(
__kernel void simple_test(__global int* data) {
    int id = get_global_id(0);
    data[id] = id * 2; // Just double the value
}
)";

// Timeout helper function
template<typename F, typename... Args>
bool with_timeout(int timeout_ms, F&& f, Args&&... args) {
    auto future = std::async(std::launch::async, std::forward<F>(f), std::forward<Args>(args)...);
    auto status = future.wait_for(std::chrono::milliseconds(timeout_ms));
    return status == std::future_status::ready;
}

int main() {
    try {
        std::cout << "Starting simple OpenCL example (with timeouts)..." << std::endl;

        // Create a processor with OpenCL backend but minimal configuration
        ebpf_gpu::EventProcessor::Config config;
        config.backend_type = ebpf_gpu::BackendType::OpenCL;
        config.buffer_size = 1024;  // Small buffer for testing
        config.max_stream_count = 1; // Single stream for simplicity
        
        std::cout << "Creating EventProcessor..." << std::endl;
        
        // Set a 5 second timeout for processor creation
        ebpf_gpu::EventProcessor* processor_ptr = nullptr;
        
        bool success = with_timeout(5000, [&]() -> bool {
            try {
                processor_ptr = new ebpf_gpu::EventProcessor(config);
                return true;
            } catch (const std::exception& e) {
                std::cerr << "Error creating processor: " << e.what() << std::endl;
                return false;
            }
        });
        
        if (!success) {
            std::cerr << "Timed out creating EventProcessor" << std::endl;
            return 1;
        }
        
        if (!processor_ptr) {
            std::cerr << "Failed to create processor" << std::endl;
            return 1;
        }
        
        std::unique_ptr<ebpf_gpu::EventProcessor> processor(processor_ptr);
        
        std::cout << "EventProcessor created successfully" << std::endl;
        
        // Print device info
        auto device_info = processor->get_device_info();
        std::cout << "Using device: " << device_info.name << std::endl;
        std::cout << "Available memory: " << device_info.available_memory / (1024*1024) << " MB" << std::endl;
        
        // Load the kernel from source code
        std::cout << "Loading kernel..." << std::endl;
        auto result = processor->load_kernel_from_source(test_kernel, "simple_test", {}, {});
        if (result != ebpf_gpu::ProcessingResult::Success) {
            std::cerr << "Failed to load kernel: " << static_cast<int>(result) << std::endl;
            return 1;
        }
        
        std::cout << "Kernel loaded successfully" << std::endl;
        
        // Create test data - an array of integers
        const int data_size = 10;
        std::vector<int> test_data(data_size, 1);  // Initialize with 1s
        
        // Process the data using our kernel with timeout
        std::cout << "Processing data..." << std::endl;
        success = with_timeout(5000, [&]() -> bool {
            auto proc_result = processor->process_events(
                test_data.data(),
                test_data.size() * sizeof(int),
                test_data.size()
            );
            return proc_result == ebpf_gpu::ProcessingResult::Success;
        });
        
        if (!success) {
            std::cerr << "Timed out processing data" << std::endl;
            return 1;
        }
        
        // Print output (should be doubled values)
        std::cout << "Results: ";
        for (int i = 0; i < data_size; i++) {
            std::cout << test_data[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Example completed successfully" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 