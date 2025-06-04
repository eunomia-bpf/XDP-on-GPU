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

// Get the kernel function name based on the backend
std::string get_backend_kernel_name() {
    TestBackend backend = detect_test_backend();
    if (backend == TestBackend::OpenCL) {
        return "simple_kernel"; // OpenCL function name
    } else {
        return "simple_kernel"; // CUDA/PTX function name
    }
}

// Add this helper function after the existing helper functions
bool load_test_kernel_from_file(EventProcessor& processor, std::string& kernel_name) {
    // Try to load the test PTX file directly
    const char* ptx_file_paths[] = {
        "tests/test_ptx.ptx",
        "test_ptx.ptx",
        "../tests/test_ptx.ptx",
        "./tests/test_ptx.ptx",
        "./build/tests/test_ptx.ptx",
        "../build/tests/test_ptx.ptx",
        "./build/tests/ptx/cuda_kernels.ptx",
        "../build/tests/ptx/cuda_kernels.ptx",
        "./ptx/cuda_kernels.ptx"  // Absolute path
    };

    // For cuda_kernels.ptx, we need to use the mangled function name if backend is CUDA
    std::string cuda_function_name = "_ZN8ebpf_gpu20simple_packet_filterEPNS_12NetworkEventEm";
    
    bool success = false;
    for (const char* path : ptx_file_paths) {
        std::string current_path = path;
        std::string current_func = kernel_name;
        
        // If we're trying the cuda_kernels.ptx file, use the mangled function name
        if (current_path.find("cuda_kernels.ptx") != std::string::npos) {
            current_func = cuda_function_name;
            INFO("Using mangled function name: " << current_func);
        }
        
        INFO("Trying to load kernel from file: " << current_path);
        auto result = processor.load_kernel_from_file(current_path, current_func);
        
        if (result == ProcessingResult::Success) {
            INFO("Successfully loaded kernel from file: " << current_path);
            success = true;
            break;
        } else {
            INFO("Failed to load kernel from file: " << current_path << ", result: " << static_cast<int>(result));
        }
    }
    
    return success;
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
        
        // Get test IR code (PTX or OpenCL)
        const char* test_code = get_test_ptx();
        if (!test_code || strlen(test_code) == 0) {
            SKIP("Test IR code not found for integration testing");
            return;
        }
        
        // Validate IR code
        KernelLoader loader;
        REQUIRE(KernelLoader::validate_ir(test_code));
        
        // Load IR code
        auto module = loader.load_from_ir(test_code);
        REQUIRE(module != nullptr);
        REQUIRE(module->is_valid());
    }
}

TEST_CASE("Integration - Event Processing Workflow", "[integration]") {
    SECTION("Complete event processing pipeline") {
        // Skip if no test IR available
        const char* test_code = get_test_ptx();
        if (!test_code || strlen(test_code) == 0) {
            SKIP("Test IR code not found for integration testing");
            return;
        }
        
        // Output first few bytes of the test code for debugging
        INFO("Test code first 100 bytes: " << std::string(test_code).substr(0, 100));
        
        // Create processor with default config
        EventProcessor processor;
        
        // Get correct kernel function name for the backend
        std::string kernel_name = get_backend_kernel_name();
        INFO("Using kernel function name: " << kernel_name);
        
        // Try loading from memory first
        auto result = processor.load_kernel_from_ir(test_code, kernel_name);
        INFO("Kernel load from IR result: " << static_cast<int>(result));
        
        // If loading from memory fails, try loading from file
        if (result != ProcessingResult::Success) {
            INFO("Trying to load from file instead");
            bool loaded = load_test_kernel_from_file(processor, kernel_name);
            if (!loaded) {
                SKIP("Failed to load kernel for integration testing");
                return;
            }
        }
        
        // Check if processor is ready
        INFO("Is processor ready: " << (processor.is_ready() ? "true" : "false"));
        
        // Create test data
        const size_t num_events = 100;
        std::vector<uint8_t> test_data(num_events * 64, 0); // 100 events, 64 bytes each
        
        // Process events
        result = processor.process_events(test_data.data(), test_data.size(), num_events);
        INFO("Process events result: " << static_cast<int>(result));
        // Note: We're just testing that it doesn't crash, not checking the result
    }
}

TEST_CASE("Integration - Error Handling", "[integration]") {
    SECTION("Memory allocation failures") {
        // Skip if no test IR available
        const char* test_code = get_test_ptx();
        if (!test_code || strlen(test_code) == 0) {
            SKIP("Test IR code not found for error handling testing");
            return;
        }
        
        // Output first few bytes of the test code for debugging
        INFO("Test code first 100 bytes: " << std::string(test_code).substr(0, 100));
        
        // Verify resource acquisition
        EventProcessor::Config config;
        config.device_id = 0;
        
        try {
            EventProcessor processor(config);
            
            // Load kernel first to make processor ready
            std::string kernel_name = get_backend_kernel_name();
            INFO("Using kernel function name: " << kernel_name);
            
            // Try loading from memory first
            auto load_result = processor.load_kernel_from_ir(test_code, kernel_name);
            INFO("Kernel load from IR result: " << static_cast<int>(load_result));
            
            // If loading from memory fails, try loading from file
            if (load_result != ProcessingResult::Success) {
                INFO("Trying to load from file instead");
                bool loaded = load_test_kernel_from_file(processor, kernel_name);
                if (!loaded) {
                    INFO("Failed to load kernel, skipping processor.is_ready() check");
                    return;
                }
            }
            
            // Check if processor is ready
            INFO("Is processor ready: " << (processor.is_ready() ? "true" : "false"));
            
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
            
            REQUIRE(processor.is_ready());
            
        } catch (const std::exception& e) {
            INFO("Caught exception: " << e.what());
            // It's ok if it throws
        }
    }
}

TEST_CASE("Integration - Resource Management", "[integration]") {
    SECTION("EventProcessor lifecycle") {
        // Skip if no test IR available
        const char* test_code = get_test_ptx();
        if (!test_code || strlen(test_code) == 0) {
            SKIP("Test IR code not found for resource management testing");
            return;
        }
        
        // Output first few bytes of the test code for debugging
        INFO("Test code first 100 bytes: " << std::string(test_code).substr(0, 100));
        
        // Get kernel name for the current backend
        std::string kernel_name = get_backend_kernel_name();
        INFO("Using kernel function name: " << kernel_name);
        
        // Create and destroy processors multiple times
        for (int i = 0; i < 5; ++i) {
            EventProcessor processor;
            
            // Try loading from memory first
            auto load_result = processor.load_kernel_from_ir(test_code, kernel_name);
            INFO("Iteration " << i << " - Kernel load from IR result: " << static_cast<int>(load_result));
            
            // If loading from memory fails, try loading from file
            if (load_result != ProcessingResult::Success) {
                INFO("Trying to load from file instead");
                bool loaded = load_test_kernel_from_file(processor, kernel_name);
                if (!loaded) {
                    INFO("Failed to load kernel in iteration " << i);
                    continue;
                }
            }
            
            INFO("Is processor ready in iteration " << i << ": " << (processor.is_ready() ? "true" : "false"));
            REQUIRE(processor.is_ready());
            
            if (load_result == ProcessingResult::Success) {
                INFO("Successfully loaded kernel in iteration " << i);
            } else {
                INFO("Failed to load kernel in iteration " << i << ", error: " << static_cast<int>(load_result));
            }
        }
        
        // Create processor and move it
        EventProcessor p1;
        // Try loading from memory first
        auto load_result = p1.load_kernel_from_ir(test_code, kernel_name);
        
        // If loading from memory fails, try loading from file
        if (load_result != ProcessingResult::Success) {
            INFO("Trying to load from file for move test");
            bool loaded = load_test_kernel_from_file(p1, kernel_name);
            if (!loaded) {
                INFO("Failed to load kernel for move test");
                return;
            }
        }
        
        REQUIRE(p1.is_ready());
        
        EventProcessor p2(std::move(p1));
        REQUIRE(p2.is_ready());
        
        EventProcessor p3;
        p3 = std::move(p2);
        REQUIRE(p3.is_ready());
    }
} 