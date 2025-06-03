#include "../include/ebpf_gpu_processor.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

// Helper function to read file content
std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

void run_with_backend(ebpf_gpu::BackendType backend_type, const std::string& kernel_path, const std::string& kernel_function) {
    std::cout << "Testing with " << (backend_type == ebpf_gpu::BackendType::CUDA ? "CUDA" : "OpenCL") << " backend" << std::endl;
    
    try {
        // Create event processor with selected backend
        ebpf_gpu::EventProcessor::Config config;
        config.backend_type = backend_type;
        config.device_id = -1; // Auto-select device
        config.buffer_size = 1024 * 1024; // 1MB
        config.block_size = 256; // Work group size
        
        ebpf_gpu::EventProcessor processor(config);
        
        // Get device info
        auto device_info = processor.get_device_info();
        
        // Check if we're running with a stub backend (no real GPU support)
        if (device_info.name.empty() && device_info.total_memory == 0) {
            std::cout << "WARNING: Using stub backend (no real GPU support)" << std::endl;
            std::cout << "This is likely because neither CUDA nor OpenCL is available on this system." << std::endl;
            return;
        }
        
        std::cout << "Device: " << device_info.name << std::endl;
        std::cout << "Memory: " << (device_info.total_memory / (1024 * 1024)) << " MB" << std::endl;
        
        // Load kernel from file
        std::string kernel_source;
        try {
            kernel_source = read_file(kernel_path);
        } catch (const std::exception& e) {
            std::cerr << "Failed to read kernel file: " << e.what() << std::endl;
            return;
        }
        
        auto result = processor.load_kernel_from_source(kernel_source, kernel_function);
        
        if (result != ebpf_gpu::ProcessingResult::Success) {
            std::cerr << "Failed to load kernel" << std::endl;
            return;
        }
        
        // Create test data
        const size_t num_events = 1024;
        std::vector<uint32_t> events(num_events, 5); // Initialize all events with value 5
        
        // Process events
        auto start = std::chrono::high_resolution_clock::now();
        
        result = processor.process_events(events.data(), events.size() * sizeof(uint32_t), events.size());
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        if (result != ebpf_gpu::ProcessingResult::Success) {
            std::cerr << "Failed to process events" << std::endl;
            return;
        }
        
        // Verify results
        bool correct = true;
        for (size_t i = 0; i < events.size(); i++) {
            if (events[i] != 6) { // Should be incremented by 1
                std::cerr << "Error at event " << i << ": expected 6, got " << events[i] << std::endl;
                correct = false;
                break;
            }
        }
        
        if (correct) {
            std::cout << "All events processed correctly!" << std::endl;
        }
        
        std::cout << "Processing time: " << elapsed.count() << " ms" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}

int main() {
    // Try CUDA backend first if available
#ifdef USE_CUDA_BACKEND
    try {
        run_with_backend(ebpf_gpu::BackendType::CUDA, "cuda_kernel.cu", "increment");
    } catch (const std::exception& e) {
        std::cerr << "CUDA backend failed: " << e.what() << std::endl;
        std::cerr << "Falling back to OpenCL backend..." << std::endl;
    }
#else
    std::cout << "CUDA backend not available (not compiled with USE_CUDA_BACKEND)" << std::endl;
#endif

    // Try OpenCL backend
#ifdef USE_OPENCL_BACKEND
    try {
        run_with_backend(ebpf_gpu::BackendType::OpenCL, "opencl_kernel.cl", "increment");
    } catch (const std::exception& e) {
        std::cerr << "OpenCL backend failed: " << e.what() << std::endl;
    }
#else
    std::cout << "OpenCL backend not available (not compiled with USE_OPENCL_BACKEND)" << std::endl;
#endif

    return 0;
} 