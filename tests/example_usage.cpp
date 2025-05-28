#include "ebpf_gpu_processor.hpp"
#include "test_utils.hpp"
#include "test_kernel.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

using namespace ebpf_gpu;

// Example struct that could be any event type
struct CustomEvent {
    uint64_t id;
    uint32_t timestamp;
    uint16_t event_type;
    uint16_t flags;
    uint8_t data[32];
};

void example_network_event_processing() {
    std::cout << "=== Network Event Processing Example ===" << std::endl;
    
    try {
        // Check for available devices
        auto devices = get_available_devices();
        if (devices.empty()) {
            std::cout << "No CUDA devices found" << std::endl;
            return;
        }
        
        // Create processor
        EventProcessor::Config config;
        config.device_id = 0;
        config.buffer_size = 1024 * 1024; // 1MB buffer
        
        EventProcessor processor(config);
        
        // Load kernel
        const char* ptx_code = get_test_ptx();
        if (!ptx_code) {
            std::cout << "Failed to load PTX code" << std::endl;
            return;
        }
        
        ProcessingResult load_result = processor.load_kernel_from_ptx(ptx_code, kernel_names::DEFAULT_TEST_KERNEL);
        if (load_result != ProcessingResult::Success) {
            std::cerr << "Failed to load kernel, error code: " << static_cast<int>(load_result) << std::endl;
            return;
        }
        
        // Create some network events for testing
        std::vector<NetworkEvent> events(5);
        for (size_t i = 0; i < events.size(); i++) {
            events[i].data = nullptr;
            events[i].length = 100 + i * 10;
            events[i].timestamp = 1000000 + i;
            events[i].src_ip = 0xC0A80001 + i; // 192.168.0.x
            events[i].dst_ip = 0x08080808;     // 8.8.8.8
            events[i].src_port = 1024 + i;
            events[i].dst_port = 80;
            events[i].protocol = 6; // TCP
            events[i].action = 0;   // DROP
        }
        
        std::cout << "Processing " << events.size() << " network events..." << std::endl;
        
        // Process all events at once (zero-copy)
        size_t buffer_size = events.size() * sizeof(NetworkEvent);
        ProcessingResult result = processor.process_events(events.data(), buffer_size, events.size());
        
        if (result == ProcessingResult::Success) {
            std::cout << "Events processed successfully!" << std::endl;
            for (size_t i = 0; i < events.size(); i++) {
                std::cout << "Event " << i << " action: " << (int)events[i].action << std::endl;
            }
        } else {
            std::cout << "Failed to process events, result: " << static_cast<int>(result) << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
}

void example_single_event_processing() {
    std::cout << "\n=== Single Event Processing Example ===" << std::endl;
    
    try {
        EventProcessor processor;
        
        // Load kernel
        const char* ptx_code = get_test_ptx();
        if (!ptx_code) {
            std::cout << "Failed to load PTX code" << std::endl;
            return;
        }
        
        ProcessingResult load_result = processor.load_kernel_from_ptx(ptx_code, kernel_names::DEFAULT_TEST_KERNEL);
        if (load_result != ProcessingResult::Success) {
            std::cerr << "Failed to load kernel, error code: " << static_cast<int>(load_result) << std::endl;
            return;
        }
        
        // Create a single network event
        NetworkEvent event;
        event.data = nullptr;
        event.length = 128;
        event.timestamp = 1000000;
        event.src_ip = 0xC0A80001; // 192.168.0.1
        event.dst_ip = 0x08080808;  // 8.8.8.8
        event.src_port = 12345;
        event.dst_port = 80;
        event.protocol = 6; // TCP
        event.action = 0;   // DROP
        
        std::cout << "Processing single event..." << std::endl;
        std::cout << "Before: action = " << (int)event.action << std::endl;
        
        // Process single event
        ProcessingResult result = processor.process_event(&event, sizeof(NetworkEvent));
        
        if (result == ProcessingResult::Success) {
            std::cout << "Event processed successfully!" << std::endl;
            std::cout << "After: action = " << (int)event.action << std::endl;
        } else {
            std::cout << "Failed to process event, result: " << static_cast<int>(result) << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
}

void example_custom_event_processing() {
    std::cout << "\n=== Custom Event Processing Example ===" << std::endl;
    
    try {
        EventProcessor processor;
        
        // For custom events, you would need a different kernel
        // This is just to demonstrate the generic interface
        
        // Create custom events
        std::vector<CustomEvent> custom_events(3);
        for (size_t i = 0; i < custom_events.size(); i++) {
            custom_events[i].id = i + 1;
            custom_events[i].timestamp = 1000 + i;
            custom_events[i].event_type = 100 + i;
            custom_events[i].flags = 0;
            std::memset(custom_events[i].data, i, sizeof(custom_events[i].data));
        }
        
        std::cout << "Custom events created (would need appropriate kernel):" << std::endl;
        for (size_t i = 0; i < custom_events.size(); i++) {
            std::cout << "Event " << i << " - ID: " << custom_events[i].id 
                      << ", Type: " << custom_events[i].event_type << std::endl;
        }
        
        // Note: This would require a kernel designed for CustomEvent processing
        // size_t buffer_size = custom_events.size() * sizeof(CustomEvent);
        // ProcessingResult result = processor.process_events(custom_events.data(), buffer_size, custom_events.size());
        
        std::cout << "Custom event processing would work the same way with appropriate kernel!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "eBPF GPU Processor - Usage Examples" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Example 1: Processing multiple network events
    example_network_event_processing();
    
    // Example 2: Processing single events
    example_single_event_processing();
    
    // Example 3: Generic interface with custom events
    example_custom_event_processing();
    
    std::cout << "\n=== Key Benefits of the New Interface ===" << std::endl;
    std::cout << "1. Generic - accepts any memory buffer/event type" << std::endl;
    std::cout << "2. Zero-copy - direct buffer processing" << std::endl;
    std::cout << "3. Simplified - only process_event and process_events methods" << std::endl;
    std::cout << "4. Flexible - event definitions moved to application layer" << std::endl;
    
    return 0;
} 