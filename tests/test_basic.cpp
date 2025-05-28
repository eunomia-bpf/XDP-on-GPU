#include "ebpf_gpu_processor.hpp"
#include "test_utils.hpp"
#include "test_kernel.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

using namespace std;
using namespace ebpf_gpu;

void print_event(const ebpf_gpu::NetworkEvent& event, int index) {
    cout << "Event " << index << ":" << endl;
    cout << "  Length: " << event.length << endl;
    cout << "  Src IP: 0x" << hex << event.src_ip << dec << endl;
    cout << "  Dst IP: 0x" << hex << event.dst_ip << dec << endl;
    cout << "  Src Port: " << event.src_port << endl;
    cout << "  Dst Port: " << event.dst_port << endl;
    cout << "  Protocol: " << (int)event.protocol << endl;
    cout << "  Action: " << (int)event.action << " (";
    cout << (event.action == 0 ? "DROP" : 
             event.action == 1 ? "PASS" : "REDIRECT") << ")" << endl;
    cout << endl;
}

void create_sample_events(vector<ebpf_gpu::NetworkEvent>& events) {
    srand(time(nullptr));
    
    for (size_t i = 0; i < events.size(); i++) {
        events[i].data = nullptr;
        events[i].length = 64 + (rand() % 1400);
        events[i].timestamp = time(nullptr) * 1000000 + i;
        events[i].src_ip = 0xC0A80000 + (rand() % 256); // 192.168.0.x
        events[i].dst_ip = 0x08080808; // 8.8.8.8
        events[i].src_port = 1024 + (rand() % 60000);
        events[i].dst_port = (rand() % 2) ? 80 : 443; // HTTP or HTTPS
        events[i].protocol = (rand() % 2) ? 6 : 17; // TCP or UDP
        events[i].action = 0; // Initialize to DROP
    }
}

int test_cpp_api() {
    cout << "=== Testing Modern C++ API ===" << endl;
    
    try {
        // Check available devices
        auto devices = get_available_devices();
        if (devices.empty()) {
            cout << "No CUDA devices found" << endl;
            return -1;
        }
        
        cout << "Found " << devices.size() << " CUDA device(s)" << endl;
        
        // Create processor with default config
        EventProcessor::Config config;
        config.device_id = 0;
        config.buffer_size = 1024 * 1024;
        
        EventProcessor processor(config);
        
        // Get PTX code
        const char* ptx_code = get_test_ptx();
        if (!ptx_code) {
            cout << "Failed to load PTX code" << endl;
            return -1;
        }
        
        // Load PTX kernel
        processor.load_kernel_from_ptx(ptx_code, kernel_names::DEFAULT_TEST_KERNEL);
        
        // Create sample events
        vector<ebpf_gpu::NetworkEvent> events(10);
        create_sample_events(events);
        
        cout << "Events before processing:" << endl;
        for (size_t i = 0; i < 3; i++) {
            print_event(events[i], i);
        }
        
        // Process events using the new interface
        size_t buffer_size = events.size() * sizeof(ebpf_gpu::NetworkEvent);
        ProcessingResult result = processor.process_events(events.data(), buffer_size, events.size());
        if (result != ProcessingResult::Success) {
            cout << "Failed to process events" << endl;
            return -1;
        }
        
        cout << "Events after processing:" << endl;
        for (size_t i = 0; i < 3; i++) {
            print_event(events[i], i);
        }
        
        cout << "C++ API test completed successfully!" << endl << endl;
        return 0;
        
    } catch (const exception& e) {
        cout << "Exception: " << e.what() << endl;
        return -1;
    }
}

int test_single_event_interface() {
    cout << "=== Testing Single Event Interface ===" << endl;
    
    try {
        EventProcessor processor;
        
        // Get PTX code
        const char* ptx_code = get_test_ptx();
        if (!ptx_code) {
            cout << "Failed to load PTX code" << endl;
            return -1;
        }
        
        // Load PTX kernel
        processor.load_kernel_from_ptx(ptx_code, kernel_names::DEFAULT_TEST_KERNEL);
        
        // Create a single event
        ebpf_gpu::NetworkEvent event;
        event.data = nullptr;
        event.length = 128;
        event.timestamp = time(nullptr) * 1000000;
        event.src_ip = 0xC0A80001; // 192.168.0.1
        event.dst_ip = 0x08080808; // 8.8.8.8
        event.src_port = 12345;
        event.dst_port = 80;
        event.protocol = 6; // TCP
        event.action = 0; // DROP
        
        cout << "Single event before processing:" << endl;
        print_event(event, 0);
        
        // Process single event
        ProcessingResult result = processor.process_event(&event, sizeof(ebpf_gpu::NetworkEvent));
        if (result != ProcessingResult::Success) {
            cout << "Failed to process single event" << endl;
            return -1;
        }
        
        cout << "Single event after processing:" << endl;
        print_event(event, 0);
        
        cout << "Single event interface test completed successfully!" << endl << endl;
        return 0;
        
    } catch (const exception& e) {
        cout << "Exception: " << e.what() << endl;
        return -1;
    }
}

int test_buffer_interface() {
    cout << "=== Testing Multiple Events Interface ===" << endl;
    
    try {
        EventProcessor processor;
        
        // Get PTX code
        const char* ptx_code = get_test_ptx();
        if (!ptx_code) {
            cout << "Failed to load PTX code" << endl;
            return -1;
        }
        
        // Load PTX kernel
        processor.load_kernel_from_ptx(ptx_code, kernel_names::DEFAULT_TEST_KERNEL);
        
        // Create sample events in a buffer
        const size_t num_events = 5;
        size_t buffer_size = num_events * sizeof(ebpf_gpu::NetworkEvent);
        vector<ebpf_gpu::NetworkEvent> events(num_events);
        create_sample_events(events);
        
        cout << "Buffer events before processing:" << endl;
        for (size_t i = 0; i < num_events; i++) {
            print_event(events[i], i);
        }
        
        // Process events using new buffer interface
        ProcessingResult result = processor.process_events(events.data(), buffer_size, num_events);
        if (result != ProcessingResult::Success) {
            cout << "Failed to process events buffer" << endl;
            return -1;
        }
        
        cout << "Buffer events after processing:" << endl;
        for (size_t i = 0; i < num_events; i++) {
            print_event(events[i], i);
        }
        
        cout << "Multiple events interface test completed successfully!" << endl << endl;
        return 0;
        
    } catch (const exception& e) {
        cout << "Exception: " << e.what() << endl;
        return -1;
    }
}

int main() {
    cout << "CUDA Event Processor - Basic Test" << endl;
    cout << "==================================" << endl << endl;
    
    // Test modern C++ API
    if (test_cpp_api() != 0) {
        cout << "C++ API test failed" << endl;
        return -1;
    }
    
    // Test single event interface
    if (test_single_event_interface() != 0) {
        cout << "Single event interface test failed" << endl;
        return -1;
    }
    
    // Test multiple events interface
    if (test_buffer_interface() != 0) {
        cout << "Multiple events interface test failed" << endl;
        return -1;
    }
    
    cout << "All tests completed successfully!" << endl;
    return 0;
} 