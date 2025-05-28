#include "cuda_event_processor.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

using namespace std;

// Load PTX from pre-generated file
const char* get_ptx_code() {
    static char* ptx_code = nullptr;
    if (ptx_code) return ptx_code;
    
#ifdef PTX_FILE_PATH
    const char* ptx_path = PTX_FILE_PATH;
#else
    const char* ptx_path = "cuda_kernels.ptx";
#endif
    
    cout << "Loading PTX from: " << ptx_path << endl;
    
    // Read the pre-generated PTX file
    FILE* file = fopen(ptx_path, "r");
    if (!file) {
        cout << "Error: Could not open PTX file: " << ptx_path << endl;
        return nullptr;
    }
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    ptx_code = (char*)malloc(size + 1);
    fread(ptx_code, 1, size, file);
    ptx_code[size] = '\0';
    fclose(file);
    
    cout << "PTX loaded successfully!" << endl;
    return ptx_code;
}

void print_event(network_event_t *event, int index) {
    cout << "Event " << index << ":" << endl;
    cout << "  Length: " << event->length << endl;
    cout << "  Src IP: 0x" << hex << event->src_ip << dec << endl;
    cout << "  Dst IP: 0x" << hex << event->dst_ip << dec << endl;
    cout << "  Src Port: " << event->src_port << endl;
    cout << "  Dst Port: " << event->dst_port << endl;
    cout << "  Protocol: " << (int)event->protocol << endl;
    cout << "  Action: " << (int)event->action << " (";
    cout << (event->action == 0 ? "DROP" : 
             event->action == 1 ? "PASS" : "REDIRECT") << ")" << endl;
    cout << endl;
}

void create_sample_events(network_event_t *events, size_t num_events) {
    srand(time(nullptr));
    
    for (size_t i = 0; i < num_events; i++) {
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

int test_ptx_interface() {
    cout << "=== Testing PTX Interface ===" << endl;
    
    processor_handle_t handle;
    
    // Initialize processor
    if (init_processor(&handle, 0, 1024 * 1024) != 0) {
        cout << "Failed to initialize processor: " << get_last_error() << endl;
        return -1;
    }
    
    // Get PTX code
    const char* ptx_code = get_ptx_code();
    if (!ptx_code) {
        cout << "Failed to load PTX code" << endl;
        cleanup_processor(&handle);
        return -1;
    }
    
    // Load PTX kernel with correct mangled name
    if (load_ptx_kernel(&handle, ptx_code, "_Z20simple_packet_filterP15network_event_tm") != 0) {
        cout << "Failed to load PTX kernel: " << get_last_error() << endl;
        cleanup_processor(&handle);
        return -1;
    }
    
    // Create sample events
    const size_t num_events = 10;
    network_event_t *events = (network_event_t*)malloc(num_events * sizeof(network_event_t));
    create_sample_events(events, num_events);
    
    cout << "Events before processing:" << endl;
    for (size_t i = 0; i < 3; i++) {
        print_event(&events[i], i);
    }
    
    // Process events
    if (process_events(&handle, events, num_events) != 0) {
        cout << "Failed to process events: " << get_last_error() << endl;
        free(events);
        cleanup_processor(&handle);
        return -1;
    }
    
    cout << "Events after processing:" << endl;
    for (size_t i = 0; i < 3; i++) {
        print_event(&events[i], i);
    }
    
    free(events);
    cleanup_processor(&handle);
    cout << "PTX interface test completed successfully!" << endl << endl;
    return 0;
}

int test_buffer_interface() {
    cout << "=== Testing Buffer Interface ===" << endl;
    
    processor_handle_t handle;
    
    // Initialize processor
    if (init_processor(&handle, 0, 1024 * 1024) != 0) {
        cout << "Failed to initialize processor: " << get_last_error() << endl;
        return -1;
    }
    
    // Get PTX code
    const char* ptx_code = get_ptx_code();
    if (!ptx_code) {
        cout << "Failed to load PTX code" << endl;
        cleanup_processor(&handle);
        return -1;
    }
    
    // Load PTX kernel
    if (load_ptx_kernel(&handle, ptx_code, "_Z20simple_packet_filterP15network_event_tm") != 0) {
        cout << "Failed to load PTX kernel: " << get_last_error() << endl;
        cleanup_processor(&handle);
        return -1;
    }
    
    // Create sample events in a buffer
    const size_t num_events = 5;
    size_t buffer_size = num_events * sizeof(network_event_t);
    void *buffer = malloc(buffer_size);
    network_event_t *events = (network_event_t*)buffer;
    
    create_sample_events(events, num_events);
    
    cout << "Buffer events before processing:" << endl;
    for (size_t i = 0; i < num_events; i++) {
        print_event(&events[i], i);
    }
    
    // Process events using buffer interface
    if (process_events_buffer(&handle, buffer, buffer_size, num_events) != 0) {
        cout << "Failed to process events buffer: " << get_last_error() << endl;
        free(buffer);
        cleanup_processor(&handle);
        return -1;
    }
    
    cout << "Buffer events after processing:" << endl;
    for (size_t i = 0; i < num_events; i++) {
        print_event(&events[i], i);
    }
    
    free(buffer);
    cleanup_processor(&handle);
    cout << "Buffer interface test completed successfully!" << endl << endl;
    return 0;
}

int main() {
    cout << "CUDA Event Processor - Basic Test" << endl;
    cout << "==================================" << endl << endl;
    
    // Check CUDA device availability
    int device_count = get_cuda_device_count();
    if (device_count <= 0) {
        cout << "No CUDA devices found or CUDA initialization failed" << endl;
        cout << "Error: " << get_last_error() << endl;
        return -1;
    }
    
    cout << "Found " << device_count << " CUDA device(s)" << endl << endl;
    
    // Test PTX interface
    if (test_ptx_interface() != 0) {
        cout << "PTX interface test failed" << endl;
        return -1;
    }
    
    // Test buffer interface
    if (test_buffer_interface() != 0) {
        cout << "Buffer interface test failed" << endl;
        return -1;
    }
    
    cout << "All tests completed successfully!" << endl;
    return 0;
} 