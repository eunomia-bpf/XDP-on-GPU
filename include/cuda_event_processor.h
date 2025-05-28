#ifndef CUDA_EVENT_PROCESSOR_H
#define CUDA_EVENT_PROCESSOR_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Event structure - simple network packet representation
struct network_event_t {
    uint8_t *data;      // Packet data
    uint32_t length;    // Packet length
    uint64_t timestamp; // Timestamp
    uint32_t src_ip;    // Source IP
    uint32_t dst_ip;    // Destination IP
    uint16_t src_port;  // Source port
    uint16_t dst_port;  // Destination port
    uint8_t protocol;   // Protocol (TCP=6, UDP=17, etc.)
    uint8_t action;     // Processing result (0=drop, 1=pass, 2=redirect)
};

// Processor handle
struct processor_handle_t {
    void *cuda_module;
    void *cuda_function;
    void *device_buffer;
    size_t buffer_size;
    void *cuda_context;
    int device_id;
};

// Interface 1: Accept PTX or kernel function
// Load PTX code and prepare for execution
int load_ptx_kernel(processor_handle_t *handle, const char *ptx_code, const char *function_name);

// Load pre-compiled kernel function
int load_kernel_function(processor_handle_t *handle, const char *kernel_file, const char *function_name);

// Interface 2: Process events accessing a buffer pointer
// Process a batch of network events
int process_events(processor_handle_t *handle, network_event_t *events, size_t num_events);

// Process events from a raw buffer (for zero-copy scenarios)
int process_events_buffer(processor_handle_t *handle, void *buffer, size_t buffer_size, size_t num_events);

// Utility functions
int init_processor(processor_handle_t *handle, int device_id, size_t buffer_size);
int cleanup_processor(processor_handle_t *handle);
int get_cuda_device_count();
const char* get_last_error();

#ifdef __cplusplus
}
#endif

#endif // CUDA_EVENT_PROCESSOR_H 