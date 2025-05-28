#include "cuda_event_processor.h"
#include <cuda_runtime.h>

// Simple packet filtering kernel
__global__ void simple_packet_filter(network_event_t *events, size_t num_events) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_events) {
        return;
    }
    
    network_event_t *event = &events[idx];
    
    // Simple filtering logic - drop packets from specific IP
    if (event->src_ip == 0xC0A80001) { // 192.168.0.1
        event->action = 0; // DROP
    } else if (event->protocol == 6) { // TCP
        event->action = 1; // PASS
    } else if (event->protocol == 17) { // UDP
        event->action = 1; // PASS
    } else {
        event->action = 0; // DROP unknown protocols
    }
}

// Simple filter for PTX testing
__global__ void simple_filter(network_event_t *events, size_t num_events) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_events) {
        return;
    }
    
    network_event_t *event = &events[idx];
    
    // Simple logic: drop packets from 192.168.0.1, pass others
    if (event->src_ip == 0xC0A80001) { // 192.168.0.1
        event->action = 0; // DROP
    } else {
        event->action = 1; // PASS
    }
} 