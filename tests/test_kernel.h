#pragma once

#include "test_utils.hpp"
#include <vector>
#include <chrono>
#include <cstring>

namespace ebpf_gpu {

// Kernel function names for testing
namespace kernel_names {
    constexpr const char* SIMPLE_PACKET_FILTER = "simple_packet_filter";
}

namespace cpu {

// CPU function pointer type
typedef void (*FilterFunction)(NetworkEvent* events, size_t num_events);

// CPU implementation of simple packet filter
inline void simple_packet_filter(NetworkEvent* events, size_t num_events) {
    for (size_t i = 0; i < num_events; i++) {
        NetworkEvent* event = &events[i];
        
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
}

// Benchmark helper for CPU processing
template<typename FilterFunc>
inline double benchmark_cpu_filter(FilterFunc filter_func, std::vector<NetworkEvent>& events, int iterations = 1) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        // Reset events for each iteration
        for (auto& event : events) {
            event.action = 0;
        }
        
        filter_func(events.data(), events.size());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return duration.count() / static_cast<double>(iterations); // Average time in microseconds
}

} // namespace cpu
} // namespace ebpf_gpu 