#pragma once

#include "test_utils.hpp"
#include <vector>
#include <chrono>
#include <cstring>

namespace ebpf_gpu {

// Configurable kernel function names
namespace kernel_names {
    constexpr const char* SIMPLE_PACKET_FILTER = "_ZN8ebpf_gpu20simple_packet_filterEPNS_12NetworkEventEm";
    constexpr const char* PORT_BASED_FILTER = "_ZN8ebpf_gpu17port_based_filterEPNS_12NetworkEventEm";
    constexpr const char* MINIMAL_FILTER = "_ZN8ebpf_gpu14minimal_filterEPNS_12NetworkEventEm";
    constexpr const char* COMPLEX_FILTER = "_ZN8ebpf_gpu14complex_filterEPNS_12NetworkEventEm";
    constexpr const char* STATEFUL_FILTER = "_ZN8ebpf_gpu15stateful_filterEPNS_12NetworkEventEm";
    constexpr const char* HASH_LOAD_BALANCER = "_ZN8ebpf_gpu18hash_load_balancerEPNS_12NetworkEventEm";
    constexpr const char* BATCH_HASH_LOAD_BALANCER = "_ZN8ebpf_gpu24batch_hash_load_balancerEPNS_12NetworkEventEm";
    
    // Default test kernel (can be overridden by TEST_KERNEL_NAME macro)
    constexpr const char* DEFAULT_TEST_KERNEL = SIMPLE_PACKET_FILTER;
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

// CPU implementation of port-based filter
inline void port_based_filter(NetworkEvent* events, size_t num_events) {
    for (size_t i = 0; i < num_events; i++) {
        NetworkEvent* event = &events[i];
        
        // Port-based filtering
        if (event->dst_port == 22 || event->dst_port == 23) {
            // Block SSH and Telnet
            event->action = 0; // DROP
        } else if (event->dst_port == 80 || event->dst_port == 443) {
            // Allow HTTP and HTTPS
            event->action = 1; // PASS
        } else if (event->dst_port >= 1024 && event->dst_port <= 65535) {
            // Allow high ports
            event->action = 1; // PASS
        } else {
            // Block everything else
            event->action = 0; // DROP
        }
    }
}

// CPU implementation of minimal filter
inline void minimal_filter(NetworkEvent* events, size_t num_events) {
    for (size_t i = 0; i < num_events; i++) {
        // Minimal processing - just mark as processed
        events[i].action = 1; // PASS
    }
}

// CPU implementation of complex filter
inline void complex_filter(NetworkEvent* events, size_t num_events) {
    for (size_t i = 0; i < num_events; i++) {
        NetworkEvent* event = &events[i];
        
        // Complex multi-condition filtering
        bool should_drop = false;
        
        // Check for suspicious IP ranges
        uint32_t src_network = event->src_ip & 0xFFFF0000;
        if (src_network == 0x0A000000 || // 10.0.0.0/16
            src_network == 0xAC100000 || // 172.16.0.0/16
            src_network == 0xC0A80000) { // 192.168.0.0/16
            // Private networks - apply stricter rules
            if (event->dst_port < 1024 && event->dst_port != 80 && event->dst_port != 443) {
                should_drop = true;
            }
        }
        
        // Check packet size
        if (event->length > 1500 || event->length < 64) {
            should_drop = true;
        }
        
        // Protocol-specific rules
        if (event->protocol == 1) { // ICMP
            should_drop = true; // Block all ICMP
        }
        
        event->action = should_drop ? 0 : 1;
    }
}

// CPU implementation of stateful filter (simplified)
inline void stateful_filter(NetworkEvent* events, size_t num_events) {
    // Simple stateful implementation using static variables
    static std::vector<uint32_t> connection_count(256, 0);
    
    for (size_t i = 0; i < num_events; i++) {
        NetworkEvent* event = &events[i];
        
        // Simple connection counting (hash by source IP)
        uint32_t hash = event->src_ip % 256;
        connection_count[hash]++;
        
        // Rate limiting - if too many connections from same source
        if (connection_count[hash] > 10) {
            event->action = 0; // DROP
        } else {
            event->action = 1; // PASS
        }
    }
}

// CPU implementation of FNV-1a hash function
inline uint32_t cpu_fnv1a_hash(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port, uint16_t dst_port) {
    // FNV-1a hash algorithm for load balancing
    uint32_t hash = 2166136261U; // FNV offset basis
    const uint32_t prime = 16777619U; // FNV prime
    
    // Hash source IP
    hash ^= (src_ip & 0xFF);
    hash *= prime;
    hash ^= ((src_ip >> 8) & 0xFF);
    hash *= prime;
    hash ^= ((src_ip >> 16) & 0xFF);
    hash *= prime;
    hash ^= ((src_ip >> 24) & 0xFF);
    hash *= prime;
    
    // Hash destination IP
    hash ^= (dst_ip & 0xFF);
    hash *= prime;
    hash ^= ((dst_ip >> 8) & 0xFF);
    hash *= prime;
    hash ^= ((dst_ip >> 16) & 0xFF);
    hash *= prime;
    hash ^= ((dst_ip >> 24) & 0xFF);
    hash *= prime;
    
    // Hash source port
    hash ^= (src_port & 0xFF);
    hash *= prime;
    hash ^= ((src_port >> 8) & 0xFF);
    hash *= prime;
    
    // Hash destination port
    hash ^= (dst_port & 0xFF);
    hash *= prime;
    hash ^= ((dst_port >> 8) & 0xFF);
    hash *= prime;
    
    return hash;
}

// CPU implementation of hash load balancer
inline void hash_load_balancer(NetworkEvent* events, size_t num_events) {
    const uint32_t num_workers = 8; // Default number of workers
    
    for (size_t i = 0; i < num_events; i++) {
        NetworkEvent* event = &events[i];
        
        // Calculate hash for load balancing
        uint32_t hash = cpu_fnv1a_hash(event->src_ip, event->dst_ip, event->src_port, event->dst_port);
        
        // Assign to worker queue based on hash
        uint32_t worker_id = hash % num_workers;
        
        // Store worker assignment in the action field
        event->action = worker_id % 256;
        
        // Apply basic filtering - allow traffic only if assigned to even worker IDs
        if (worker_id % 2 == 0) {
            // Keep the worker assignment but mark as pass
            event->action = (event->action & 0x7F) | 0x80; // Set high bit for PASS
        }
    }
}

// CPU implementation of batch hash load balancer
inline void batch_hash_load_balancer(NetworkEvent* events, size_t num_events) {
    const uint32_t num_workers = 8; // Default number of workers
    std::vector<uint32_t> worker_counters(32, 0); // Support up to 32 workers
    
    for (size_t i = 0; i < num_events; i++) {
        NetworkEvent* event = &events[i];
        
        // Calculate hash for load balancing
        uint32_t hash = cpu_fnv1a_hash(event->src_ip, event->dst_ip, event->src_port, event->dst_port);
        
        // Assign to worker queue based on hash
        uint32_t worker_id = hash % num_workers;
        
        // Update worker counter
        if (worker_id < 32) {
            worker_counters[worker_id]++;
        }
        
        // Store worker assignment in the action field
        event->action = worker_id % 256;
        
        // Apply load balancing logic - balance traffic across workers
        uint32_t balanced_worker = (hash >> 16) % num_workers;
        if (balanced_worker != worker_id) {
            // Reassign for better balance
            event->action = balanced_worker % 256;
        }
        
        // Mark as processed (set high bit)
        event->action |= 0x80;
    }
}

// Configurable CPU test function selection
namespace test_config {
    // Function pointer mapping for CPU functions
    constexpr FilterFunction get_cpu_function_for_kernel(const char* kernel_name) {
        // Compare kernel names and return corresponding CPU function
        if (kernel_name == kernel_names::SIMPLE_PACKET_FILTER) {
            return simple_packet_filter;
        } else if (kernel_name == kernel_names::PORT_BASED_FILTER) {
            return port_based_filter;
        } else if (kernel_name == kernel_names::MINIMAL_FILTER) {
            return minimal_filter;
        } else if (kernel_name == kernel_names::COMPLEX_FILTER) {
            return complex_filter;
        } else if (kernel_name == kernel_names::STATEFUL_FILTER) {
            return stateful_filter;
        } else {
            return simple_packet_filter; // Default fallback
        }
    }
    
    // Default CPU test function (matches DEFAULT_TEST_KERNEL)
    constexpr FilterFunction DEFAULT_CPU_FUNCTION = get_cpu_function_for_kernel(kernel_names::DEFAULT_TEST_KERNEL);
}

// Helper function to get CPU function by kernel name at runtime
inline FilterFunction get_cpu_function_by_name(const char* kernel_name) {
    if (std::strcmp(kernel_name, kernel_names::SIMPLE_PACKET_FILTER) == 0) {
        return simple_packet_filter;
    } else if (std::strcmp(kernel_name, kernel_names::PORT_BASED_FILTER) == 0) {
        return port_based_filter;
    } else if (std::strcmp(kernel_name, kernel_names::MINIMAL_FILTER) == 0) {
        return minimal_filter;
    } else if (std::strcmp(kernel_name, kernel_names::COMPLEX_FILTER) == 0) {
        return complex_filter;
    } else if (std::strcmp(kernel_name, kernel_names::STATEFUL_FILTER) == 0) {
        return stateful_filter;
    } else if (std::strcmp(kernel_name, kernel_names::HASH_LOAD_BALANCER) == 0) {
        return hash_load_balancer;
    } else if (std::strcmp(kernel_name, kernel_names::BATCH_HASH_LOAD_BALANCER) == 0) {
        return batch_hash_load_balancer;
    } else {
        return simple_packet_filter; // Default fallback
    }
}

// Helper function to get function name for display
inline const char* get_function_display_name(const char* kernel_name) {
    if (std::strcmp(kernel_name, kernel_names::SIMPLE_PACKET_FILTER) == 0) {
        return "Simple Packet Filter";
    } else if (std::strcmp(kernel_name, kernel_names::PORT_BASED_FILTER) == 0) {
        return "Port-Based Filter";
    } else if (std::strcmp(kernel_name, kernel_names::MINIMAL_FILTER) == 0) {
        return "Minimal Filter";
    } else if (std::strcmp(kernel_name, kernel_names::COMPLEX_FILTER) == 0) {
        return "Complex Filter";
    } else if (std::strcmp(kernel_name, kernel_names::STATEFUL_FILTER) == 0) {
        return "Stateful Filter";
    } else if (std::strcmp(kernel_name, kernel_names::HASH_LOAD_BALANCER) == 0) {
        return "Hash Load Balancer";
    } else if (std::strcmp(kernel_name, kernel_names::BATCH_HASH_LOAD_BALANCER) == 0) {
        return "Batch Hash Load Balancer";
    } else {
        return "Unknown Filter";
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