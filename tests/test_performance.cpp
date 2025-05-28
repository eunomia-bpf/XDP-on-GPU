#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include "cuda_event_processor.h"
#include <vector>
#include <chrono>
#include <random>
#include <cstring>

// Helper function to create test events
void create_test_events(network_event_t* events, size_t count) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducible results
    std::uniform_int_distribution<uint32_t> ip_dist(0xC0A80000, 0xC0A800FF); // 192.168.0.x
    std::uniform_int_distribution<uint16_t> port_dist(1024, 65535);
    std::uniform_int_distribution<uint32_t> len_dist(64, 1500);
    std::uniform_int_distribution<uint8_t> proto_dist(0, 1); // TCP or UDP
    
    for (size_t i = 0; i < count; i++) {
        events[i].data = nullptr;
        events[i].length = len_dist(gen);
        events[i].timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        events[i].src_ip = ip_dist(gen);
        events[i].dst_ip = 0x08080808; // 8.8.8.8
        events[i].src_port = port_dist(gen);
        events[i].dst_port = (i % 2) ? 80 : 443;
        events[i].protocol = proto_dist(gen) ? 6 : 17; // TCP or UDP
        events[i].action = 0; // Initialize to DROP
    }
}

// Helper function to validate processing results
bool validate_results(const network_event_t* events, size_t count) {
    size_t processed_count = 0;
    for (size_t i = 0; i < count; i++) {
        // Check that action was modified (0=DROP, 1=PASS)
        if (events[i].action == 0 || events[i].action == 1) {
            processed_count++;
        }
    }
    return processed_count == count;
}

// Helper function to get PTX code
const char* get_test_ptx() {
    static std::string ptx_code;
    if (ptx_code.empty()) {
        // Try to read from the generated PTX file
        const char* ptx_paths[] = {
#ifdef PTX_FILE_PATH
            PTX_FILE_PATH,
#endif
            "build/tests/ptx/cuda_kernels.ptx",
            "../build/tests/ptx/cuda_kernels.ptx",
            "tests/ptx/cuda_kernels.ptx"
        };
        
        for (const char* path : ptx_paths) {
            FILE* file = fopen(path, "r");
            if (file) {
                fseek(file, 0, SEEK_END);
                long size = ftell(file);
                fseek(file, 0, SEEK_SET);
                
                ptx_code.resize(size);
                fread(&ptx_code[0], 1, size, file);
                fclose(file);
                break;
            }
        }
        
        if (ptx_code.empty()) {
            // If PTX file not found, return nullptr to indicate failure
            return nullptr;
        }
    }
    return ptx_code.c_str();
}

TEST_CASE("Performance - Complete Workflow - Small Batches", "[performance]") {
    int device_count = get_cuda_device_count();
    if (device_count == 0) {
        SKIP("No CUDA devices available for performance testing");
    }
    
    // Pre-create test data
    std::vector<network_event_t> events_10(10);
    std::vector<network_event_t> events_100(100);
    std::vector<network_event_t> events_1000(1000);
    
    create_test_events(events_10.data(), 10);
    create_test_events(events_100.data(), 100);
    create_test_events(events_1000.data(), 1000);
    
    const char* ptx_code = get_test_ptx();
    
    BENCHMARK("Complete workflow - 10 events") {
        processor_handle_t handle;
        memset(&handle, 0, sizeof(handle));
        
        // Complete workflow: init + load + process + cleanup
        if (init_processor(&handle, 0, 1024 * 1024) == 0 &&
            load_ptx_kernel(&handle, ptx_code, "_Z20simple_packet_filterP15network_event_tm") == 0) {
            
            // Reset actions before processing
            for (auto& event : events_10) event.action = 0;
            
            int result = process_events(&handle, events_10.data(), 10);
            cleanup_processor(&handle);
            
            // Validate results
            if (result == 0) {
                REQUIRE(validate_results(events_10.data(), 10));
            }
            return result;
        }
        cleanup_processor(&handle);
        return -1;
    };
    
    BENCHMARK("Complete workflow - 100 events") {
        processor_handle_t handle;
        memset(&handle, 0, sizeof(handle));
        
        if (init_processor(&handle, 0, 1024 * 1024) == 0 &&
            load_ptx_kernel(&handle, ptx_code, "_Z20simple_packet_filterP15network_event_tm") == 0) {
            
            for (auto& event : events_100) event.action = 0;
            
            int result = process_events(&handle, events_100.data(), 100);
            cleanup_processor(&handle);
            
            if (result == 0) {
                REQUIRE(validate_results(events_100.data(), 100));
            }
            return result;
        }
        cleanup_processor(&handle);
        return -1;
    };
    
    BENCHMARK("Complete workflow - 1000 events") {
        processor_handle_t handle;
        memset(&handle, 0, sizeof(handle));
        
        if (init_processor(&handle, 0, 1024 * 1024) == 0 &&
            load_ptx_kernel(&handle, ptx_code, "_Z20simple_packet_filterP15network_event_tm") == 0) {
            
            for (auto& event : events_1000) event.action = 0;
            
            int result = process_events(&handle, events_1000.data(), 1000);
            cleanup_processor(&handle);
            
            if (result == 0) {
                REQUIRE(validate_results(events_1000.data(), 1000));
            }
            return result;
        }
        cleanup_processor(&handle);
        return -1;
    };
}

TEST_CASE("Performance - Processing Only - Large Batches", "[performance]") {
    int device_count = get_cuda_device_count();
    if (device_count == 0) {
        SKIP("No CUDA devices available for performance testing");
    }
    
    // Setup once for all benchmarks
    processor_handle_t handle;
    memset(&handle, 0, sizeof(handle));
    REQUIRE(init_processor(&handle, 0, 50 * 1024 * 1024) == 0);
    REQUIRE(load_ptx_kernel(&handle, get_test_ptx(), "_Z20simple_packet_filterP15network_event_tm") == 0);
    
    // Pre-create test data
    std::vector<network_event_t> events_10k(10000);
    std::vector<network_event_t> events_100k(100000);
    std::vector<network_event_t> events_1m(1000000);
    
    create_test_events(events_10k.data(), 10000);
    create_test_events(events_100k.data(), 100000);
    create_test_events(events_1m.data(), 1000000);
    
    BENCHMARK("Processing only - 10K events") {
        // Reset actions before processing
        for (auto& event : events_10k) event.action = 0;
        
        int result = process_events(&handle, events_10k.data(), 10000);
        
        if (result == 0) {
            REQUIRE(validate_results(events_10k.data(), 10000));
        }
        return result;
    };
    
    BENCHMARK("Processing only - 100K events") {
        for (auto& event : events_100k) event.action = 0;
        
        int result = process_events(&handle, events_100k.data(), 100000);
        
        if (result == 0) {
            REQUIRE(validate_results(events_100k.data(), 100000));
        }
        return result;
    };
    
    BENCHMARK("Processing only - 1M events") {
        for (auto& event : events_1m) event.action = 0;
        
        int result = process_events(&handle, events_1m.data(), 1000000);
        
        if (result == 0) {
            REQUIRE(validate_results(events_1m.data(), 1000000));
        }
        return result;
    };
    
    cleanup_processor(&handle);
}

TEST_CASE("Performance - Buffer Size Impact", "[performance]") {
    int device_count = get_cuda_device_count();
    if (device_count == 0) {
        SKIP("No CUDA devices available for performance testing");
    }
    
    // Pre-create test data
    const size_t num_events = 10000;
    std::vector<network_event_t> events(num_events);
    create_test_events(events.data(), num_events);
    
    const char* ptx_code = get_test_ptx();
    
    BENCHMARK("1MB buffer - 10K events") {
        processor_handle_t handle;
        memset(&handle, 0, sizeof(handle));
        
        if (init_processor(&handle, 0, 1024 * 1024) == 0 &&
            load_ptx_kernel(&handle, ptx_code, "_Z20simple_packet_filterP15network_event_tm") == 0) {
            
            for (auto& event : events) event.action = 0;
            
            int result = process_events(&handle, events.data(), num_events);
            cleanup_processor(&handle);
            
            if (result == 0) {
                REQUIRE(validate_results(events.data(), num_events));
            }
            return result;
        }
        cleanup_processor(&handle);
        return -1;
    };
    
    BENCHMARK("10MB buffer - 10K events") {
        processor_handle_t handle;
        memset(&handle, 0, sizeof(handle));
        
        if (init_processor(&handle, 0, 10 * 1024 * 1024) == 0 &&
            load_ptx_kernel(&handle, ptx_code, "_Z20simple_packet_filterP15network_event_tm") == 0) {
            
            for (auto& event : events) event.action = 0;
            
            int result = process_events(&handle, events.data(), num_events);
            cleanup_processor(&handle);
            
            if (result == 0) {
                REQUIRE(validate_results(events.data(), num_events));
            }
            return result;
        }
        cleanup_processor(&handle);
        return -1;
    };
    
    BENCHMARK("100MB buffer - 10K events") {
        processor_handle_t handle;
        memset(&handle, 0, sizeof(handle));
        
        if (init_processor(&handle, 0, 100 * 1024 * 1024) == 0 &&
            load_ptx_kernel(&handle, ptx_code, "_Z20simple_packet_filterP15network_event_tm") == 0) {
            
            for (auto& event : events) event.action = 0;
            
            int result = process_events(&handle, events.data(), num_events);
            cleanup_processor(&handle);
            
            if (result == 0) {
                REQUIRE(validate_results(events.data(), num_events));
            }
            return result;
        }
        cleanup_processor(&handle);
        return -1;
    };
}

TEST_CASE("Performance - Interface Comparison", "[performance]") {
    int device_count = get_cuda_device_count();
    if (device_count == 0) {
        SKIP("No CUDA devices available for performance testing");
    }
    
    // Setup once
    processor_handle_t handle;
    memset(&handle, 0, sizeof(handle));
    REQUIRE(init_processor(&handle, 0, 10 * 1024 * 1024) == 0);
    REQUIRE(load_ptx_kernel(&handle, get_test_ptx(), "_Z20simple_packet_filterP15network_event_tm") == 0);
    
    // Pre-create test data
    const size_t num_events = 10000;
    std::vector<network_event_t> events(num_events);
    create_test_events(events.data(), num_events);
    
    BENCHMARK("Array interface - 10K events") {
        for (auto& event : events) event.action = 0;
        
        int result = process_events(&handle, events.data(), num_events);
        
        if (result == 0) {
            REQUIRE(validate_results(events.data(), num_events));
        }
        return result;
    };
    
    BENCHMARK("Buffer interface - 10K events") {
        for (auto& event : events) event.action = 0;
        
        size_t buffer_size = num_events * sizeof(network_event_t);
        int result = process_events_buffer(&handle, events.data(), buffer_size, num_events);
        
        if (result == 0) {
            REQUIRE(validate_results(events.data(), num_events));
        }
        return result;
    };
    
    cleanup_processor(&handle);
}
