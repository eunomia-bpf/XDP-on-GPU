#include <catch2/catch_test_macros.hpp>
#include "cuda_event_processor.h"
#include <cstring>

TEST_CASE("C API - Basic Functionality", "[c_api]") {
    SECTION("Device count") {
        int device_count = get_cuda_device_count();
        REQUIRE(device_count >= 0);
        
        if (device_count == 0) {
            SKIP("No CUDA devices available for C API testing");
        }
    }
    
    SECTION("Processor lifecycle") {
        int device_count = get_cuda_device_count();
        if (device_count == 0) {
            SKIP("No CUDA devices available");
        }
        
        processor_handle_t handle;
        memset(&handle, 0, sizeof(handle));
        
        // Initialize processor
        int result = init_processor(&handle, 0, 1024 * 1024);
        REQUIRE(result == 0);
        
        // Cleanup processor
        result = cleanup_processor(&handle);
        REQUIRE(result == 0);
    }
}

TEST_CASE("C API - Error Handling", "[c_api]") {
    SECTION("Invalid parameters") {
        // Test with null handle
        int result = init_processor(nullptr, 0, 1024);
        REQUIRE(result != 0);
        
        const char* error = get_last_error();
        REQUIRE(error != nullptr);
        REQUIRE(strlen(error) > 0);
    }
    
    SECTION("Invalid device ID") {
        processor_handle_t handle;
        memset(&handle, 0, sizeof(handle));
        
        // Try to initialize with invalid device ID
        int result = init_processor(&handle, 999, 1024);
        REQUIRE(result != 0);
        
        const char* error = get_last_error();
        REQUIRE(error != nullptr);
    }
} 