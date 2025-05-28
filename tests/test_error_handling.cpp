#include <catch2/catch_test_macros.hpp>
#include "error_handling.hpp"

using namespace ebpf_gpu;

TEST_CASE("Error Handling - Exception Types", "[error_handling]") {
    SECTION("CudaException basic functionality") {
        CudaException ex("Test error message");
        REQUIRE(std::string(ex.what()) == "Test error message");
    }
    
    SECTION("Exception inheritance") {
        CudaException base_ex("Base error");
        REQUIRE_NOTHROW(throw base_ex);
        
        try {
            throw base_ex;
        } catch (const std::runtime_error& e) {
            REQUIRE(std::string(e.what()) == "Base error");
        }
    }
}

TEST_CASE("Error Handling - RAII Classes", "[error_handling]") {
    SECTION("DeviceMemory basic operations") {
        // This test requires CUDA to be available
        // We'll test the interface without actually allocating GPU memory
        REQUIRE(true); // Placeholder test
    }
} 