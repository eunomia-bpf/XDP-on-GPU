#include <catch2/catch_test_macros.hpp>
#include "../include/kernel_loader.hpp"
#include "test_utils.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace ebpf_gpu;

TEST_CASE("KernelLoader - Basic Functionality", "[kernel_loader]") {
    SECTION("PTX validation") {
        // Get the test PTX
        const char* valid_ptx = get_test_ptx();
        REQUIRE(valid_ptx != nullptr);
        
        const char* invalid_ptx = "This is not valid PTX code";
        
        // Test validation
        REQUIRE(KernelLoader::validate_ir(valid_ptx));
        REQUIRE_FALSE(KernelLoader::validate_ir(invalid_ptx));
        REQUIRE_FALSE(KernelLoader::validate_ir(""));
    }
    
    SECTION("Backend detection") {
        KernelLoader loader;
        BackendType backend = loader.get_backend();
        
        // Just verify that the backend is one of the known types
        REQUIRE((backend == BackendType::CUDA || 
                 backend == BackendType::OpenCL || 
                 backend == BackendType::Unknown));
        
        // Output the backend type for informational purposes
        INFO("Detected backend: " << 
            (backend == BackendType::CUDA ? "CUDA" : 
             backend == BackendType::OpenCL ? "OpenCL" : "Unknown"));
    }
    
    SECTION("File reading") {
        // Create a temporary test file
        std::string test_file = "/tmp/test_ptx.txt";
        std::string test_content = "test content for file reading";
        
        std::ofstream file(test_file);
        file << test_content;
        file.close();
        
        auto content = KernelLoader::read_file(test_file);
        std::string read_content(content.begin(), content.end());
        
        REQUIRE(read_content.find(test_content) != std::string::npos);
        
        // Cleanup
        std::remove(test_file.c_str());
        
        // Test non-existent file
        REQUIRE_THROWS_AS(KernelLoader::read_file("/non/existent/file.txt"), std::runtime_error);
    }
}

TEST_CASE("KernelLoader - Error Handling", "[kernel_loader]") {
    KernelLoader loader;
    
    SECTION("Invalid input") {
        // Check backend type first
        BackendType backend = loader.get_backend();
        if (backend == BackendType::Unknown) {
            SKIP("Skipping test because no backend is available");
            return;
        }
        
        // Empty input should fail gracefully
        try {
            REQUIRE_THROWS_AS(loader.load_from_ir(""), std::runtime_error);
        } catch (...) {
            // If test fails, we're possibly catching the exception thrown by the load_from_ir method
            // instead of catching a test failure
            FAIL("load_from_ir with empty string should throw std::runtime_error");
        }
    }
    
    SECTION("Invalid file paths") {
        // Non-existent file should fail gracefully
        auto module = loader.load_from_file("/path/to/nonexistent/file.ptx");
        REQUIRE(module == nullptr);
    }
} 