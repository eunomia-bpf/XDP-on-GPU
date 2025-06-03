#include <catch2/catch_test_macros.hpp>
#include "kernel_loader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace ebpf_gpu;

TEST_CASE("KernelLoader - Basic Functionality", "[kernel_loader]") {
    SECTION("PTX validation") {
        const char* valid_ptx = R"(
            .version 6.0
            .target sm_30
            .address_size 64
            
            .visible .entry kernel_function()
            {
                ret;
            }
        )";
        
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
        // Empty input should fail gracefully
        REQUIRE_THROWS_AS(loader.load_from_ir(""), std::runtime_error);
    }
    
    SECTION("Invalid file paths") {
        // Non-existent file should fail gracefully
        auto module = loader.load_from_file("/path/to/nonexistent/file.ptx");
        REQUIRE(module == nullptr);
    }
} 