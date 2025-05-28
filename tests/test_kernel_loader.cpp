#include <catch2/catch_test_macros.hpp>
#include "kernel_loader.hpp"
#include "error_handling.hpp"
#include <fstream>
#include <sstream>

using namespace ebpf_gpu;

TEST_CASE("KernelLoader - Basic Functionality", "[kernel_loader]") {
    KernelLoader loader;
    
    SECTION("PTX validation") {
        // Valid PTX code
        std::string valid_ptx = R"(
.version 7.0
.target sm_75
.address_size 64

.visible .entry test_kernel(
    .param .u64 test_kernel_param_0
)
{
    ret;
}
)";
        
        REQUIRE(KernelLoader::validate_ptx(valid_ptx));
        
        // Invalid PTX code
        std::string invalid_ptx = "not ptx code";
        REQUIRE_FALSE(KernelLoader::validate_ptx(invalid_ptx));
        
        // Empty PTX code
        REQUIRE_FALSE(KernelLoader::validate_ptx(""));
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
    
    SECTION("Empty PTX code") {
        REQUIRE_THROWS_AS(loader.load_from_ptx(""), std::invalid_argument);
    }
    
    SECTION("Non-existent file") {
        REQUIRE_THROWS_AS(loader.load_from_file("/non/existent/file.ptx"), std::runtime_error);
    }
} 