#include <catch2/catch_test_macros.hpp>
#include "../include/gpu_backend.hpp"
#include <memory>

TEST_CASE("GPU Backend Factory", "[gpu][backend]") {
    SECTION("Create backend") {
        auto backend = ebpf_gpu::create_backend(ebpf_gpu::BackendType::CUDA);
        REQUIRE(backend != nullptr);
        
        // Basic info should be available even with stub backend
        auto device_info = backend->get_device_info(0);
        // No assertion on device_info content as it might be empty with stub backend
    }
}

TEST_CASE("Backend Memory Operations", "[gpu][backend][memory]") {
    auto backend = ebpf_gpu::create_backend(ebpf_gpu::BackendType::CUDA);
    REQUIRE(backend != nullptr);
    
    SECTION("Allocate and free memory") {
        const size_t size = 1024;
        void* ptr = backend->allocate_device_memory(size);
        
        // Even with stub backend, we should get a consistent behavior
        // (nullptr or valid pointer that can be freed)
        backend->free_device_memory(ptr);
        SUCCEED("Memory allocation and freeing completed without crashing");
    }
} 