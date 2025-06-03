#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

// Put NetworkEvent in the ebpf_gpu namespace to match the expected kernel signature
namespace ebpf_gpu {

// NetworkEvent definition for testing purposes
struct NetworkEvent {
    uint64_t timestamp = 0;
    uint32_t src_ip = 0;
    uint32_t dst_ip = 0;
    uint16_t src_port = 0;
    uint16_t dst_port = 0;
    uint8_t protocol = 0;
    uint8_t action = 0;  // 0=drop, 1=pass, 2=redirect
};

} // namespace ebpf_gpu

// Helper function to get PTX code - shared across all test files
inline const char* get_test_ptx() {
    static char* ptx_code = nullptr;
    if (ptx_code) return ptx_code;
    
#ifdef PTX_FILE_PATH
    const char* ptx_path = PTX_FILE_PATH;
#else
    const char* ptx_path = "cuda_kernels.ptx";
#endif
    
    // Read the pre-generated PTX file
    FILE* file = std::fopen(ptx_path, "r");
    if (!file) {
        // Try alternative paths
        const char* ptx_paths[] = {
            "build/tests/ptx/cuda_kernels.ptx",
            "../build/tests/ptx/cuda_kernels.ptx",
            "./build/tests/ptx/cuda_kernels.ptx",
            "tests/ptx/cuda_kernels.ptx",
            "./ptx/cuda_kernels.ptx",
            "../tests/ptx/cuda_kernels.ptx",
            "../ptx/cuda_kernels.ptx",
            "ptx/cuda_kernels.ptx"
        };
        
        for (const char* path : ptx_paths) {
            file = std::fopen(path, "r");
            if (file) {
                ptx_path = path;
                break;
            }
        }
        
        if (!file) {
            return nullptr;
        }
    }
    
    std::fseek(file, 0, SEEK_END);
    long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    
    ptx_code = (char*)std::malloc(size + 1);
    std::fread(ptx_code, 1, size, file);
    ptx_code[size] = '\0';
    std::fclose(file);
    
    return ptx_code;
} 