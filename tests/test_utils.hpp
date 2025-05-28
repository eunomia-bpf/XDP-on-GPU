#pragma once

#include <cstdio>
#include <cstdlib>

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