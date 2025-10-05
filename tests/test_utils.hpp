#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>
#include <iostream>

#ifdef USE_CUDA_BACKEND
#include "test_config.h"
#endif

// Put NetworkEvent in the ebpf_gpu namespace to match the expected kernel signature
namespace ebpf_gpu {

// NetworkEvent definition for testing purposes
struct NetworkEvent {
    uint8_t* data = nullptr;
    uint32_t length = 0;
    uint64_t timestamp = 0;
    uint32_t src_ip = 0;
    uint32_t dst_ip = 0;
    uint16_t src_port = 0;
    uint16_t dst_port = 0;
    uint8_t protocol = 0;
    uint8_t action = 0;  // 0=drop, 1=pass, 2=redirect
};

// Static assert to verify the size of NetworkEvent
static_assert(sizeof(NetworkEvent) == 34 || sizeof(NetworkEvent) == 40, 
    "NetworkEvent size is unexpected - check memory layout for padding");

// Backend detection
enum class TestBackend {
    CUDA,
    OpenCL,
    Unknown
};

inline TestBackend detect_test_backend() {
#ifdef USE_CUDA_BACKEND
    return TestBackend::CUDA;
#elif defined(USE_OPENCL_BACKEND)
    return TestBackend::OpenCL;
#else
    return TestBackend::Unknown;
#endif
}

} // namespace ebpf_gpu

// Helper function to get PTX/OpenCL code - shared across all test files
inline const char* get_test_ptx() {
    static char* code = nullptr;
    if (code) return code;
    
    // Detect backend
    ebpf_gpu::TestBackend backend = ebpf_gpu::detect_test_backend();
    
    // Try to load appropriate test code file
    const char* file_path = nullptr;
    
    if (backend == ebpf_gpu::TestBackend::CUDA) {
#ifdef PTX_FILE_PATH
        file_path = PTX_FILE_PATH;
#else
        file_path = "test_ptx.ptx";
#endif
    } else if (backend == ebpf_gpu::TestBackend::OpenCL) {
        // Try to find OpenCL test kernel
        const char* opencl_paths[] = {
            "test_opencl.cl",
            "tests/test_opencl.cl",
            "../tests/test_opencl.cl",
            "./tests/test_opencl.cl",
            "./build/tests/test_opencl.cl",
            "../build/tests/test_opencl.cl",
            "./build/test_opencl.cl"
        };
        
        for (const char* path : opencl_paths) {
            if (std::ifstream(path).good()) {
                file_path = path;
                break;
            }
        }
        
        // If no OpenCL file found, we'll use embedded fallback
    }
    
    // Try to read the file if found
    FILE* file = nullptr;
    if (file_path) {
        file = std::fopen(file_path, "r");
    }
    
    if (!file && backend == ebpf_gpu::TestBackend::CUDA) {
        // Try alternative PTX paths for CUDA
        const char* ptx_paths[] = {
            "test_ptx.ptx",
            "tests/test_ptx.ptx",
            "../tests/test_ptx.ptx",
            "./tests/test_ptx.ptx",
            "./build/tests/test_ptx.ptx",
            "../build/tests/test_ptx.ptx",
            "./build/test_ptx.ptx"
        };
        
        for (const char* path : ptx_paths) {
            file = std::fopen(path, "r");
            if (file) {
                file_path = path;
                break;
            }
        }
    }
    
    if (file) {
        // Read from file
        std::fseek(file, 0, SEEK_END);
        long size = std::ftell(file);
        std::fseek(file, 0, SEEK_SET);
        
        code = (char*)std::malloc(size + 1);
        if (size > 0) {
            size_t read = std::fread(code, 1, size, file);
            if (read < (size_t)size) {
                // Handle read error
                code[read] = '\0';
            } else {
                code[size] = '\0';
            }
        } else {
            code[0] = '\0';
        }
        
        std::fclose(file);
    } else {
        // Use embedded fallback based on backend
        const char* embedded_code;
        size_t code_size;
        
        if (backend == ebpf_gpu::TestBackend::OpenCL) {
            // Embedded OpenCL kernel
            static const char embedded_opencl[] = 
                "__kernel void simple_kernel(\n"
                "    __global const unsigned char* input_ptr,\n"
                "    __global unsigned int* output_ptr,\n"
                "    const unsigned int length\n"
                ")\n"
                "{\n"
                "    // Get thread ID\n"
                "    int id = get_global_id(0);\n"
                "    \n"
                "    // Set output to 1 (simple test pass condition)\n"
                "    if (id == 0) {\n"
                "        output_ptr[0] = 1;\n"
                "    }\n"
                "}\n";
            
            embedded_code = embedded_opencl;
            code_size = sizeof(embedded_opencl);
        } else {
            // Embedded PTX (for CUDA)
            static const char embedded_ptx[] = 
                ".version 6.0\n"
                ".target sm_30\n"
                ".address_size 64\n"
                "\n"
                ".visible .entry simple_kernel(\n"
                "    .param .u64 input_ptr,\n"
                "    .param .u64 output_ptr,\n"
                "    .param .u32 length\n"
                ")\n"
                "{\n"
                "    .reg .u64 %rd<5>;\n"
                "    .reg .u32 %r<5>;\n"
                "    \n"
                "    ld.param.u64 %rd1, [input_ptr];\n"
                "    ld.param.u64 %rd2, [output_ptr];\n"
                "    ld.param.u32 %r1, [length];\n"
                "    \n"
                "    mov.u32 %r5, 1;\n"
                "    st.global.u32 [%rd2], %r5;\n"
                "    \n"
                "    ret;\n"
                "}\n";
            
            embedded_code = embedded_ptx;
            code_size = sizeof(embedded_ptx);
        }
        
        code = (char*)std::malloc(code_size);
        std::memcpy(code, embedded_code, code_size);
    }
    
    return code;
} 