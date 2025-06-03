#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

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

} // namespace ebpf_gpu

// Helper function to get PTX code - shared across all test files
inline const char* get_test_ptx() {
    static char* ptx_code = nullptr;
    if (ptx_code) return ptx_code;
    
#ifdef PTX_FILE_PATH
    const char* ptx_path = PTX_FILE_PATH;
#else
    const char* ptx_path = "test_ptx.ptx";
#endif
    
    // Read the test PTX file
    FILE* file = std::fopen(ptx_path, "r");
    if (!file) {
        // Try alternative paths
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
                ptx_path = path;
                break;
            }
        }
        
        if (!file) {
            // Embedded fallback test PTX if file not found
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
                
            size_t size = sizeof(embedded_ptx);
            ptx_code = (char*)std::malloc(size);
            std::memcpy(ptx_code, embedded_ptx, size);
            return ptx_code;
        }
    }
    
    std::fseek(file, 0, SEEK_END);
    long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    
    ptx_code = (char*)std::malloc(size + 1);
    if (size > 0) {
        size_t read = std::fread(ptx_code, 1, size, file);
        if (read < (size_t)size) {
            // Handle read error
            ptx_code[read] = '\0';
        } else {
            ptx_code[size] = '\0';
        }
    } else {
        ptx_code[0] = '\0';
    }
    
    std::fclose(file);
    
    return ptx_code;
} 