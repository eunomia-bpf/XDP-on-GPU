#include "../../include/gpu_device_manager.hpp"
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>

namespace ebpf_gpu {

namespace opencl {

#ifdef USE_OPENCL_BACKEND
#include <CL/cl.h>

// Helper function to get device platform and device IDs
bool get_platform_and_devices(std::vector<cl_platform_id>& platforms, 
                              std::vector<cl_device_id>& devices) {
    cl_int err;
    cl_uint num_platforms;
    
    // Get number of platforms
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "OpenCL: No platforms found" << std::endl;
        return false;
    }
    
    // Get platform IDs
    platforms.resize(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to get platform IDs: " << err << std::endl;
        return false;
    }
    
    // Debug: List all platforms
    for (cl_uint i = 0; i < num_platforms; i++) {
        char platform_name[256] = {0};
        char platform_vendor[256] = {0};
        
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, nullptr);
        
        std::cout << "Platform " << i << ": " << platform_name << " (" << platform_vendor << ")" << std::endl;
    }
    
    // Get all devices from all platforms
    devices.clear();
    
    for (cl_uint i = 0; i < num_platforms; i++) {
        // Try to get ALL device types (GPU, CPU, Accelerator, etc.)
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        
        if (err != CL_SUCCESS || num_devices == 0) {
            std::cerr << "OpenCL: No devices found on platform " << i << std::endl;
            continue;  // No devices on this platform
        }
        
        // Allocate space for device IDs
        size_t old_size = devices.size();
        devices.resize(old_size + num_devices);
        
        // Get ALL device types - this is more inclusive
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, 
                           devices.data() + old_size, nullptr);
        
        if (err != CL_SUCCESS) {
            std::cerr << "OpenCL: Failed to get device IDs: " << err << std::endl;
            devices.resize(old_size);  // Restore original size
            continue;
        }
        
        // Debug: List devices on this platform
        for (cl_uint j = 0; j < num_devices; j++) {
            cl_device_id device = devices[old_size + j];
            char device_name[256] = {0};
            cl_device_type device_type;
            
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);
            
            std::string type_str;
            if (device_type & CL_DEVICE_TYPE_CPU) type_str += "CPU ";
            if (device_type & CL_DEVICE_TYPE_GPU) type_str += "GPU ";
            if (device_type & CL_DEVICE_TYPE_ACCELERATOR) type_str += "Accelerator ";
            
            std::cout << "  Device " << j << ": " << device_name << " (" << type_str << ")" << std::endl;
        }
    }
    
    return !devices.empty();
}

int get_device_count() {
    std::vector<cl_platform_id> platforms;
    std::vector<cl_device_id> devices;
    
    if (!get_platform_and_devices(platforms, devices)) {
        return 0;
    }
    
    return static_cast<int>(devices.size());
}

size_t get_available_memory(int device_id) {
    std::vector<cl_platform_id> platforms;
    std::vector<cl_device_id> devices;
    
    if (!get_platform_and_devices(platforms, devices) || 
        device_id < 0 || device_id >= static_cast<int>(devices.size())) {
        return 0;
    }
    
    cl_device_id device = devices[device_id];
    cl_ulong global_mem_size = 0;
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 
                               sizeof(global_mem_size), &global_mem_size, nullptr);
    
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to get device memory info: " << err << std::endl;
        return 0;
    }
    
    // OpenCL doesn't provide a direct way to query available memory
    // Return a percentage of the total memory as a conservative estimate
    return global_mem_size * 3 / 4;  // Return 75% of total memory
}

GpuDeviceInfo query_device_info(int device_id) {
    GpuDeviceInfo info;
    std::vector<cl_platform_id> platforms;
    std::vector<cl_device_id> devices;
    
    if (!get_platform_and_devices(platforms, devices)) {
        std::cerr << "OpenCL: No devices found" << std::endl;
        throw std::runtime_error("No OpenCL devices found");
    }
    
    if (device_id < 0 || device_id >= static_cast<int>(devices.size())) {
        std::cerr << "OpenCL: Invalid device ID: " << device_id 
                 << " (found " << devices.size() << " devices)" << std::endl;
        throw std::runtime_error("Invalid OpenCL device ID");
    }
    
    cl_device_id device = devices[device_id];
    cl_int err;
    
    // Get device name using a more robust approach
    std::vector<char> device_name_buffer;
    size_t name_size = 0;
    
    // First get the required size
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &name_size);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to get device name size: " << err << std::endl;
        throw std::runtime_error("Failed to get OpenCL device name size");
    }
    
    // Allocate buffer with additional space for null terminator
    device_name_buffer.resize(name_size + 1, 0);
    
    // Now get the actual name
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, name_size, device_name_buffer.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to get device name: " << err << std::endl;
        throw std::runtime_error("Failed to get OpenCL device name");
    }
    
    // Ensure null termination
    device_name_buffer[name_size] = '\0';
    
    // Get device vendor using the same robust approach
    std::vector<char> device_vendor_buffer;
    size_t vendor_size = 0;
    
    // First get the required size
    err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, nullptr, &vendor_size);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to get device vendor size: " << err << std::endl;
        // Not critical, use empty string
        vendor_size = 0;
    }
    
    std::string vendor_str;
    if (vendor_size > 0) {
        // Allocate buffer with additional space for null terminator
        device_vendor_buffer.resize(vendor_size + 1, 0);
        
        // Now get the actual vendor
        err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, vendor_size, device_vendor_buffer.data(), nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "OpenCL: Failed to get device vendor: " << err << std::endl;
            // Not critical, use empty string
        } else {
            // Ensure null termination
            device_vendor_buffer[vendor_size] = '\0';
            vendor_str = device_vendor_buffer.data();
        }
    }
    
    // Get compute units
    cl_uint compute_units = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, 
                         sizeof(compute_units), &compute_units, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to get compute units: " << err << std::endl;
        compute_units = 0;
    }
    
    // Get clock frequency
    cl_uint clock_frequency = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, 
                         sizeof(clock_frequency), &clock_frequency, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to get clock frequency: " << err << std::endl;
        clock_frequency = 0;
    }
    
    // Get global memory
    cl_ulong global_memory = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 
                         sizeof(global_memory), &global_memory, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to get global memory: " << err << std::endl;
        global_memory = 0;
    }
    
    // Get device type
    cl_device_type device_type = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_TYPE, 
                         sizeof(device_type), &device_type, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL: Failed to get device type: " << err << std::endl;
        device_type = CL_DEVICE_TYPE_DEFAULT;
    }
    
    // Populate the device info structure
    info.backend_type = BackendType::OpenCL;
    info.device_id = device_id;
    
    // Create a more descriptive name that includes vendor and device type
    std::string name_str = device_name_buffer.data();
    if (!vendor_str.empty()) {
        name_str = vendor_str + " " + name_str;
    }
    
    // Add device type suffix if it's not a GPU
    if (device_type & CL_DEVICE_TYPE_CPU) {
        name_str += " (CPU)";
    } else if (device_type & CL_DEVICE_TYPE_ACCELERATOR) {
        name_str += " (Accelerator)";
    } else if (!(device_type & CL_DEVICE_TYPE_GPU)) {
        name_str += " (Unknown)";
    }
    
    info.name = name_str;
    info.compute_capability_major = 0;  // Not applicable for OpenCL
    info.compute_capability_minor = 0;  // Not applicable for OpenCL
    info.clock_rate = clock_frequency * 1000;  // Convert MHz to kHz for consistency
    info.num_cores = compute_units;
    info.total_memory = global_memory;
    info.available_memory = global_memory * 3 / 4;  // Conservative estimate
    
    return info;
}

#else // Stub implementations when OpenCL is not available

// Use weak attribute to allow linking even when OpenCL is disabled
__attribute__((weak)) int get_device_count() {
    return 0;
}

__attribute__((weak)) size_t get_available_memory(int device_id) {
    return 0;
}

__attribute__((weak)) GpuDeviceInfo query_device_info(int device_id) {
    GpuDeviceInfo info;
    info.device_id = -1;
    info.backend_type = BackendType::OpenCL;
    info.name = "OpenCL support not compiled in";
    return info;
}

#endif // USE_OPENCL_BACKEND

} // namespace opencl

} // namespace ebpf_gpu 