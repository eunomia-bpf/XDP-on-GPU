#include "cuda_event_processor.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

// Global error message buffer
static char g_error_message[1024] = {0};

// Helper function to set error message
static void set_error(const char *msg) {
    strncpy(g_error_message, msg, sizeof(g_error_message) - 1);
    g_error_message[sizeof(g_error_message) - 1] = '\0';
}

// Helper function to check CUDA runtime errors
static int check_cuda_runtime_error(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        snprintf(g_error_message, sizeof(g_error_message), 
                "%s failed: %s", operation, cudaGetErrorString(result));
        return -1;
    }
    return 0;
}

// Helper function to check CUDA driver errors
static int check_cuda_error(CUresult result, const char *operation) {
    if (result != CUDA_SUCCESS) {
        const char *error_name;
        const char *error_string;
        cuGetErrorName(result, &error_name);
        cuGetErrorString(result, &error_string);
        snprintf(g_error_message, sizeof(g_error_message), 
                "%s failed: %s (%s)", operation, error_name, error_string);
        return -1;
    }
    return 0;
}

int get_cuda_device_count() {
    int device_count = 0;
    cudaError_t result = cudaGetDeviceCount(&device_count);
    if (result != cudaSuccess) {
        set_error("Failed to get CUDA device count");
        return -1;
    }
    return device_count;
}

const char* get_last_error() {
    return g_error_message;
}

int init_processor(processor_handle_t *handle, int device_id, size_t buffer_size) {
    if (!handle) {
        set_error("Invalid handle");
        return -1;
    }

    // Initialize CUDA runtime first
    cudaError_t cuda_result = cudaSetDevice(device_id);
    if (check_cuda_runtime_error(cuda_result, "cudaSetDevice")) {
        return -1;
    }

    // Initialize CUDA driver API
    CUresult result = cuInit(0);
    if (check_cuda_error(result, "cuInit")) {
        return -1;
    }

    // Get device
    CUdevice device;
    result = cuDeviceGet(&device, device_id);
    if (check_cuda_error(result, "cuDeviceGet")) {
        return -1;
    }

    // Create context
    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (check_cuda_error(result, "cuCtxCreate")) {
        return -1;
    }

    // Allocate device buffer using runtime API (simpler)
    void *device_buffer;
    cuda_result = cudaMalloc(&device_buffer, buffer_size);
    if (check_cuda_runtime_error(cuda_result, "cudaMalloc")) {
        cuCtxDestroy(context);
        return -1;
    }

    // Initialize handle
    memset(handle, 0, sizeof(processor_handle_t));
    handle->cuda_context = (void*)context;
    handle->device_buffer = device_buffer;
    handle->buffer_size = buffer_size;
    handle->device_id = device_id;

    return 0;
}

int cleanup_processor(processor_handle_t *handle) {
    if (!handle) {
        return -1;
    }

    // Free device buffer
    if (handle->device_buffer) {
        cudaFree(handle->device_buffer);
    }

    // Unload module
    if (handle->cuda_module) {
        cuModuleUnload((CUmodule)handle->cuda_module);
    }

    // Destroy context
    if (handle->cuda_context) {
        cuCtxDestroy((CUcontext)handle->cuda_context);
    }

    memset(handle, 0, sizeof(processor_handle_t));
    return 0;
}

int load_ptx_kernel(processor_handle_t *handle, const char *ptx_code, const char *function_name) {
    if (!handle || !ptx_code || !function_name) {
        set_error("Invalid parameters");
        return -1;
    }

    // Set context
    CUresult result = cuCtxSetCurrent((CUcontext)handle->cuda_context);
    if (check_cuda_error(result, "cuCtxSetCurrent")) {
        return -1;
    }

    // Load module from PTX
    CUmodule module;
    result = cuModuleLoadData(&module, ptx_code);
    if (check_cuda_error(result, "cuModuleLoadData")) {
        return -1;
    }

    // Get function
    CUfunction function;
    result = cuModuleGetFunction(&function, module, function_name);
    if (check_cuda_error(result, "cuModuleGetFunction")) {
        cuModuleUnload(module);
        return -1;
    }

    // Clean up old module if exists
    if (handle->cuda_module) {
        cuModuleUnload((CUmodule)handle->cuda_module);
    }

    handle->cuda_module = (void*)module;
    handle->cuda_function = (void*)function;

    return 0;
}

int load_kernel_function(processor_handle_t *handle, const char *kernel_file, const char *function_name) {
    if (!handle || !kernel_file || !function_name) {
        set_error("Invalid parameters");
        return -1;
    }

    // Read PTX file
    FILE *file = fopen(kernel_file, "rb");
    if (!file) {
        set_error("Cannot open kernel file");
        return -1;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *ptx_code = (char*)malloc(file_size + 1);
    if (!ptx_code) {
        fclose(file);
        set_error("Memory allocation failed");
        return -1;
    }

    fread(ptx_code, 1, file_size, file);
    ptx_code[file_size] = '\0';
    fclose(file);

    int result = load_ptx_kernel(handle, ptx_code, function_name);
    free(ptx_code);

    return result;
}

int process_events(processor_handle_t *handle, network_event_t *events, size_t num_events) {
    if (!handle || !events || num_events == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    if (!handle->cuda_function) {
        set_error("No kernel loaded");
        return -1;
    }

    // Set context
    CUresult result = cuCtxSetCurrent((CUcontext)handle->cuda_context);
    if (check_cuda_error(result, "cuCtxSetCurrent")) {
        return -1;
    }

    // Calculate required buffer size
    size_t required_size = num_events * sizeof(network_event_t);
    if (required_size > handle->buffer_size) {
        set_error("Buffer too small for events");
        return -1;
    }

    // Copy events to device using runtime API
    cudaError_t cuda_result = cudaMemcpy(handle->device_buffer, events, required_size, cudaMemcpyHostToDevice);
    if (check_cuda_runtime_error(cuda_result, "cudaMemcpy H2D")) {
        return -1;
    }

    // Set up kernel parameters
    void *args[] = {
        &handle->device_buffer,
        &num_events
    };

    // Launch kernel
    result = cuLaunchKernel(
        (CUfunction)handle->cuda_function,
        (num_events + 255) / 256, 1, 1,  // Grid dimensions
        256, 1, 1,                       // Block dimensions
        0,                               // Shared memory
        0,                               // Stream
        args,                            // Parameters
        NULL                             // Extra
    );
    if (check_cuda_error(result, "cuLaunchKernel")) {
        return -1;
    }

    // Wait for completion
    cuda_result = cudaDeviceSynchronize();
    if (check_cuda_runtime_error(cuda_result, "cudaDeviceSynchronize")) {
        return -1;
    }

    // Copy results back
    cuda_result = cudaMemcpy(events, handle->device_buffer, required_size, cudaMemcpyDeviceToHost);
    if (check_cuda_runtime_error(cuda_result, "cudaMemcpy D2H")) {
        return -1;
    }

    return 0;
}

int process_events_buffer(processor_handle_t *handle, void *buffer, size_t buffer_size, size_t num_events) {
    if (!handle || !buffer || buffer_size == 0 || num_events == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    if (!handle->cuda_function) {
        set_error("No kernel loaded");
        return -1;
    }

    // Set context
    CUresult result = cuCtxSetCurrent((CUcontext)handle->cuda_context);
    if (check_cuda_error(result, "cuCtxSetCurrent")) {
        return -1;
    }

    // Check buffer size
    if (buffer_size > handle->buffer_size) {
        set_error("Buffer too large");
        return -1;
    }

    // Copy buffer to device
    cudaError_t cuda_result = cudaMemcpy(handle->device_buffer, buffer, buffer_size, cudaMemcpyHostToDevice);
    if (check_cuda_runtime_error(cuda_result, "cudaMemcpy H2D")) {
        return -1;
    }

    // Set up kernel parameters
    void *args[] = {
        &handle->device_buffer,
        &num_events
    };

    // Launch kernel
    result = cuLaunchKernel(
        (CUfunction)handle->cuda_function,
        (num_events + 255) / 256, 1, 1,  // Grid dimensions
        256, 1, 1,                       // Block dimensions
        0,                               // Shared memory
        0,                               // Stream
        args,                            // Parameters
        NULL                             // Extra
    );
    if (check_cuda_error(result, "cuLaunchKernel")) {
        return -1;
    }

    // Wait for completion
    cuda_result = cudaDeviceSynchronize();
    if (check_cuda_runtime_error(cuda_result, "cudaDeviceSynchronize")) {
        return -1;
    }

    // Copy results back
    cuda_result = cudaMemcpy(buffer, handle->device_buffer, buffer_size, cudaMemcpyDeviceToHost);
    if (check_cuda_runtime_error(cuda_result, "cudaMemcpy D2H")) {
        return -1;
    }

    return 0;
} 