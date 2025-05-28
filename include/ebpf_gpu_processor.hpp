#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_device_manager.hpp"
#include "kernel_loader.hpp"
#include <memory>
#include <vector>
#include <functional>

namespace ebpf_gpu {

// Forward declarations
class EventProcessor;

enum class ProcessingResult {
    Success = 0,
    Error = -1,
    InvalidInput = -2,
    DeviceError = -3,
    KernelError = -4
};

class EventProcessor {
public:
    struct Config {
        int device_id;  // -1 for auto-select
        size_t buffer_size;  // 1MB default
        bool enable_profiling;
        bool use_unified_memory;
        
        // Kernel launch configuration
        int block_size;  // CUDA block size (threads per block)
        size_t shared_memory_size;  // Shared memory per block in bytes
        int max_grid_size;  // Maximum grid size (0 for unlimited)
        
        Config() : device_id(-1), 
                  buffer_size(1024 * 1024), 
                  enable_profiling(false), 
                  use_unified_memory(false),
                  block_size(256),  // Good default for most GPUs
                  shared_memory_size(0),  // No shared memory by default
                  max_grid_size(65535)  // CUDA maximum grid dimension
        {}
    };

    explicit EventProcessor(const Config& config = Config{});
    ~EventProcessor();

    // Non-copyable, movable
    EventProcessor(const EventProcessor&) = delete;
    EventProcessor& operator=(const EventProcessor&) = delete;
    EventProcessor(EventProcessor&&) noexcept;
    EventProcessor& operator=(EventProcessor&&) noexcept;

    // Kernel management
    ProcessingResult load_kernel_from_ptx(const std::string& ptx_code, const std::string& function_name);
    ProcessingResult load_kernel_from_file(const std::string& file_path, const std::string& function_name);
    ProcessingResult load_kernel_from_source(const std::string& cuda_source, const std::string& function_name,
                                const std::vector<std::string>& include_paths = {},
                                const std::vector<std::string>& compile_options = {});

    // Event processing - simplified interface
    // Single event processing
    ProcessingResult process_event(void* event_data, size_t event_size);
    
    // Multiple events processing (zero-copy)
    ProcessingResult process_events(void* events_buffer, size_t buffer_size, size_t event_count);

    // Utility functions for pinned memory management
    static void* allocate_pinned_buffer(size_t size);
    static void free_pinned_buffer(void* pinned_ptr);

    // Utility functions for registering/unregistering existing host memory
    
    /**
     * @brief Register existing host memory as pinned (page-locked) for faster GPU transfers
     * 
     * BACKGROUND:
     * Normal host memory (allocated with malloc, new, or std::vector) is "pageable", meaning
     * the OS can swap it to disk and change its physical address. When transferring pageable
     * memory to/from GPU, CUDA must:
     * 1. Allocate a temporary pinned buffer
     * 2. Copy: pageable → pinned (CPU memcpy)
     * 3. Transfer: pinned → GPU (fast DMA)
     * This results in double copying and reduced performance.
     * 
     * MECHANISM:
     * cudaHostRegister() makes existing memory "pinned" (page-locked):
     * - OS guarantees the memory won't be swapped to disk
     * - Physical address remains constant
     * - DMA hardware can access it directly
     * - Eliminates the need for intermediate staging buffers
     * 
     * PERFORMANCE BENEFITS:
     * - Up to 2-6x faster memory transfers (depends on system)
     * - Higher PCIe bandwidth utilization
     * - Reduced memory copy overhead
     * - Lower transfer latency
     * 
     * USAGE PATTERN:
     * ```cpp
     * std::vector<MyData> data(1000);
     * // ... fill data ...
     * 
     * // Pin the existing vector's memory
     * EventProcessor::register_host_buffer(data.data(), data.size() * sizeof(MyData));
     * 
     * // Now transfers to this buffer are fast
     * processor.process_events(data.data(), data.size() * sizeof(MyData), data.size());
     * 
     * // IMPORTANT: Unpin before vector is destroyed or reallocated
     * EventProcessor::unregister_host_buffer(data.data());
     * ```
     * 
     * IMPORTANT CONSIDERATIONS:
     * - Memory must remain at the same address while registered
     * - Don't resize/reallocate containers (std::vector::push_back, etc.) while registered
     * - Always call unregister_host_buffer() before freeing the memory
     * - Registration has overhead - beneficial for buffers used multiple times
     * - System has limited pinned memory - don't pin everything simultaneously
     * 
     * @param ptr Pointer to the start of the memory region to pin
     * @param size Size of the memory region in bytes
     * @param flags CUDA host registration flags:
     *              - 0 (default): cudaHostRegisterDefault - standard pinning
     *              - cudaHostRegisterPortable: usable across multiple CUDA contexts
     *              - cudaHostRegisterMapped: enables zero-copy GPU access on supported hardware
     *              - cudaHostRegisterIoMemory: for GPUDirect RDMA applications
     * 
     * @return ProcessingResult::Success on successful registration,
     *         ProcessingResult::InvalidInput for null pointer or zero size,
     *         ProcessingResult::DeviceError if CUDA registration fails
     * 
     * @note This function wraps cudaHostRegister(). See CUDA documentation for hardware
     *       requirements and limitations.
     */
    static ProcessingResult register_host_buffer(void* ptr, size_t size, unsigned int flags = 0 /* cudaHostRegisterDefault */);
    
    /**
     * @brief Unregister previously pinned host memory
     * 
     * Releases the pinned status of memory previously registered with register_host_buffer().
     * This MUST be called before the memory is freed or goes out of scope to avoid resource leaks.
     * 
     * After unregistering, the memory becomes regular pageable memory again, and transfers
     * to/from GPU will use the slower staged copy mechanism.
     * 
     * @param ptr Pointer to the memory region to unpin (same pointer used in register_host_buffer)
     * 
     * @return ProcessingResult::Success on successful unregistration,
     *         ProcessingResult::InvalidInput for null pointer,
     *         ProcessingResult::DeviceError if CUDA unregistration fails
     * 
     * @note This function wraps cudaHostUnregister()
     */
    static ProcessingResult unregister_host_buffer(void* ptr);

    // Device information
    GpuDeviceInfo get_device_info() const;
    size_t get_available_memory() const;
    bool is_ready() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Utility functions
std::vector<GpuDeviceInfo> get_available_devices();
int select_best_device(size_t min_memory = 0);
bool validate_ptx_code(const std::string& ptx_code);

} // namespace ebpf_gpu 