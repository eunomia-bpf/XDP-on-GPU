#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <csignal>
#include <ctime>
#include <sys/time.h>
#include <unistd.h>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>

extern "C" {
#include "dpdk_driver.h"
}

#include "../include/ebpf_gpu_processor.hpp"

/* Configuration */
#define METRICS_INTERVAL 1
#define MAX_PACKETS_PER_POLL 1024
#define MAX_PORTS 32
#define MAX_PATH_LEN 256

/* Simple metrics structure */
struct port_metrics {
    uint64_t rx_packets;
    uint64_t rx_bytes;
    uint64_t processed_packets;  /* For GPU processing */
};

/* Application configuration */
struct app_config {
    bool use_gpu;
    char kernel_path[MAX_PATH_LEN];
    char function_name[MAX_PATH_LEN];
    int device_id;
    int batch_size;
};

/* Global metrics */
struct {
    uint64_t start_time_sec;
    uint64_t total_rx_packets;
    uint64_t total_rx_bytes;
    uint64_t total_processed_packets;  /* For GPU processing */
    struct port_metrics ports[MAX_PORTS];
} g_metrics = {0};

/* Global variables */
static volatile bool force_quit = false;
static struct app_config g_config = {0};
static std::unique_ptr<ebpf_gpu::EventProcessor> g_processor;

/* Get current timestamp */
static uint64_t get_timestamp_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec;
}

/* Initialize metrics */
static void init_metrics(void)
{
    memset(&g_metrics, 0, sizeof(g_metrics));
    g_metrics.start_time_sec = get_timestamp_sec();
    std::cout << "Metrics collection initialized" << std::endl;
}

/* Ultra-fast port metrics update - minimal code */
static inline void update_metrics(uint16_t port, uint32_t nb_rx, uint64_t rx_bytes, uint32_t processed)
{
    if (port < MAX_PORTS) {
        g_metrics.ports[port].rx_packets += nb_rx;
        g_metrics.ports[port].rx_bytes += rx_bytes;
        g_metrics.ports[port].processed_packets += processed;
        g_metrics.total_rx_packets += nb_rx;
        g_metrics.total_rx_bytes += rx_bytes;
        g_metrics.total_processed_packets += processed;
    }
}

/* Print final metrics */
static void print_metrics(void)
{
    uint64_t current_time = get_timestamp_sec();
    uint64_t runtime_sec = current_time - g_metrics.start_time_sec;
    uint16_t num_ports = dpdk_get_port_count();
    
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "DPDK PACKET PROCESSING METRICS WITH GPU ACCELERATION" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "Runtime: " << runtime_sec << " seconds" << std::endl;
    std::cout << "Total RX Packets: " << g_metrics.total_rx_packets << std::endl;
    std::cout << "Total RX Bytes: " << g_metrics.total_rx_bytes 
              << " (" << (g_metrics.total_rx_bytes / (1024.0 * 1024.0)) << " MB)" << std::endl;
    
    if (g_config.use_gpu) {
        std::cout << "Total Processed Packets (GPU): " << g_metrics.total_processed_packets << std::endl;
    }
    
    if (runtime_sec > 0) {
        std::cout << "Average RX Rate: " << ((double)g_metrics.total_rx_packets / runtime_sec) << " pps, "
                  << ((double)g_metrics.total_rx_bytes * 8 / runtime_sec / (1024*1024)) << " Mbps" << std::endl;
        
        if (g_config.use_gpu) {
            std::cout << "Average GPU Processing Rate: " << ((double)g_metrics.total_processed_packets / runtime_sec) << " pps" << std::endl;
        }
    }
    
    std::cout << "\nPER-PORT STATISTICS:" << std::endl;
    if (g_config.use_gpu) {
        std::cout << std::left << std::setw(5) << "Port" << std::setw(12) << "RX Packets" 
                 << std::setw(12) << "RX Bytes" << std::setw(12) << "Processed" << std::endl;
    } else {
        std::cout << std::left << std::setw(5) << "Port" << std::setw(12) << "RX Packets" 
                 << std::setw(12) << "RX Bytes" << std::endl;
    }
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    
    for (uint16_t port = 0; port < num_ports && port < MAX_PORTS; port++) {
        if (g_config.use_gpu) {
            std::cout << std::left << std::setw(5) << port << std::setw(12) << g_metrics.ports[port].rx_packets
                     << std::setw(12) << g_metrics.ports[port].rx_bytes
                     << std::setw(12) << g_metrics.ports[port].processed_packets << std::endl;
        } else {
            std::cout << std::left << std::setw(5) << port << std::setw(12) << g_metrics.ports[port].rx_packets
                     << std::setw(12) << g_metrics.ports[port].rx_bytes << std::endl;
        }
    }
    std::cout << "================================================================================" << std::endl;
}

/* Signal handler */
static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM) {
        std::cout << "\nSignal " << signum << " received, preparing to exit..." << std::endl;
        force_quit = true;
    }
}

/* Initialize the GPU processor */
static bool init_gpu_processor()
{
    if (!g_config.use_gpu || g_config.kernel_path[0] == '\0' || g_config.function_name[0] == '\0') {
        return false;
    }
    
    try {
        // Create GPU processor configuration
        ebpf_gpu::EventProcessor::Config config;
        config.device_id = g_config.device_id;
        config.max_batch_size = g_config.batch_size;
        config.use_zero_copy = true;  // Enable zero-copy for better performance
        config.enable_profiling = false;
        
        // Create processor instance
        g_processor = std::make_unique<ebpf_gpu::EventProcessor>(config);
        
        // Print GPU device info
        auto device_info = g_processor->get_device_info();
        std::cout << "Using GPU: " << device_info.name << " (Device " << device_info.device_id << ")" << std::endl;
        std::cout << "GPU Memory: " << (device_info.total_memory / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "CUDA Capability: " << device_info.compute_capability_major << "."
                 << device_info.compute_capability_minor << std::endl;
        
        // Load the kernel
        ebpf_gpu::ProcessingResult result;
        std::string kernel_path = g_config.kernel_path;
        std::string function_name = g_config.function_name;
        
        // Check file extension to determine loading method
        if (kernel_path.size() > 4 && kernel_path.substr(kernel_path.size()-4) == ".ptx") {
            // Load PTX file
            std::cout << "Loading PTX kernel from: " << kernel_path << std::endl;
            result = g_processor->load_kernel_from_file(kernel_path, function_name);
        } else {
            // Load CUDA source
            std::cout << "Loading CUDA source from: " << kernel_path << std::endl;
            result = g_processor->load_kernel_from_source(kernel_path, function_name);
        }
        
        if (result != ebpf_gpu::ProcessingResult::Success) {
            std::cerr << "Failed to load kernel: " << static_cast<int>(result) << std::endl;
            return false;
        }
        
        std::cout << "GPU processor initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing GPU processor: " << e.what() << std::endl;
        return false;
    }
}

/* The main processing loop */
static void main_loop(void)
{
    std::cout << "\nProcessing packets. [Ctrl+C to quit]" << std::endl;
    
    /* Initialize metrics */
    init_metrics();
    
    /* Pre-allocate packet array */
    dpdk_packet_t packets[MAX_PACKETS_PER_POLL];
    
    /* Track if we've shown packet samples */
    uint64_t display_count = 0;
    
    /* Buffers for GPU processing */
    void* gpu_buffer = nullptr;
    std::vector<uint32_t> packet_sizes;
    std::vector<uint32_t> results;
    const size_t max_buffer_size = MAX_PACKETS_PER_POLL * 2048;  // Assume 2KB max packet size
    
    /* Allocate GPU processing buffers if using GPU */
    if (g_config.use_gpu && g_processor) {
        // Allocate pinned host memory for better PCIe transfer performance
        gpu_buffer = ebpf_gpu::EventProcessor::allocate_pinned_buffer(max_buffer_size);
        if (!gpu_buffer) {
            std::cerr << "Failed to allocate pinned buffer for GPU processing" << std::endl;
            return;
        }
        
        // Allocate vectors for packet sizes and results
        packet_sizes.resize(MAX_PACKETS_PER_POLL);
        results.resize(MAX_PACKETS_PER_POLL);
    }
    
    /* Run until the application is quit or killed */
    while (!force_quit) {
        /* Poll for packets */
        uint64_t bytes_received = 0;
        int nb_rx = dpdk_poll(packets, MAX_PACKETS_PER_POLL, &bytes_received);
        
        if (nb_rx < 0) {
            std::cerr << "Error polling for packets: " << nb_rx << std::endl;
            break;
        }
        
        if (nb_rx > 0) {
            /* Display first few packets */
            for (int i = 0; i < nb_rx && display_count < 10; i++, display_count++) {
                std::cout << "Packet on port " << packets[i].port 
                          << ": length = " << packets[i].length << " bytes" << std::endl;
            }
            
            /* Process packets on GPU if enabled */
            uint32_t processed_count = 0;
            if (g_config.use_gpu && g_processor && gpu_buffer) {
                // Prepare data for GPU processing
                size_t total_size = 0;
                for (int i = 0; i < nb_rx; i++) {
                    // Ensure we don't exceed buffer capacity
                    if (total_size + packets[i].length <= max_buffer_size) {
                        // Copy packet data to the GPU buffer
                        memcpy(static_cast<char*>(gpu_buffer) + total_size, 
                               packets[i].data, 
                               packets[i].length);
                        
                        // Store packet size
                        packet_sizes[i] = packets[i].length;
                        total_size += packets[i].length;
                    }
                }
                
                // Process the packets on GPU
                void* size_buffer = packet_sizes.data();
                void* result_buffer = results.data();
                
                // Register host buffers for zero-copy
                ebpf_gpu::EventProcessor::register_host_buffer(size_buffer, packet_sizes.size() * sizeof(uint32_t));
                ebpf_gpu::EventProcessor::register_host_buffer(result_buffer, results.size() * sizeof(uint32_t));
                
                // Call the GPU to process the packets
                auto result = g_processor->process_events(gpu_buffer, total_size, nb_rx);
                
                // Unregister host buffers
                ebpf_gpu::EventProcessor::unregister_host_buffer(size_buffer);
                ebpf_gpu::EventProcessor::unregister_host_buffer(result_buffer);
                
                if (result == ebpf_gpu::ProcessingResult::Success) {
                    // Count processed packets by checking the result array
                    for (int i = 0; i < nb_rx; i++) {
                        if (results[i] != 0) {
                            processed_count++;
                        }
                    }
                } else {
                    std::cerr << "GPU processing failed with code: " << static_cast<int>(result) << std::endl;
                }
            }
            
            /* Count packets by port */
            uint16_t packets_by_port[MAX_PORTS] = {0};
            uint64_t bytes_by_port[MAX_PORTS] = {0};
            uint32_t processed_by_port[MAX_PORTS] = {0};
            
            for (int i = 0; i < nb_rx; i++) {
                uint16_t port = packets[i].port;
                if (port < MAX_PORTS) {
                    packets_by_port[port]++;
                    bytes_by_port[port] += packets[i].length;
                    
                    // Count processed packets per port
                    if (g_config.use_gpu && i < nb_rx && results[i] != 0) {
                        processed_by_port[port]++;
                    }
                }
            }
            
            /* Update metrics in batch */
            for (uint16_t port = 0; port < MAX_PORTS; port++) {
                if (packets_by_port[port] > 0) {
                    update_metrics(port, packets_by_port[port], bytes_by_port[port], 
                                  g_config.use_gpu ? processed_by_port[port] : 0);
                }
            }
            
            /* Free packets */
            dpdk_free_packets(packets, nb_rx);
        }
    }
    
    /* Free GPU resources */
    if (gpu_buffer) {
        ebpf_gpu::EventProcessor::free_pinned_buffer(gpu_buffer);
    }
    
    /* Release GPU processor */
    g_processor.reset();
    
    std::cout << "\nExiting main loop. Printing final metrics..." << std::endl;
    print_metrics();
}

/* Print usage information */
static void print_usage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " [EAL options] -- [application options]" << std::endl;
    std::cout << "Application options:" << std::endl;
    std::cout << "  --kernel=PATH         Path to the CUDA kernel file (.cu or .ptx)" << std::endl;
    std::cout << "  --function=NAME       CUDA kernel function name to use" << std::endl;
    std::cout << "  --no-gpu              Disable GPU processing" << std::endl;
    std::cout << "  --device=ID           GPU device ID to use (-1 for auto)" << std::endl;
    std::cout << "  --batch-size=SIZE     Maximum batch size for GPU processing" << std::endl;
    std::cout << "  --help                Display this help message" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " --vdev=net_null0 -l 0 -- --kernel=examples/simple_packet_filter.cu --function=packet_filter" << std::endl;
    std::cout << "  " << program_name << " --vdev=net_tap0,iface=test0 -l 0 -- --no-gpu" << std::endl;
}

/* Parse application arguments */
static void parse_app_args(int argc, char *argv[])
{
    /* Default configuration */
    g_config.use_gpu = false;  /* Disabled by default */
    g_config.device_id = -1;   /* Auto-select */
    g_config.batch_size = 10000;
    g_config.kernel_path[0] = '\0';
    g_config.function_name[0] = '\0';
    
    /* Find the EAL arguments separator */
    int app_args_idx = 1;
    for (; app_args_idx < argc; app_args_idx++) {
        if (strcmp(argv[app_args_idx], "--") == 0) {
            break;
        }
    }
    
    /* Process application arguments */
    if (app_args_idx < argc) {
        for (int i = app_args_idx + 1; i < argc; i++) {
            if (strncmp(argv[i], "--kernel=", 9) == 0) {
                strncpy(g_config.kernel_path, argv[i] + 9, sizeof(g_config.kernel_path) - 1);
                g_config.kernel_path[sizeof(g_config.kernel_path) - 1] = '\0';
                g_config.use_gpu = true;  /* Enable GPU if kernel specified */
            } else if (strncmp(argv[i], "--function=", 11) == 0) {
                strncpy(g_config.function_name, argv[i] + 11, sizeof(g_config.function_name) - 1);
                g_config.function_name[sizeof(g_config.function_name) - 1] = '\0';
                g_config.use_gpu = true;  /* Enable GPU if function specified */
            } else if (strcmp(argv[i], "--no-gpu") == 0) {
                g_config.use_gpu = false;
            } else if (strncmp(argv[i], "--device=", 9) == 0) {
                g_config.device_id = atoi(argv[i] + 9);
            } else if (strncmp(argv[i], "--batch-size=", 13) == 0) {
                g_config.batch_size = atoi(argv[i] + 13);
            } else if (strcmp(argv[i], "--help") == 0) {
                print_usage(argv[0]);
                dpdk_cleanup();
                exit(EXIT_SUCCESS);
            }
        }
    }
    
    /* Validate configuration */
    if (g_config.use_gpu) {
        /* Both kernel and function must be specified */
        if (g_config.kernel_path[0] == '\0' || g_config.function_name[0] == '\0') {
            std::cerr << "Error: Both --kernel and --function must be specified for GPU processing" << std::endl;
            print_usage(argv[0]);
            dpdk_cleanup();
            exit(EXIT_FAILURE);
        }
    }
}

/* The main function */
int main(int argc, char *argv[])
{
    /* Install signal handler */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "DPDK Packet Processing with eBPF GPU Acceleration" << std::endl;
    std::cout << "===================================================" << std::endl;
    
    /* Initialize the driver with maximum burst size */
    dpdk_config_t config = DPDK_DEFAULT_CONFIG;
    config.burst_size = 1024;  /* Maximum burst size */
    
    int ret = dpdk_init(argc, argv, &config);
    if (ret != 0) {
        std::cerr << "Failed to initialize DPDK: " << ret << std::endl;
        exit(EXIT_FAILURE);
    }
    
    /* Parse application arguments */
    parse_app_args(argc, argv);
    
    /* Check if we have any ports */
    uint16_t port_count = dpdk_get_port_count();
    if (port_count == 0) {
        std::cerr << "No ports found! Make sure to use --vdev option." << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << argv[0] << " --vdev=net_null0 -l 0" << std::endl;
        std::cout << "  " << argv[0] << " --vdev=net_tap0,iface=test0 -l 0" << std::endl;
        std::cout << "  " << argv[0] << " --vdev=net_ring0 -l 0" << std::endl;
        dpdk_cleanup();
        exit(EXIT_FAILURE);
    }
    
    std::cout << "\nStarting packet processing with " << port_count << " ports..." << std::endl;
    std::cout << "GPU Processing: " << (g_config.use_gpu ? "Enabled" : "Disabled") << std::endl;
    
    /* Initialize GPU processor if enabled */
    if (g_config.use_gpu) {
        std::cout << "CUDA Kernel: " << g_config.kernel_path << std::endl;
        std::cout << "Function: " << g_config.function_name << std::endl;
        std::cout << "Device ID: " << g_config.device_id << std::endl;
        std::cout << "Batch Size: " << g_config.batch_size << std::endl;
        
        if (!init_gpu_processor()) {
            std::cerr << "Failed to initialize GPU processor, falling back to CPU mode" << std::endl;
            g_config.use_gpu = false;
        }
    }
    
    std::cout << "To generate packets:" << std::endl;
    std::cout << "  - null PMD automatically generates packets" << std::endl;
    std::cout << "  - For TAP: ping test0 (in another terminal)" << std::endl;
    std::cout << "  - Use tcpreplay, scapy, or other tools" << std::endl << std::endl;
    
    /* Run the main processing loop */
    main_loop();
    
    /* Clean up */
    dpdk_cleanup();
    
    std::cout << "Goodbye!" << std::endl;
    return 0;
} 