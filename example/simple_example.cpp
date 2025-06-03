#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <thread>
#include <iomanip>
#include <random>

// Include the eBPF GPU processor header
#include "ebpf_gpu_processor.hpp"
#include "kernel_loader.hpp"

// For direct OpenCL access if needed
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Structure representing a simple packet
struct SimplePacket {
    uint8_t data[1500];  // Fixed size array for packet data
    uint32_t length;     // Actual length of the packet
};

// Generate random packet data
void generate_random_packets(std::vector<SimplePacket>& packets, int num_packets) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> len_dist(64, 1500);
    std::uniform_int_distribution<> data_dist(0, 255);
    
    packets.resize(num_packets);
    
    for (int i = 0; i < num_packets; i++) {
        packets[i].length = len_dist(gen);
        for (uint32_t j = 0; j < packets[i].length; j++) {
            packets[i].data[j] = static_cast<uint8_t>(data_dist(gen));
        }
    }
}

// Print packet hexdump
void print_packet(const SimplePacket& packet, int max_bytes = 64) {
    const int bytes_per_line = 16;
    int bytes_to_print = std::min(static_cast<int>(packet.length), max_bytes);
    
    std::cout << "Packet length: " << packet.length << " bytes\n";
    std::cout << "Data hexdump (first " << bytes_to_print << " bytes):\n";
    
    for (int i = 0; i < bytes_to_print; i += bytes_per_line) {
        std::cout << std::setw(4) << std::setfill('0') << std::hex << i << ": ";
        
        // Print hex values
        for (int j = 0; j < bytes_per_line; j++) {
            if (i + j < bytes_to_print) {
                std::cout << std::setw(2) << std::setfill('0') << std::hex 
                          << static_cast<int>(packet.data[i + j]) << " ";
            } else {
                std::cout << "   ";
            }
        }
        
        // Print ASCII values
        std::cout << " | ";
        for (int j = 0; j < bytes_per_line; j++) {
            if (i + j < bytes_to_print) {
                char c = packet.data[i + j];
                if (c >= 32 && c <= 126) { // Printable ASCII
                    std::cout << c;
                } else {
                    std::cout << ".";
                }
            } else {
                std::cout << " ";
            }
        }
        
        std::cout << std::endl;
    }
    std::cout << std::dec; // Reset to decimal
}

// Helper to determine backend type as string
std::string backend_type_to_string(ebpf_gpu::BackendType type) {
    switch (type) {
        case ebpf_gpu::BackendType::CUDA: return "CUDA";
        case ebpf_gpu::BackendType::OpenCL: return "OpenCL";
        default: return "Unknown";
    }
}

// Helper to get default kernel file based on backend
std::string get_default_kernel_file(ebpf_gpu::BackendType type) {
    switch (type) {
        case ebpf_gpu::BackendType::CUDA:
            return "../examples/simple_packet_filter.cu";
        case ebpf_gpu::BackendType::OpenCL:
            return "../examples/simple_packet_filter_cl.cl";
        default:
            // Default to CUDA
            return "../examples/simple_packet_filter.cu";
    }
}

int main(int argc, char* argv[]) {
    // Get the current backend type
    ebpf_gpu::BackendType backend = ebpf_gpu::get_backend_type();
    
    // Parse command line arguments
    std::string kernel_file = get_default_kernel_file(backend);
    std::string kernel_function = "packet_filter";
    int num_packets = 1000;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            kernel_file = argv[++i];
        } else if (strcmp(argv[i], "--function") == 0 && i + 1 < argc) {
            kernel_function = argv[++i];
        } else if (strcmp(argv[i], "--packets") == 0 && i + 1 < argc) {
            num_packets = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --kernel FILE     eBPF kernel file path\n"
                      << "  --function NAME   Kernel function name\n"
                      << "  --packets NUM     Number of packets to process\n"
                      << "  --help            Show this help message\n";
            return 0;
        }
    }
    
    std::cout << "Simple eBPF GPU Processor Example\n";
    std::cout << "--------------------------------\n";
    std::cout << "Backend: " << backend_type_to_string(backend) << "\n";
    std::cout << "Kernel file: " << kernel_file << "\n";
    std::cout << "Kernel function: " << kernel_function << "\n";
    std::cout << "Number of packets: " << num_packets << "\n\n";
    
    try {
        // Generate random packets for testing
        std::vector<SimplePacket> packets;
        generate_random_packets(packets, num_packets);
        
        // Print a sample packet
        std::cout << "Sample input packet:\n";
        print_packet(packets[0]);
        std::cout << "\n";
        
        // Initialize the eBPF GPU processor
        ebpf_gpu::EventProcessor processor;
        
        // Load the eBPF program
        auto result = processor.load_kernel_from_file(kernel_file, kernel_function);
        if (result != ebpf_gpu::ProcessingResult::Success) {
            std::cerr << "Failed to load eBPF program from file: " << kernel_file << std::endl;
            return 1;
        }
        
        std::cout << "eBPF program loaded successfully\n";
        
        // Process a single packet using the direct data approach
        // We'll create a buffer containing:
        // 1. The packet data
        // 2. Room for the result at the end
        
        SimplePacket& packet = packets[0];
        
        // Create a buffer to hold the packet data and result
        std::vector<uint8_t> packet_buffer(packet.length + sizeof(uint32_t));
        
        // Copy packet data to the buffer
        std::memcpy(packet_buffer.data(), packet.data, packet.length);
        
        // Set result to 0 initially
        uint32_t* result_ptr = reinterpret_cast<uint32_t*>(packet_buffer.data() + packet.length);
        *result_ptr = 0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process the packet using the layout-independent approach
        result = processor.process_event(packet_buffer.data(), packet_buffer.size());
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        if (result != ebpf_gpu::ProcessingResult::Success) {
            std::cerr << "Failed to process packet" << std::endl;
        } else {
            std::cout << "Successfully processed packet" << std::endl;
            // Read the result from the end of the buffer
            uint32_t packet_result = *result_ptr;
            std::cout << "Packet action: " << (packet_result == 1 ? "ACCEPT" : "DROP") << std::endl;
        }
        
        std::cout << "eBPF program executed" << std::endl;
        std::cout << "Execution time: " << duration.count() / 1000.0 << " ms" << std::endl;
        
        // Process multiple packets in batch
        if (result == ebpf_gpu::ProcessingResult::Success) {
            std::cout << "\nProcessing 10 packets in batch...\n";
            
            // Create a buffer to hold multiple packets and their results
            std::vector<std::vector<uint8_t>> batch_buffers;
            
            // Prepare 10 packets for batch processing
            for (int i = 0; i < 10; i++) {
                SimplePacket& batch_packet = packets[i];
                std::vector<uint8_t> buffer(batch_packet.length + sizeof(uint32_t));
                
                // Copy packet data
                std::memcpy(buffer.data(), batch_packet.data, batch_packet.length);
                
                // Initialize result to 0
                uint32_t* res_ptr = reinterpret_cast<uint32_t*>(buffer.data() + batch_packet.length);
                *res_ptr = 0;
                
                batch_buffers.push_back(std::move(buffer));
            }
            
            // Process each packet individually for simplicity
            start_time = std::chrono::high_resolution_clock::now();
            
            int accepted = 0;
            for (auto& buffer : batch_buffers) {
                result = processor.process_event(buffer.data(), buffer.size());
                
                if (result == ebpf_gpu::ProcessingResult::Success) {
                    // Check the result
                    uint32_t* res_ptr = reinterpret_cast<uint32_t*>(
                        buffer.data() + buffer.size() - sizeof(uint32_t));
                    if (*res_ptr == 1) {
                        accepted++;
                    }
                }
            }
            
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            std::cout << "Successfully processed batch packets" << std::endl;
            std::cout << "Accepted packets: " << accepted << " out of 10" << std::endl;
            std::cout << "Batch execution time: " << duration.count() / 1000.0 << " ms" << std::endl;
            std::cout << "Batch throughput: " << (10 * 1000000.0 / duration.count()) 
                      << " packets/second" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 