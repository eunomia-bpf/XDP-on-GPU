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
#include "../include/ebpf_gpu_processor.hpp"
#include "../include/kernel_loader.hpp"

// Structure representing a simple packet
struct SimplePacket {
    uint8_t data[1500];
    uint32_t length;
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
        
        // Create a buffer for results (1 = accept, 0 = drop)
        std::vector<uint32_t> results(num_packets, 0);
        
        // Prepare packet data for GPU processing
        // We need to flatten the packet data into a single buffer and keep track of offsets
        std::vector<uint8_t> packet_data_buffer;
        std::vector<uint32_t> packet_offsets(num_packets + 1, 0);
        
        // Calculate total buffer size and offsets
        size_t total_size = 0;
        for (int i = 0; i < num_packets; i++) {
            packet_offsets[i] = total_size;
            total_size += packets[i].length;
        }
        packet_offsets[num_packets] = total_size;
        
        // Allocate and fill packet data buffer
        packet_data_buffer.resize(total_size);
        for (int i = 0; i < num_packets; i++) {
            std::memcpy(packet_data_buffer.data() + packet_offsets[i], 
                       packets[i].data, 
                       packets[i].length);
        }
        
        // Custom event structure that matches our OpenCL kernel expectations
        struct PacketEvent {
            uint32_t offset;     // Offset in the packet data buffer
            uint32_t length;     // Length of the packet
            uint32_t result;     // Result of processing (output)
        };
        
        // Create event buffer
        std::vector<PacketEvent> events(num_packets);
        for (int i = 0; i < num_packets; i++) {
            events[i].offset = packet_offsets[i];
            events[i].length = packets[i].length;
            events[i].result = 0;  // Initialize to 0 (drop)
        }
        
        // Execute the eBPF program on the GPU
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Set up kernel arguments for both packet data and event buffer
        std::vector<void*> args = {
            packet_data_buffer.data(),   // Packet data buffer
            events.data()                // Event buffer with offsets, lengths, and results
        };
        
        // Process packets in batch
        result = processor.process_events(events.data(), 
                                          events.size() * sizeof(PacketEvent),
                                          events.size());
        
        if (result != ebpf_gpu::ProcessingResult::Success) {
            std::cerr << "Failed to run eBPF program" << std::endl;
            return 1;
        }
        
        // Now we can analyze the results
        int accepted_packets = 0;
        for (int i = 0; i < num_packets; i++) {
            if (events[i].result == 1) {
                accepted_packets++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "eBPF program executed successfully\n";
        std::cout << "Execution time: " << duration.count() / 1000.0 << " ms\n";
        std::cout << "Throughput: " << (num_packets * 1000000.0 / duration.count()) 
                  << " packets/second\n";
        std::cout << "Accepted packets: " << accepted_packets << " out of " << num_packets 
                  << " (" << (accepted_packets * 100.0 / num_packets) << "%)\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 