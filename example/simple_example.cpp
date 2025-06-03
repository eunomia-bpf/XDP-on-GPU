#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <thread>
#include <iomanip>
#include <random>

// Include the eBPF GPU processor header - we need the full namespace
#include "ebpf_gpu_processor.hpp"

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

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string kernel_file = "examples/simple_packet_filter.cu";
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
        
        // Initialize the eBPF GPU processor with explicit namespace
        ebpf_gpu::EventProcessor processor;
        
        // Load the eBPF program
        auto result = processor.load_kernel_from_file(kernel_file, kernel_function);
        if (result != ebpf_gpu::ProcessingResult::Success) {
            std::cerr << "Failed to load eBPF program from file: " << kernel_file << std::endl;
            return 1;
        }
        
        std::cout << "eBPF program loaded successfully\n";
        
        // Prepare input and output buffers
        std::vector<void*> input_ptrs(num_packets);
        std::vector<uint32_t> input_sizes(num_packets);
        
        for (int i = 0; i < num_packets; i++) {
            input_ptrs[i] = packets[i].data;
            input_sizes[i] = packets[i].length;
        }
        
        // Execute the eBPF program on the GPU
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process packets in batch
        result = processor.process_events(packets.data(), 
                                          num_packets * sizeof(SimplePacket),
                                          num_packets);
        
        if (result != ebpf_gpu::ProcessingResult::Success) {
            std::cerr << "Failed to run eBPF program" << std::endl;
            return 1;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Here we would normally analyze the results
        // Since we don't have direct access to the results in this example,
        // we'll just print some stats
        
        std::cout << "eBPF program executed successfully\n";
        std::cout << "Execution time: " << duration.count() / 1000.0 << " ms\n";
        std::cout << "Throughput: " << (num_packets * 1000000.0 / duration.count()) 
                  << " packets/second\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 