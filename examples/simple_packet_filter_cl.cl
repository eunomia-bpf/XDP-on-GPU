/**
 * Simple OpenCL packet filter for eBPF on GPU
 * This kernel implements a basic packet filter that can check for specific patterns in packet data.
 */

/**
 * A simple packet filter function that checks if a packet contains a specific pattern.
 * 
 * @param packet_data The raw packet data buffer
 * @param packet_length The length of the packet in bytes
 * @param result Output parameter - set to 1 to accept packet, 0 to drop
 */
__kernel void packet_filter(
    __global const unsigned char* packet_data,
    const unsigned int packet_length,
    __global unsigned int* result
) {
    // Get the global thread ID
    const uint id = get_global_id(0);
    
    // Initialize result to 0 (drop)
    result[id] = 0;
    
    // Basic validation - ensure packet has a minimum size
    if (packet_length < 14) {
        return; // Drop undersized packets
    }
    
    // Example: Check for IPv4 Ethernet frame
    // 12-13 bytes: Ethernet type field (0x0800 for IPv4)
    if (packet_length >= 14 && 
        packet_data[12] == 0x08 && 
        packet_data[13] == 0x00) {
        
        // This is an IPv4 packet
        
        // IPv4 header starts at offset 14
        const uint ip_offset = 14;
        
        // Check if we have enough data for an IPv4 header (20 bytes minimum)
        if (packet_length >= ip_offset + 20) {
            
            // Get IP protocol field (offset 9 in IP header)
            const unsigned char protocol = packet_data[ip_offset + 9];
            
            // Example: Accept TCP (6) or UDP (17) packets
            if (protocol == 6 || protocol == 17) {
                result[id] = 1; // Accept packet
            }
        }
    }
    
    // Example: Also accept ARP packets (0x0806)
    if (packet_length >= 14 &&
        packet_data[12] == 0x08 && 
        packet_data[13] == 0x06) {
        result[id] = 1; // Accept ARP packets
    }
} 