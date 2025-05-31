#ifndef DPDK_DRIVER_H
#define DPDK_DRIVER_H

#include <stdint.h>
#include <stdbool.h>

/* Driver configuration */
typedef struct {
    bool promiscuous_mode;     /* Enable promiscuous mode */
    uint16_t rx_ring_size;     /* RX ring buffer size */
    uint16_t tx_ring_size;     /* TX ring buffer size */
    uint16_t burst_size;       /* Maximum burst size */
} dpdk_config_t;

/* Default configuration values */
extern const dpdk_config_t DPDK_DEFAULT_CONFIG;

/* Packet data structure */
typedef struct {
    uint16_t port;             /* Port the packet was received on */
    void *data;                /* Packet data */
    uint32_t length;           /* Packet length in bytes */
} dpdk_packet_t;

/**
 * Initialize the DPDK environment and ports
 * 
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @param config Driver configuration (NULL for default)
 * @return 0 on success, negative on error
 */
int dpdk_init(int argc, char *argv[], const dpdk_config_t *config);

/**
 * Get the number of available DPDK ports
 * 
 * @return Number of ports
 */
uint16_t dpdk_get_port_count(void);

/**
 * Poll for packets on all ports
 * 
 * @param pkts Array to store received packets
 * @param max_pkts Maximum number of packets to receive
 * @param bytes_received Pointer to store total bytes received (can be NULL)
 * @return Number of packets received, negative on error
 */
int dpdk_poll(dpdk_packet_t *pkts, uint16_t max_pkts, uint64_t *bytes_received);

/**
 * Free a packet
 * 
 * @param pkt Packet to free
 */
void dpdk_free_packet(dpdk_packet_t *pkt);

/**
 * Free multiple packets
 * 
 * @param pkts Array of packets to free
 * @param count Number of packets to free
 */
void dpdk_free_packets(dpdk_packet_t *pkts, int count);

/**
 * Cleanup DPDK environment
 */
void dpdk_cleanup(void);

#endif /* DPDK_DRIVER_H */ 