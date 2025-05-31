#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <signal.h>
#include <stdbool.h>
#include <time.h>       /* Needed for struct timespec and clock_gettime */
#include <sys/time.h>
#include <unistd.h>     /* Needed for usleep */
#include <errno.h>      /* For errno */

#include "dpdk_driver.h"

/* Configuration */
#define METRICS_INTERVAL 1
#define MAX_PACKETS_PER_POLL 1024
#define MAX_PORTS 32

/* Simple metrics structure */
struct port_metrics {
    uint64_t rx_packets;
    uint64_t rx_bytes;
};

/* Global metrics */
struct {
    uint64_t start_time_sec;
    uint64_t total_rx_packets;
    uint64_t total_rx_bytes;
    struct port_metrics ports[MAX_PORTS];
} g_metrics = {0};

/* Global variables */
static volatile bool force_quit = false;

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
    printf("Metrics collection initialized\n");
}

/* Ultra-fast port metrics update - minimal code */
static inline void update_metrics(uint16_t port, uint32_t nb_rx, uint64_t rx_bytes)
{
    if (port < MAX_PORTS) {
        g_metrics.ports[port].rx_packets += nb_rx;
        g_metrics.ports[port].rx_bytes += rx_bytes;
        g_metrics.total_rx_packets += nb_rx;
        g_metrics.total_rx_bytes += rx_bytes;
    }
}

/* Print final metrics */
static void print_metrics(void)
{
    uint64_t current_time = get_timestamp_sec();
    uint64_t runtime_sec = current_time - g_metrics.start_time_sec;
    uint16_t num_ports = dpdk_get_port_count();
    
    printf("\n================================================================================\n");
    printf("DPDK PACKET PROCESSING METRICS\n");
    printf("================================================================================\n");
    printf("Runtime: %"PRIu64" seconds\n", runtime_sec);
    printf("Total RX Packets: %"PRIu64"\n", g_metrics.total_rx_packets);
    printf("Total RX Bytes: %"PRIu64" (%.2f MB)\n", 
           g_metrics.total_rx_bytes, 
           g_metrics.total_rx_bytes / (1024.0 * 1024.0));
    
    if (runtime_sec > 0) {
        printf("Average RX Rate: %.2f pps, %.2f Mbps\n",
               (double)g_metrics.total_rx_packets / runtime_sec,
               (double)g_metrics.total_rx_bytes * 8 / runtime_sec / (1024*1024));
    }
    
    printf("\nPER-PORT STATISTICS:\n");
    printf("%-5s %-12s %-12s\n",
           "Port", "RX Packets", "RX Bytes");
    printf("--------------------------------------------------------------------------------\n");
    
    for (uint16_t port = 0; port < num_ports && port < MAX_PORTS; port++) {
        printf("%-5u %-12"PRIu64" %-12"PRIu64"\n",
               port, g_metrics.ports[port].rx_packets, g_metrics.ports[port].rx_bytes);
    }
    printf("================================================================================\n");
}

/* Signal handler */
static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM) {
        printf("\nSignal %d received, preparing to exit...\n", signum);
        force_quit = true;
    }
}

/* The main processing loop - ultra minimalist */
static void main_loop(void)
{
    printf("\nProcessing packets. [Ctrl+C to quit]\n");
    
    /* Initialize metrics */
    init_metrics();
    
    /* Pre-allocate packet array */
    dpdk_packet_t packets[MAX_PACKETS_PER_POLL];
    
    /* Track if we've shown packet samples */
    uint64_t display_count = 0;
    
    /* Run until the application is quit or killed */
    while (!force_quit) {
        /* Poll for packets */
        uint64_t bytes_received = 0;
        int nb_rx = dpdk_poll(packets, MAX_PACKETS_PER_POLL, &bytes_received);
        
        if (nb_rx < 0) {
            printf("Error polling for packets: %d\n", nb_rx);
            break;
        }
        
        if (nb_rx > 0) {
            /* Display first few packets */
            for (int i = 0; i < nb_rx && display_count < 10; i++, display_count++) {
                printf("Packet on port %u: length = %u bytes\n", 
                       packets[i].port, packets[i].length);
            }
            
            /* Count packets by port */
            uint16_t packets_by_port[MAX_PORTS] = {0};
            uint64_t bytes_by_port[MAX_PORTS] = {0};
            
            for (int i = 0; i < nb_rx; i++) {
                uint16_t port = packets[i].port;
                if (port < MAX_PORTS) {
                    packets_by_port[port]++;
                    bytes_by_port[port] += packets[i].length;
                }
            }
            
            /* Update metrics in batch */
            for (uint16_t port = 0; port < MAX_PORTS; port++) {
                if (packets_by_port[port] > 0) {
                    update_metrics(port, packets_by_port[port], bytes_by_port[port]);
                }
            }
            
            /* Free packets */
            dpdk_free_packets(packets, nb_rx);
        }
    }
    
    printf("\nExiting main loop. Printing final metrics...\n");
    print_metrics();
}

/* The main function */
int main(int argc, char *argv[])
{
    /* Install signal handler */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("DPDK Packet Processing Application\n");
    printf("===================================\n");
    
    /* Initialize the driver with maximum burst size */
    dpdk_config_t config = DPDK_DEFAULT_CONFIG;
    config.burst_size = 1024;  /* Maximum burst size */
    
    int ret = dpdk_init(argc, argv, &config);
    if (ret != 0) {
        printf("Failed to initialize DPDK: %d\n", ret);
        exit(EXIT_FAILURE);
    }
    
    /* Check if we have any ports */
    uint16_t port_count = dpdk_get_port_count();
    if (port_count == 0) {
        printf("No ports found! Make sure to use --vdev option.\n");
        printf("Examples:\n");
        printf("  %s --vdev=net_null0 -l 0\n", argv[0]);
        printf("  %s --vdev=net_tap0,iface=test0 -l 0\n", argv[0]);
        printf("  %s --vdev=net_ring0 -l 0\n", argv[0]);
        dpdk_cleanup();
        exit(EXIT_FAILURE);
    }
    
    printf("\nStarting packet processing with %u ports...\n", port_count);
    printf("To generate packets:\n");
    printf("  - null PMD automatically generates packets\n");
    printf("  - For TAP: ping test0 (in another terminal)\n");
    printf("  - Use tcpreplay, scapy, or other tools\n\n");
    
    /* Run the main processing loop */
    main_loop();
    
    /* Clean up */
    dpdk_cleanup();
    
    printf("Goodbye!\n");
    return 0;
} 