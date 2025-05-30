#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/queue.h>
#include <setjmp.h>
#include <stdarg.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>

#include <rte_common.h>
#include <rte_log.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_eal.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>

#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024

#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32

/* Metrics collection interval in seconds */
#define METRICS_INTERVAL 1

/* Fallback definition if not available in DPDK headers */
#ifndef RTE_MAX_ETHPORTS
#define RTE_MAX_ETHPORTS 32
#endif

/* Metrics structure */
struct port_metrics {
    uint64_t rx_packets;
    uint64_t rx_bytes;
    uint64_t rx_errors;
    uint64_t rx_missed;
    uint64_t tx_packets;
    uint64_t tx_bytes;
    uint64_t tx_errors;
    uint64_t tx_dropped;
    
    /* Rate calculations */
    uint64_t rx_pps;        /* packets per second */
    uint64_t rx_bps;        /* bytes per second */
    uint64_t tx_pps;
    uint64_t tx_bps;
    
    /* Previous values for rate calculation */
    uint64_t prev_rx_packets;
    uint64_t prev_rx_bytes;
    uint64_t prev_tx_packets;
    uint64_t prev_tx_bytes;
    
    /* Timing */
    struct timespec last_update;
};

struct global_metrics {
    uint64_t start_time_sec;
    uint64_t total_runtime_sec;
    uint64_t total_rx_packets;
    uint64_t total_rx_bytes;
    uint64_t total_tx_packets;
    uint64_t total_tx_bytes;
    uint64_t total_errors;
    
    /* Per-port metrics */
    struct port_metrics ports[RTE_MAX_ETHPORTS];
};

/* Port configuration - simplified for compatibility */
static const struct rte_eth_conf port_conf_default = {
    .rxmode = {
        /* Remove deprecated max_rx_pkt_len field */
    },
};

/* Global variables */
static volatile bool force_quit = false;
static uint64_t total_packets = 0;
static struct global_metrics g_metrics = {0};

/* Utility function to get current timestamp */
static uint64_t get_timestamp_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec;
}

static void get_timespec(struct timespec *ts)
{
    clock_gettime(CLOCK_MONOTONIC, ts);
}

/* Calculate time difference in seconds */
static double timespec_diff(struct timespec *start, struct timespec *end)
{
    return (end->tv_sec - start->tv_sec) + 
           (end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

/* Initialize metrics */
static void init_metrics(void)
{
    memset(&g_metrics, 0, sizeof(g_metrics));
    g_metrics.start_time_sec = get_timestamp_sec();
    
    /* Initialize per-port metrics */
    for (int i = 0; i < RTE_MAX_ETHPORTS; i++) {
        get_timespec(&g_metrics.ports[i].last_update);
    }
    
    printf("Metrics collection initialized\n");
}

/* Update port metrics */
static void update_port_metrics(uint16_t port, uint16_t nb_rx, uint64_t rx_bytes)
{
    struct port_metrics *pm = &g_metrics.ports[port];
    struct timespec now;
    
    /* Update counters */
    pm->rx_packets += nb_rx;
    pm->rx_bytes += rx_bytes;
    
    /* Update global counters */
    g_metrics.total_rx_packets += nb_rx;
    g_metrics.total_rx_bytes += rx_bytes;
    
    get_timespec(&now);
    
    /* Calculate rates every interval */
    double time_diff = timespec_diff(&pm->last_update, &now);
    if (time_diff >= METRICS_INTERVAL) {
        /* Calculate packets per second */
        pm->rx_pps = (pm->rx_packets - pm->prev_rx_packets) / time_diff;
        pm->rx_bps = (pm->rx_bytes - pm->prev_rx_bytes) / time_diff;
        
        /* Update previous values */
        pm->prev_rx_packets = pm->rx_packets;
        pm->prev_rx_bytes = pm->rx_bytes;
        pm->last_update = now;
    }
}

/* Collect hardware statistics */
static void collect_hw_stats(uint16_t port)
{
    struct port_metrics *pm = &g_metrics.ports[port];
    
    /* Try to collect hardware stats if available */
    #ifdef RTE_ETH_STATS_H_
    struct rte_eth_stats eth_stats;
    if (rte_eth_stats_get(port, &eth_stats) == 0) {
        pm->rx_errors = eth_stats.ierrors;
        pm->rx_missed = eth_stats.imissed;
        pm->tx_errors = eth_stats.oerrors;
        pm->tx_dropped = eth_stats.rx_nombuf;
        
        g_metrics.total_errors = pm->rx_errors + pm->tx_errors;
    }
    #else
    /* Fallback when DPDK stats are not available */
    (void)pm; /* Suppress unused variable warning */
    (void)port; /* Suppress unused variable warning */
    #endif
}

/* Print comprehensive metrics */
static void print_metrics(void)
{
    uint64_t current_time = get_timestamp_sec();
    g_metrics.total_runtime_sec = current_time - g_metrics.start_time_sec;
    
    printf("\n================================================================================\n");
    printf("DPDK PACKET PROCESSING METRICS\n");
    printf("================================================================================\n");
    printf("Runtime: %"PRIu64" seconds\n", g_metrics.total_runtime_sec);
    printf("Total RX Packets: %"PRIu64"\n", g_metrics.total_rx_packets);
    printf("Total RX Bytes: %"PRIu64" (%.2f MB)\n", 
           g_metrics.total_rx_bytes, 
           g_metrics.total_rx_bytes / (1024.0 * 1024.0));
    printf("Total Errors: %"PRIu64"\n", g_metrics.total_errors);
    
    if (g_metrics.total_runtime_sec > 0) {
        printf("Average RX Rate: %.2f pps, %.2f Mbps\n",
               (double)g_metrics.total_rx_packets / g_metrics.total_runtime_sec,
               (double)g_metrics.total_rx_bytes * 8 / g_metrics.total_runtime_sec / (1024*1024));
    }
    
    printf("\nPER-PORT STATISTICS:\n");
    printf("%-5s %-12s %-12s %-10s %-10s %-8s %-8s\n",
           "Port", "RX Packets", "RX Bytes", "RX PPS", "RX Mbps", "Errors", "Missed");
    printf("--------------------------------------------------------------------------------\n");
    
    uint16_t port;
    RTE_ETH_FOREACH_DEV(port) {
        struct port_metrics *pm = &g_metrics.ports[port];
        double rx_mbps = (double)pm->rx_bps * 8 / (1024 * 1024);
        
        printf("%-5u %-12"PRIu64" %-12"PRIu64" %-10"PRIu64" %-10.2f %-8"PRIu64" %-8"PRIu64"\n",
               port, pm->rx_packets, pm->rx_bytes, pm->rx_pps, rx_mbps,
               pm->rx_errors, pm->rx_missed);
    }
    printf("================================================================================\n");
}

/* Print periodic brief stats */
static void print_brief_stats(void)
{
    static uint64_t last_print_time = 0;
    uint64_t current_time = get_timestamp_sec();
    
    if (current_time - last_print_time >= METRICS_INTERVAL) {
        printf("[%"PRIu64"s] Total: %"PRIu64" pkts, %"PRIu64" bytes, %"PRIu64" errors\n",
               current_time - g_metrics.start_time_sec,
               g_metrics.total_rx_packets,
               g_metrics.total_rx_bytes,
               g_metrics.total_errors);
        last_print_time = current_time;
    }
}

/* Signal handler */
static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM) {
        printf("\nSignal %d received, preparing to exit...\n", signum);
        force_quit = true;
    }
}

/*
 * Initialize a given port using global settings and with the RX buffers
 * coming from the mbuf_pool passed as a parameter.
 */
static inline int
port_init(uint16_t port, struct rte_mempool *mbuf_pool)
{
    struct rte_eth_conf port_conf = port_conf_default;
    const uint16_t rx_rings = 1, tx_rings = 1;
    uint16_t nb_rxd = RX_RING_SIZE;
    uint16_t nb_txd = TX_RING_SIZE;
    int retval;
    uint16_t q;
    struct rte_eth_dev_info dev_info;
    struct rte_eth_txconf txconf;

    if (!rte_eth_dev_is_valid_port(port))
        return -1;

    retval = rte_eth_dev_info_get(port, &dev_info);
    if (retval != 0) {
        printf("Error during getting device (port %u) info: %s\n",
                port, strerror(-retval));
        return retval;
    }

    printf("Device info: driver=%s\n", dev_info.driver_name);

    /* Configure the Ethernet device. */
    retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
    if (retval != 0) {
        printf("Failed to configure port %u: %s\n", port, strerror(-retval));
        return retval;
    }

    retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
    if (retval != 0) {
        printf("Failed to adjust descriptors for port %u: %s\n", port, strerror(-retval));
        return retval;
    }

    /* Allocate and set up 1 RX queue per Ethernet port. */
    for (q = 0; q < rx_rings; q++) {
        retval = rte_eth_rx_queue_setup(port, q, nb_rxd,
                rte_eth_dev_socket_id(port), NULL, mbuf_pool);
        if (retval < 0) {
            printf("Failed to setup RX queue %u for port %u: %s\n", 
                   q, port, strerror(-retval));
            return retval;
        }
    }

    txconf = dev_info.default_txconf;
    txconf.offloads = port_conf.txmode.offloads;
    /* Allocate and set up 1 TX queue per Ethernet port. */
    for (q = 0; q < tx_rings; q++) {
        retval = rte_eth_tx_queue_setup(port, q, nb_txd,
                rte_eth_dev_socket_id(port), &txconf);
        if (retval < 0) {
            printf("Failed to setup TX queue %u for port %u: %s\n", 
                   q, port, strerror(-retval));
            return retval;
        }
    }

    /* Start the Ethernet port. */
    retval = rte_eth_dev_start(port);
    if (retval < 0) {
        printf("Failed to start port %u: %s\n", port, strerror(-retval));
        return retval;
    }

    /* Display the port MAC address. */
    struct rte_ether_addr addr;
    retval = rte_eth_macaddr_get(port, &addr);
    if (retval != 0) {
        printf("Failed to get MAC address for port %u: %s\n", port, strerror(-retval));
        return retval;
    }

    printf("Port %u MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8
           " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 "\n",
           port,
           addr.addr_bytes[0], addr.addr_bytes[1],
           addr.addr_bytes[2], addr.addr_bytes[3],
           addr.addr_bytes[4], addr.addr_bytes[5]);

    /* Enable RX in promiscuous mode for the Ethernet device. */
    retval = rte_eth_promiscuous_enable(port);
    if (retval != 0)
        printf("Warning: failed to enable promiscuous mode for port %u: %s\n",
               port, strerror(-retval));

    return 0;
}

/*
 * The lcore main function that processes packets
 */
static void
lcore_main(void)
{
    uint16_t port;
    uint64_t last_stats_time = 0;

    /*
     * Check that the port is on the same NUMA node as the polling thread
     * for best performance.
     */
    RTE_ETH_FOREACH_DEV(port)
        if (rte_eth_dev_socket_id(port) >= 0 &&
                rte_eth_dev_socket_id(port) !=
                        (int)rte_socket_id())
            printf("WARNING, port %u is on remote NUMA node to "
                    "polling thread.\n\tPerformance will "
                    "not be optimal.\n", port);

    printf("\nCore %u processing packets. [Ctrl+C to quit]\n",
            rte_lcore_id());

    /* Initialize metrics collection */
    init_metrics();

    /* Run until the application is quit or killed. */
    while (!force_quit) {
        uint64_t current_time = get_timestamp_sec();
        
        /*
         * Receive packets on each available port
         */
        RTE_ETH_FOREACH_DEV(port) {

            /* Get burst of RX packets */
            struct rte_mbuf *bufs[BURST_SIZE];
            const uint16_t nb_rx = rte_eth_rx_burst(port, 0,
                    bufs, BURST_SIZE);

            if (unlikely(nb_rx == 0))
                continue;

            total_packets += nb_rx;

            /* Calculate total bytes received */
            uint64_t total_bytes = 0;
            for (uint16_t i = 0; i < nb_rx; i++) {
                total_bytes += rte_pktmbuf_pkt_len(bufs[i]);
            }

            /* Update metrics */
            update_port_metrics(port, nb_rx, total_bytes);

            /* Print packet info for first few packets */
            if (total_packets <= 10) {
                printf("Received %u packets on port %u (total: %"PRIu64")\n", 
                       nb_rx, port, total_packets);
                
                for (uint16_t i = 0; i < nb_rx && i < 3; i++) {
                    printf("  Packet %u: length = %u bytes\n", 
                           i, rte_pktmbuf_pkt_len(bufs[i]));
                }
            }

            /* Free the mbufs */
            for (uint16_t i = 0; i < nb_rx; i++) {
                rte_pktmbuf_free(bufs[i]);
            }
        }

        /* Collect hardware stats periodically */
        if (current_time - last_stats_time >= METRICS_INTERVAL) {
            RTE_ETH_FOREACH_DEV(port) {
                collect_hw_stats(port);
            }
            last_stats_time = current_time;
        }

        /* Print brief periodic stats */
        print_brief_stats();
    }

    printf("\nExiting main loop. Printing final metrics...\n");
    print_metrics();
}

/*
 * The main function, which does initialization and calls the per-lcore
 * functions.
 */
int
main(int argc, char *argv[])
{
    struct rte_mempool *mbuf_pool;
    unsigned nb_ports;
    uint16_t portid;

    /* Install signal handler */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* Initialize the Environment Abstraction Layer (EAL). */
    int ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");

    argc -= ret;
    argv += ret;

    /* Check that there are ports available */
    nb_ports = rte_eth_dev_count_avail();
    printf("Found %u ports\n", nb_ports);

    if (nb_ports == 0) {
        printf("No ports found! Make sure to use --vdev option.\n");
        printf("Examples:\n");
        printf("  %s --vdev=net_null0 -l 0\n", argv[0]);
        printf("  %s --vdev=net_tap0,iface=test0 -l 0\n", argv[0]);
        printf("  %s --vdev=net_ring0 -l 0\n", argv[0]);
        rte_exit(EXIT_FAILURE, "No ports available\n");
    }

    /* Creates a new mempool in memory to hold the mbufs. */
    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS * nb_ports,
        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());

    if (mbuf_pool == NULL)
        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

    /* Initialize all ports. */
    RTE_ETH_FOREACH_DEV(portid)
        if (port_init(portid, mbuf_pool) != 0)
            rte_exit(EXIT_FAILURE, "Cannot init port %"PRIu16 "\n",
                    portid);

    if (rte_lcore_count() > 1)
        printf("\nWARNING: Too many lcores enabled. Only 1 used.\n");

    printf("\nStarting packet processing...\n");
    printf("To generate packets:\n");
    printf("  - null PMD automatically generates packets\n");
    printf("  - For TAP: ping test0 (in another terminal)\n");
    printf("  - Use tcpreplay, scapy, or other tools\n\n");

    /* Call lcore_main on the main core only. */
    lcore_main();

    /* Clean up before exit */
    printf("Cleaning up...\n");
    RTE_ETH_FOREACH_DEV(portid) {
        printf("Closing port %d...\n", portid);
        rte_eth_dev_stop(portid);
        rte_eth_dev_close(portid);
    }

    /* Clean up the EAL */
    rte_eal_cleanup();

    printf("Goodbye!\n");
    return 0;
} 