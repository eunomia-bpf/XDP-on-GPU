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

/* Port configuration - simplified for compatibility */
static const struct rte_eth_conf port_conf_default = {
    .rxmode = {
        /* Remove deprecated max_rx_pkt_len field */
    },
};

/* Global variables */
static volatile bool force_quit = false;
static uint64_t total_packets = 0;

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

    /* Run until the application is quit or killed. */
    while (!force_quit) {
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

            /* Print packet info for first few packets */
            if (total_packets <= 10) {
                printf("Received %u packets on port %u (total: %"PRIu64")\n", 
                       nb_rx, port, total_packets);
                
                for (uint16_t i = 0; i < nb_rx && i < 3; i++) {
                    printf("  Packet %u: length = %u bytes\n", 
                           i, rte_pktmbuf_pkt_len(bufs[i]));
                }
            }

            /* Print periodic stats */
            if (total_packets % 1000 == 0 && total_packets > 0) {
                printf("Total packets processed: %"PRIu64"\n", total_packets);
            }

            /* Free the mbufs */
            for (uint16_t i = 0; i < nb_rx; i++) {
                rte_pktmbuf_free(bufs[i]);
            }
        }
    }

    printf("\nExiting main loop. Total packets processed: %"PRIu64"\n", total_packets);
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