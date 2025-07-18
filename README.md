# eBPF XDP on GPU

A high-performance packet processing framework that brings the power of eBPF/XDP (eXpress Data Path) to GPU acceleration. This project enables XDP-style packet filtering and processing at unprecedented speeds by leveraging GPU parallelism while maintaining the familiar eBPF programming model.

## 🎯 What is eBPF XDP on GPU?

Traditional eBPF/XDP runs in the Linux kernel, processing packets at the earliest possible stage in the network stack. **eBPF XDP on GPU** takes this concept further by:

- **Offloading packet processing to GPUs** for massive parallelization
- **Maintaining XDP semantics** with familiar DROP/PASS/TX/REDIRECT actions
- **Processing millions of packets in parallel** using GPU cores
- **Integrating with DPDK** for high-speed packet I/O
- **Supporting both CUDA and OpenCL** backends for broad hardware compatibility

### Why GPU for XDP?

| Aspect | Traditional XDP | XDP on GPU |
|--------|----------------|------------|
| **Processing Model** | Per-CPU core | Thousands of GPU cores |
| **Packet Batch Size** | 1-64 packets | 10K-1M packets |
| **Complex Operations** | Limited by CPU | Hash tables, crypto, ML inference |
| **Throughput** | 10-20 Mpps per core | 100+ Mpps total |
| **Use Cases** | Simple filtering | Complex stateful processing |

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Network Traffic                       │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                      DPDK PMD                            │
│              (Poll Mode Driver)                          │
└─────────────────────┬───────────────────────────────────┘
                      │ Packet Batches
┌─────────────────────▼───────────────────────────────────┐
│               eBPF XDP on GPU                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │          EventProcessor API                      │   │
│  │  - Batch packet collection                       │   │
│  │  - GPU memory management                         │   │
│  │  - Kernel invocation                            │   │
│  └────────────────┬─────────────────────────────────┘   │
│                   │                                      │
│  ┌────────────────▼─────────────────────────────────┐   │
│  │         GPU Kernel Execution                     │   │
│  │  - Parallel packet processing                    │   │
│  │  - XDP action determination                      │   │
│  │  - Stateful operations                          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      │ XDP Actions
┌─────────────────────▼───────────────────────────────────┐
│                 Action Handler                           │
│        DROP | PASS | TX | REDIRECT                       │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

### 1. **XDP Action Semantics**
```cuda
// GPU kernel with XDP-style actions
__global__ void xdp_filter(NetworkEvent* events, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    NetworkEvent* pkt = &events[idx];
    
    // XDP_DROP by default
    pkt->action = XDP_DROP;
    
    // Process packet headers
    if (pkt->protocol == IPPROTO_TCP) {
        if (pkt->dst_port == 80 || pkt->dst_port == 443) {
            pkt->action = XDP_PASS;  // Allow HTTP/HTTPS
        }
    }
    
    // Can also implement XDP_TX, XDP_REDIRECT
}
```

### 2. **High-Performance Hash Tables**
```cuda
// FNV-1a hash for load balancing (like XDP samples)
__device__ uint32_t xdp_hash_tuple(uint32_t sip, uint32_t dip, 
                                   uint16_t sport, uint16_t dport) {
    uint32_t hash = 2166136261U;
    hash = (hash ^ sip) * 16777619U;
    hash = (hash ^ dip) * 16777619U;
    hash = (hash ^ sport) * 16777619U;
    hash = (hash ^ dport) * 16777619U;
    return hash;
}

__global__ void xdp_load_balancer(NetworkEvent* events, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    auto* pkt = &events[idx];
    uint32_t hash = xdp_hash_tuple(pkt->src_ip, pkt->dst_ip,
                                   pkt->src_port, pkt->dst_port);
    
    // Redirect to backend based on hash
    pkt->action = XDP_REDIRECT;
    pkt->redirect_ifindex = hash % NUM_BACKENDS;
}
```

### 3. **Stateful Processing**
Unlike CPU-based XDP, GPU allows complex stateful operations:
- Connection tracking
- Rate limiting with token buckets
- DDoS detection with sliding windows
- Crypto operations
- Pattern matching

### 4. **DPDK Integration**
```cpp
// DPDK + GPU processing loop
while (!quit) {
    // Receive packet burst from DPDK
    uint16_t nb_rx = rte_eth_rx_burst(port, queue, mbufs, BURST_SIZE);
    
    // Convert to GPU format
    for (int i = 0; i < nb_rx; i++) {
        parse_packet(mbufs[i], &events[i]);
    }
    
    // GPU processing (like XDP program)
    processor.processEvents(events, nb_rx);
    
    // Handle XDP actions
    for (int i = 0; i < nb_rx; i++) {
        switch (events[i].action) {
        case XDP_DROP:
            rte_pktmbuf_free(mbufs[i]);
            break;
        case XDP_PASS:
            forward_packet(mbufs[i]);
            break;
        case XDP_TX:
            reflect_packet(mbufs[i]);
            break;
        case XDP_REDIRECT:
            redirect_packet(mbufs[i], events[i].redirect_ifindex);
            break;
        }
    }
}
```

## 📊 Performance Comparison

### XDP on CPU vs GPU

Based on our benchmark results:

| Workload | Packet Count | GPU Performance | CPU Performance | Speedup |
|----------|--------------|-----------------|-----------------|---------|
| Simple Filter | 1M | 994 Mpps (1.01ms) | 741 Mpps (1.35ms) | 1.3x |
| Simple Filter | 10M | 1068 Mpps (9.37ms) | 405 Mpps (24.69ms) | 2.6x |
| Hash Load Balancer | 1M | 1011 Mpps (0.99ms) | 163 Mpps (6.12ms) | 6.2x |
| Hash Load Balancer | 10M | 1068 Mpps (9.36ms) | 78 Mpps (127.83ms) | 13.7x |

Key observations:
- GPU excels at complex operations (hash computing shows 13.7x speedup)
- GPU efficiency improves with larger batch sizes
- Simple operations show less speedup due to memory transfer overhead

### Latency Characteristics

- **CPU XDP**: ~50ns per packet (single packet)
- **GPU XDP**: ~1ms per batch (10K packets)
- **Effective**: ~100ns per packet at high load

## 🛠️ Building and Installation

### Prerequisites

```bash
# For CUDA backend
- CUDA Toolkit 11.0+
- NVIDIA GPU (Compute Capability 5.2+)

# For OpenCL backend  
- OpenCL 1.2+
- Any OpenCL-capable GPU

# For DPDK integration
- DPDK 21.11+
- Hugepages configured
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yunwei37/eBPF-on-GPU.git
cd eBPF-on-GPU

# Build with auto-detection
make

# Run tests
make test

# Run benchmarks
make bench
```

## 💻 Programming Guide

### Basic XDP Filter

```cpp
// 1. Define your XDP program (GPU kernel)
const char* xdp_program = R"(
extern "C" __global__ void xdp_prog(NetworkEvent* ctx, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    NetworkEvent* pkt = &ctx[idx];
    
    // Default XDP_DROP
    pkt->action = 0;
    
    // Parse and filter
    if (pkt->protocol == IPPROTO_ICMP) {
        pkt->action = 1; // XDP_PASS
    }
}
)";

// 2. Initialize GPU processor
EventProcessor processor;
processor.loadKernel(xdp_program, "xdp_prog");

// 3. Process packets
std::vector<NetworkEvent> packets(10000);
// ... fill with packet data ...

processor.processEvents(packets.data(), packets.size());

// 4. Handle results based on XDP actions
for (auto& pkt : packets) {
    handle_xdp_action(pkt);
}
```

### Advanced: Stateful Firewall

```cuda
// GPU memory for connection tracking
__device__ ConnectionTable conn_table;

extern "C" __global__ void xdp_stateful_fw(NetworkEvent* pkts, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    NetworkEvent* pkt = &pkts[idx];
    
    // Extract 5-tuple
    FlowKey key = {
        .src_ip = pkt->src_ip,
        .dst_ip = pkt->dst_ip,
        .src_port = pkt->src_port,
        .dst_port = pkt->dst_port,
        .proto = pkt->protocol
    };
    
    // Check connection state
    ConnState* state = conn_table.lookup(key);
    
    if (state && state->established) {
        pkt->action = XDP_PASS;
        state->packets++;
        state->bytes += pkt->length;
    } else if (is_syn_packet(pkt)) {
        // New connection
        conn_table.insert(key, CONN_SYN);
        pkt->action = XDP_PASS;
    } else {
        pkt->action = XDP_DROP;
    }
}
```

## 🔍 Use Cases

### 1. **High-Speed DDoS Mitigation**
- Process 100M+ packets/second
- Complex heuristics and ML-based detection
- Real-time blacklist updates

### 2. **Smart Load Balancing**
- Content-aware routing
- SSL/TLS session affinity
- Weighted round-robin with health checks

### 3. **Network Analytics**
- Line-rate packet sampling
- Flow aggregation
- Anomaly detection

### 4. **5G/Edge Computing**
- GTP tunnel processing
- Network slicing
- QoS enforcement

## 📈 Benchmarks

### Test Environment
- **GPU**: NVIDIA RTX 4090
- **CPU**: AMD EPYC 7742 64-Core
- **Network**: 100G Mellanox ConnectX-6

### Results

Based on actual benchmark data from our test environment:

```bash
# Run comprehensive benchmarks
make bench

# Actual benchmark results:
Simple Packet Filter (10M packets):
  GPU: 1068 Mpps (9.37ms total)
  CPU: 405 Mpps (24.69ms total)
  Speedup: 2.6x
  
Hash Load Balancer (10M packets):
  GPU: 1068 Mpps (9.36ms total)
  CPU: 78 Mpps (127.83ms total)  
  Speedup: 13.7x
  
Batch Hash Load Balancer (10M packets):
  GPU: 1075 Mpps (9.31ms total)
  CPU: 79 Mpps (127.09ms total)
  Speedup: 13.7x

# Per-packet processing time:
Simple Filter: GPU ~0.94ns/packet, CPU ~2.47ns/packet
Hash LB: GPU ~0.94ns/packet, CPU ~12.78ns/packet
```

Note: CPU measurements are single-threaded. Multi-core scaling would improve CPU performance linearly with core count.

## 🤝 Contributing

We welcome contributions! Areas of interest:

- eBPF bytecode to GPU kernel translation
- XDP feature parity (metadata, helpers)
- Performance optimizations
- Additional backends (Intel GPU, AMD GPU)
- Integration with kernel XDP

## 📚 Resources

- [eBPF/XDP Tutorial](https://github.com/xdp-project/xdp-tutorial)
- [DPDK Documentation](https://doc.dpdk.org/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Our Paper: "Accelerating XDP with GPUs"](papers/xdp-gpu.pdf)

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

This project builds upon:
- Linux kernel XDP implementation
- DPDK community
- eBPF ecosystem
- NVIDIA CUDA team

---

**Note**: This is not a kernel module. It's a userspace implementation that provides XDP-like packet processing semantics on GPUs for maximum performance.
