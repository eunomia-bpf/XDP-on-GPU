# Simple DPDK Packet Processor Example

This is a minimal DPDK example that demonstrates packet processing using virtual devices. **No GPU or physical NICs required!**

## Features

- Works with DPDK virtual devices (null PMD, TAP, ring)
- Simple packet capture and processing
- Easy to test and understand
- No hardware dependencies

## Quick Start

### 1. Install DPDK (if not already installed)

```bash
sudo apt update
sudo apt install -y dpdk dpdk-dev libdpdk-dev
```

### 2. Run the Test

```bash
# Run the automated test script
sudo ./simple_test.sh
```

The script will:
- Compile the application
- Set up hugepages
- Test with null PMD (auto-generates packets)
- Test with TAP interface (creates virtual network interface)

### 3. Manual Testing

```bash
# Compile manually
gcc -I/usr/include/dpdk -I/usr/include/x86_64-linux-gnu/dpdk \
    -include rte_config.h -march=native \
    minimal_dpdk_example.c -o minimal_dpdk_example \
    $(pkg-config --libs libdpdk)

# Test with null PMD (auto-generates packets)
sudo ./minimal_dpdk_example --vdev=net_null0 -l 0

# Test with TAP interface
sudo ./minimal_dpdk_example --vdev=net_tap0,iface=test0 -l 0
# In another terminal: ping test0
```

## Understanding the Code

The `minimal_dpdk_example.c` shows:

1. **DPDK Initialization**: Sets up the DPDK Environment Abstraction Layer (EAL)
2. **Port Setup**: Configures virtual network ports
3. **Packet Reception**: Captures packets using `rte_eth_rx_burst()`
4. **Packet Processing**: Simple packet inspection and statistics
5. **Memory Management**: Proper cleanup of packet buffers

## Virtual Device Types

### null PMD
- **Usage**: `--vdev=net_null0`
- **Purpose**: Auto-generates packets for testing
- **Good for**: Performance testing, basic functionality verification

### TAP PMD
- **Usage**: `--vdev=net_tap0,iface=test0`
- **Purpose**: Creates a virtual network interface accessible from the host
- **Good for**: Real packet testing, integration with host networking

### Ring PMD
- **Usage**: `--vdev=net_ring0`
- **Purpose**: Creates ring-based packet sharing between applications
- **Good for**: Inter-process communication, complex testing scenarios

## Expected Output

```
=== Setting up DPDK test environment ===
1. Compiling the DPDK application...
✓ Compilation successful!
2. Setting up hugepages...
Hugepages already configured: 256
3. Testing different virtual devices...

=== Test 1: null PMD (auto-generates packets) ===
Running for 5 seconds...
Found 1 ports
Device info: driver=net_null
Port 0 MAC: 02 00 00 00 00 00
Starting packet processing...
...
Core 0 processing packets. [Ctrl+C to quit]
Received 32 packets on port 0 (total: 32)
  Packet 0: length = 64 bytes
  Packet 1: length = 64 bytes
  Packet 2: length = 64 bytes
Total packets processed: 1000
...
Test completed

=== Test 2: TAP interface ===
Creating TAP interface...
Starting DPDK application with TAP interface...
...
✓ TAP interface 'test0' created successfully
✓ TAP test completed

=== Test Summary ===
Both tests demonstrate DPDK packet processing with virtual devices.
...
```

## Next Steps

This example provides a foundation for:

1. **Adding Custom Packet Processing**: Modify the packet processing logic in `lcore_main()`
2. **Integration with Other Systems**: Use TAP interfaces to connect with real networks
3. **Performance Testing**: Use null PMD for high-throughput testing
4. **GPU Integration**: Add GPU processing to the packet handling pipeline

## Troubleshooting

### Compilation Errors
- Make sure DPDK is properly installed: `pkg-config --exists libdpdk`
- Check include paths: `pkg-config --cflags libdpdk`

### Runtime Errors
- **No hugepages**: Run `sudo ./simple_test.sh` or set up hugepages manually
- **Permission denied**: Run with `sudo` or add user to appropriate groups
- **No ports found**: Make sure to use `--vdev` parameter

### Common Issues
- **CPU instruction errors**: The script tries different compilation flags automatically
- **Interface creation fails**: Check if you have permissions to create network interfaces

## Learn More

- [DPDK Documentation](https://doc.dpdk.org/)
- [DPDK Programmer's Guide](https://doc.dpdk.org/guides/prog_guide/)
- [Virtual Device Examples](https://doc.dpdk.org/guides/nics/) 