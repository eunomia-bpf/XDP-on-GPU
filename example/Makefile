# DPDK Example Makefile

# Binary name
APP = dpdk_example

# Source files
SRCS = dpdk_example.c

# DPDK configuration
DPDK_CFLAGS = $(shell pkg-config --cflags libdpdk)
DPDK_LDFLAGS = $(shell pkg-config --libs libdpdk)

# Compiler and flags
CC = gcc
CFLAGS = -O3 -Wall -Wextra $(DPDK_CFLAGS)
LDFLAGS = $(DPDK_LDFLAGS)

# Build rule
$(APP): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Clean rule
clean:
	rm -f $(APP)

# Test with different virtual devices
test-null:
	sudo ./$(APP) --vdev=net_null0 -l 0

test-tap:
	sudo ./$(APP) --vdev=net_tap0,iface=test0 -l 0

test-ring:
	sudo ./$(APP) --vdev=net_ring0 -l 0

# Help
help:
	@echo "Available targets:"
	@echo "  $(APP)     - Build the application"
	@echo "  clean      - Clean build artifacts"
	@echo "  test-null  - Test with null PMD (auto-generates packets)"
	@echo "  test-tap   - Test with TAP interface"
	@echo "  test-ring  - Test with ring PMD"
	@echo "  help       - Show this help"

.PHONY: clean test-null test-tap test-ring help 