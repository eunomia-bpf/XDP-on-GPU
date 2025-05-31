# DPDK Example Makefile

# Binary name
APP = dpdk_example

# Source files
SRCS = dpdk_example.c dpdk_driver.c

# Object files
OBJS = $(SRCS:.c=.o)

# Header files
HEADERS = dpdk_driver.h

# DPDK configuration
DPDK_CFLAGS = $(shell pkg-config --cflags libdpdk)
DPDK_LDFLAGS = $(shell pkg-config --libs libdpdk)

# Compiler and flags
CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c99 -D_GNU_SOURCE -include rte_config.h $(DPDK_CFLAGS)
LDFLAGS = $(DPDK_LDFLAGS) -lrt -lpthread -ldl

# Default target
all: $(APP)

# Build the application
$(APP): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build object files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS += -g -DDEBUG -O0
debug: clean all

# Clean rule
clean:
	rm -f $(APP) $(OBJS)

# Test with different virtual devices
test-null:
	sudo ./$(APP) --vdev=net_null0 -l 0

test-tap:
	sudo ./$(APP) --vdev=net_tap0,iface=test0 -l 0

test-ring:
	sudo ./$(APP) --vdev=net_ring0 -l 0

test-multi:
	sudo ./$(APP) --vdev=net_null0 --vdev=net_null1 -l 0

# Show DPDK configuration
dpdk-info:
	@echo "DPDK CFLAGS: $(DPDK_CFLAGS)"
	@echo "DPDK LDFLAGS: $(DPDK_LDFLAGS)"
	@pkg-config --modversion libdpdk

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Build the application (default)"
	@echo "  debug      - Build debug version"
	@echo "  clean      - Clean build artifacts"
	@echo "  test-null  - Test with null PMD (auto-generates packets)"
	@echo "  test-tap   - Test with TAP interface"
	@echo "  test-ring  - Test with ring PMD"
	@echo "  test-multi - Test with multiple devices"
	@echo "  dpdk-info  - Show DPDK configuration"
	@echo "  help       - Show this help"

.PHONY: all debug clean test-null test-tap test-ring test-multi dpdk-info help 