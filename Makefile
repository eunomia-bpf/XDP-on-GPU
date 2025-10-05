# eBPF GPU Processor Makefile
# Simple build system wrapper for CMake

BUILD_TYPE ?= Release
# Build tests by default for benchmarks
BUILD_TESTS ?= ON
BUILD_DIR ?= build
JOBS ?= $(shell nproc)
BUILD_DIR ?= build
JOBS ?= $(shell nproc)

.PHONY: all configure build test bench clean install help

# Default target
all: build

# Configure with CMake
configure:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DBUILD_TESTS=$(BUILD_TESTS) ..

# Build the project
build: configure
	cd $(BUILD_DIR) && make -j$(JOBS)

# Run tests (excluding benchmarks)
test: build
	cd $(BUILD_DIR) && ctest --output-on-failure -E "bench_"

# Run benchmark tests
bench: build
	./build/tests/test_performance
	./build/tests/test_performance_cpu

# Install
install: build
	cd $(BUILD_DIR) && make install

# Clean build directory
clean:
	rm -rf $(BUILD_DIR)

# Show help
help:
	@echo "eBPF GPU Processor Build System"
	@echo "==============================="
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build everything (default)"
	@echo "  configure    - Configure with CMake"
	@echo "  build        - Build the project"
	@echo "  test         - Run tests (excluding benchmarks)"
	@echo "  bench        - Run benchmark tests"
	@echo "  clean        - Clean build directory"
	@echo "  install      - Install the library"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  BUILD_TYPE   - Build type: Debug, Release (default: Release)"
	@echo "  BUILD_TESTS  - Build tests: ON, OFF (default: ON)"
	@echo "  BUILD_DIR    - Build directory (default: build)"
	@echo "  JOBS         - Number of parallel jobs (default: auto-detect)" 