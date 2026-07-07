# CPU targets build anywhere (clang/gcc); CUDA targets need nvcc.
# `make test` — CPU-side correctness (reference vs tiled algorithm)
# `make test-gpu` / `make bench` — require an NVIDIA GPU

CXX      ?= c++
NVCC     ?= nvcc
ARCH     ?= sm_70
BUILD    := build

CXXFLAGS  := -std=c++17 -O2 -Wall -Wextra -Isrc -MMD -MP
NVCCFLAGS := -std=c++17 -O3 -arch=$(ARCH) -Isrc -lineinfo \
             --compiler-options -Wall,-Wextra

$(BUILD):
	mkdir -p $@

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: tests/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/test_cpu: $(BUILD)/test_cpu.o $(BUILD)/reference.o $(BUILD)/flash_cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(BUILD)/test_cpu
	$(BUILD)/test_cpu

clean:
	rm -rf $(BUILD)

-include $(BUILD)/*.d
.PHONY: test clean
