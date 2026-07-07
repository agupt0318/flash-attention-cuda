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

# ---- CUDA (needs nvcc; binaries need an NVIDIA GPU) ----
$(BUILD)/%.cu.o: src/%.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/%.cu.o: tests/%.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/test_gpu: $(BUILD)/test_gpu.cu.o $(BUILD)/flash_fwd.cu.o \
                   $(BUILD)/reference.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD)/bench: $(BUILD)/bench.cu.o $(BUILD)/flash_fwd.cu.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

cuda: $(BUILD)/test_gpu $(BUILD)/bench      # compile without running (CI)

test-gpu: $(BUILD)/test_gpu
	$(BUILD)/test_gpu

bench: $(BUILD)/bench
	$(BUILD)/bench

clean:
	rm -rf $(BUILD)

-include $(BUILD)/*.d
.PHONY: test cuda test-gpu bench clean
