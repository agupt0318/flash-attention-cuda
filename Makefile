# CPU targets build anywhere (clang/gcc); CUDA targets need nvcc.
# `make test`: CPU-side correctness (reference vs tiled algorithm)
# `make test-gpu` / `make bench`: require an NVIDIA GPU

CXX      ?= c++
NVCC     ?= nvcc
ARCH     ?= sm_70
BUILD    := build

CXXFLAGS  := -std=c++17 -O2 -Wall -Wextra -Isrc -MMD -MP
# -Xptxas=-v prints per-kernel register counts and spill bytes at
# compile time, the first thing to read when performance is flat.
NVCCFLAGS := -std=c++17 -O3 -arch=$(ARCH) -Isrc -lineinfo \
             -Xptxas=-v --compiler-options -Wall,-Wextra

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

# ---- Fast CPU kernel (NEON + query-tile blocking + threads) ----
$(BUILD)/test_cpu_fast: $(BUILD)/test_cpu_fast.o $(BUILD)/reference.o \
                        $(BUILD)/flash_cpu_fast.o
	$(CXX) $(CXXFLAGS) -pthread $^ -o $@

$(BUILD)/bench_cpu: $(BUILD)/bench_cpu.o $(BUILD)/flash_cpu.o \
                    $(BUILD)/flash_cpu_fast.o
	$(CXX) $(CXXFLAGS) -pthread $^ -o $@

test-fast: $(BUILD)/test_cpu_fast
	$(BUILD)/test_cpu_fast

bench-cpu: $(BUILD)/bench_cpu
	$(BUILD)/bench_cpu

# ---- On-device demo: a real TinyStories model through the CPU kernel ----
# Weights are downloaded, not committed (see README "Running a real model").
# Build with `make story`, then `./build/story -p "Once upon a time"`.
$(BUILD)/story: edge/story.cpp $(BUILD)/flash_cpu_fast.o
	$(CXX) $(CXXFLAGS) -pthread edge/story.cpp $(BUILD)/flash_cpu_fast.o -o $@

story: $(BUILD)/story

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

# What is the machine actually running? SASS of the kernel, for reading
# on machines that can compile but not execute (CI, this laptop).
sass: $(BUILD)/flash_fwd.cu.o
	cuobjdump -sass $< > $(BUILD)/flash_fwd.sass
	@grep -cE '^\s+/' $(BUILD)/flash_fwd.sass | xargs echo "instructions:"

test-gpu: $(BUILD)/test_gpu
	$(BUILD)/test_gpu

bench: $(BUILD)/bench
	$(BUILD)/bench

clean:
	rm -rf $(BUILD)

-include $(BUILD)/*.d
.PHONY: test cuda sass test-gpu bench clean
