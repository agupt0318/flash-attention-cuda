# flash-attention-cuda

> [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) implemented from scratch in CUDA. Exact attention computed tile-by-tile in on-chip SRAM, so the N×N score matrix **never exists in HBM**.

![ci](https://github.com/agupt0318/flash-attention-cuda/actions/workflows/ci.yml/badge.svg)
![cuda](https://img.shields.io/badge/CUDA-fp32%20forward-76b900)
![license](https://img.shields.io/badge/license-MIT-lightgrey)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agupt0318/flash-attention-cuda/blob/main/demo/flash_attention_colab.ipynb)

<p align="center">
  <img src="assets/tiling.svg" width="880" alt="Animated view of the kernel: a query tile pinned in registers while K/V tiles stream through SRAM; the online-softmax statistics update and the output sharpens tile by tile">
</p>

## The idea

Attention is `O = softmax(QKᵀ/√d)·V`. The standard implementation writes the N×N matrices `S` and `P` to GPU HBM and reads them back. At N=4K that is gigabytes of traffic per head-batch, all of it for intermediates the caller discards. Most of "attention is slow" is that traffic: the operation is memory-bound, and HBM is an order of magnitude slower than on-chip SRAM.

FlashAttention restructures the computation so each K/V tile is loaded to SRAM once and fully consumed. Tiling stays legal thanks to the **online softmax**: softmax over a concatenation decomposes if you carry a running max `m` and normalizer `ℓ`,

```
m'   = max(m, s)                       // new score s arrives
ℓ'   = ℓ·e^(m−m') + e^(s−m')
acc' = acc·e^(m−m') + e^(s−m')·v
```

so the output row is `acc/ℓ` once the last tile folds in: exact, with `O(N)` extra memory, down from the `O(N²)` a materialized score matrix costs. HBM accesses drop from `Θ(Nd + N²)` to `Θ(N²d²M⁻¹)` (Theorem 2 of the paper).

## The kernel

[src/flash_fwd.cu](src/flash_fwd.cu) is the fp32 forward kernel: head dims 32/64, optional causal mask.

- **Grid**: one CTA per (query tile, batch·head); one thread per query row. The row's `q`, running `m`/`ℓ`, and unnormalized accumulator live in **registers** for the entire pass. The only HBM writes are the final `O` row and the logsumexp.
- **K/V stream through shared memory** in 64-row tiles, loaded cooperatively by the whole CTA. Score matrix rows exist one register at a time.
- **Softmax statistics** update per-key (Algorithm 1 with `Bc=1` for the stats; the IO tiling is the paper's). A fresh max rescales history by `e^(m−m')` before the key folds in.
- **Causal masking** costs what it saves: tiles past a CTA's last row are never loaded, and each row breaks out of the key stream at the diagonal. Watching the loads is the clearest way to see it. Green tiles stream in full, amber diagonal tiles stream and get masked per-row, dark ones never leave HBM:

<p align="center">
  <img src="assets/causal_tiles.gif" width="640" alt="Animated map of which K/V tiles each query tile loads under causal masking, where roughly half never leave HBM">
</p>

- **Logsumexp per row** is written out. The backward pass recomputes `P` from that statistic, so the forward is already backward-shaped.

## Watch the recurrence work

One query row streaming through K/V tiles. The green window is the tile currently in SRAM. The weights renormalize as the running max updates, and the output reaches **exact** (‖error‖ ≈ 1e-7) the moment the last key folds in. The result stays exact throughout; the animation is generated from the same recurrence the kernel runs ([tools/make_convergence_gif.py](tools/make_convergence_gif.py), deterministic).

<p align="center">
  <img src="assets/convergence.gif" width="640" alt="Attention weights revealed tile by tile, with the max-norm error against exact attention dropping to 1e-7 at the end of the stream">
</p>

## Correctness without a GPU\

This was built on a machine with no NVIDIA hardware, so correctness is relative and layered:

1. [src/reference.cpp](src/reference.cpp): naive attention with double accumulation, the ground truth.
2. [src/flash_cpu.cpp](src/flash_cpu.cpp) runs **the flash algorithm on the CPU**, same tiling and the same float precision the kernel uses. Validated against the reference across ragged shapes (N = 33, 257, 300…) in both mask modes: `make test`, errors ~1e-6.
3. [src/flash_fwd.cu](src/flash_fwd.cu) mirrors the validated recurrence; CI compiles it with nvcc on every push, and [tests/test_gpu.cu](tests/test_gpu.cu) checks it against the reference at 5e-5 on real hardware (`make test-gpu`).

So, essentially I did not have to "discover" whether FlashAttention was mathematically correct by running CUDA, since I could prove and test the algorithm on CPU first. So when they later wrote the CUDA kernel, any new failure is much more likely to be a GPU implementation bug, not an algorithm bug.

<p align="center">
  <img src="assets/tests.svg" width="740" alt="Animated terminal: make test running the tiled algorithm against the naive reference across ten shapes, all passing at ~1e-6">
</p>

## PyTorch

The kernel is a drop-in op with the same layout contract as `scaled_dot_product_attention` and explicit heads:

```python
import sys; sys.path.insert(0, "pytorch")
from flash_attn import flash_attention

o = flash_attention(q, k, v, causal=True)   # [batch, heads, seq, head_dim] fp32 CUDA
```

The extension JIT-compiles on first import (needs `nvcc`, a CUDA build of torch, and `pip install ninja`, which torch's extension builder requires), launches on torch's current stream, and validates its whole contract up front. `python3 pytorch/test_parity.py` checks it against a **float64** reference with SDPA alongside as the sanity anchor; add `bench` for a timing table vs SDPA. Inference-only until the backward kernel lands.

## The paper's future work, answered: Triton

Section 5 of the paper names its own biggest limitation: every attention variant needs a new hand-written CUDA kernel, tied to an architecture. It calls for *"writing attention algorithms in a high-level language … and compiling to IO-aware implementations in CUDA — similar to efforts such as Halide."* That compiler now exists in the ecosystem: **Triton**. [triton/flash_attn_triton.py](triton/flash_attn_triton.py) is the same forward pass expressed through it, so this repo holds both sides of the abstraction argument:

| | [src/flash_fwd.cu](src/flash_fwd.cu) (hand CUDA) | [triton/flash_attn_triton.py](triton/flash_attn_triton.py) (compiled) |
|---|---|---|
| softmax stats | per-key, in registers | per-tile (`Algorithm 1` verbatim), `tl.dot` matmuls |
| scheduling | explicit: smem tiles, `__syncthreads`, cooperative loads | the compiler's problem |
| portability | `-arch=sm_XX` | retune two block sizes |
| lines of kernel | ~120 | ~70 |

Same contract, same float64 judge: `python3 triton/test_parity.py` checks the Triton kernel (and the CUDA one alongside, when its extension is importable) against the exact reference on the usual hostile shapes. `allow_tf32` is off so the tolerances mean something; it is the first knob to flip when chasing speed over digits.

The four phases of the recurrence, highlighted in both languages at once. The left pane spends its time on loads, barriers, and register loops; the right pane's equivalent is a masked `tl.load` and a `tl.dot`:

<p align="center">
  <img src="assets/cuda_vs_triton.svg" width="880" alt="Animated side-by-side of the CUDA and Triton kernels: stream the tile, score it, rescale history, fold it in, with matching lines highlighted in sync">
</p>

## Try it on a free GPU

[The Colab notebook](https://colab.research.google.com/github/agupt0318/flash-attention-cuda/blob/main/demo/flash_attention_colab.ipynb) clones the repo on a free T4, runs the CPU suite, compiles and checks the CUDA kernel for whatever GPU it lands on, runs both parity suites, and ends with a CUDA-vs-Triton-vs-SDPA timing plot.

## Numbers (Colab T4, fp32 forward, B=4 H=8 d=64)

From a full notebook run, every test green, then the timing sweep:

| N | cuda (ours) | triton (ours) | torch SDPA | naive (N² in HBM) | naive peak mem |
|---|---|---|---|---|---|
| 512 | 2.115 ms | 4.406 ms | 1.610 ms | **1.121 ms** | 0.09 GiB |
| 1024 | 3.363 ms | 8.646 ms | 2.770 ms | 4.705 ms | 0.29 GiB |
| 2048 | 12.592 ms | 34.036 ms | 11.335 ms | 19.201 ms | 1.07 GiB |
| 4096 | 47.714 ms | 135.868 ms | 45.492 ms | 83.251 ms | **4.13 GiB** |

(flash-family peak mem is 0.02 to 0.13 GiB: the inputs and output.)

- **The algorithm's claim reproduces.** Naive attention (cuBLAS matmuls plus a materialized softmax) *wins below N≈1024*: its matmuls are perfect and the score matrix still fits in cache. That crossover is in the paper. Past it, N² physics takes over: at N=4096 flash is **1.75× faster and uses 32× less memory**, and the naive curve is heading toward an OOM that a tiled implementation avoids entirely.
- **SDPA beats our kernel at every length**, by 1.3× at N=512 and ~5% at N=4096. The context: at fp32 on a T4, SDPA dispatches to its memory-efficient backend, itself a flash-style IO-aware kernel with years of tuning. This is a loss to a production sibling that runs the same algorithm. The gap is being worked. An ILP round (independent partial sums in the dot) bought 16% at N=1024 and nothing at N=4096, which killed the latency-bound hypothesis and pointed at **shared-memory issue rate**: one 4-byte `LDS` per FMA caps the math pipes. `float4` smem reads are the current attempt. The Triton rendering trails ~3× on strict-IEEE dots; autotuning recovered ~10%.

## CPU: attention where there is no GPU

Edge and ARM devices (phones, SBCs like the RK3588) run inference with no CUDA to reach for, so the same online-softmax algorithm runs as a fast CPU kernel in [src/flash_cpu_fast.cpp](src/flash_cpu_fast.cpp). It keeps the exact numerics of the reference and adds three throughput levers: NEON-vectorized inner loops on AArch64 (with a scalar path the compiler auto-vectorizes elsewhere), query-tile blocking so each K/V tile is reused across a block of query rows while it stays cache-hot, and the independent (head, query-tile) units spread across cores.

`make bench-cpu` on an Apple M4 Pro (10 performance cores), B=1 H=8 d=64, fp32, non-causal:

| N | scalar flash | fast | speedup | fast GFLOP/s |
|---|---|---|---|---|
| 512 | 35.4 ms | 3.7 ms | 9.5× | 144 |
| 1024 | 142.7 ms | 14.5 ms | 9.8× | 148 |
| 2048 | 579.1 ms | 43.7 ms | 13.2× | 197 |

Read honestly, the levers split apart. NEON plus cache blocking on a single thread is about 1.5× over the scalar path, because clang already auto-vectorizes the simple loops at `-O2`. The rest of the win is parallelism across the ten cores; the work is independent per row, so it scales close to linearly. `FLASH_CPU_THREADS=N` pins the core count for attribution or for leaving cores free. Correctness is judged the same way as the GPU path, against the double-accum reference over shapes that stress the tile boundaries (`make test-fast`), and the parallel path runs clean under ThreadSanitizer and Address/UB sanitizers.

## Running a real model on-device

The CPU kernel is enough to run a real trained transformer end to end, with no GPU anywhere in the loop. [edge/story.cpp](edge/story.cpp) loads [TinyStories](https://huggingface.co/karpathy/tinyllamas) `stories42M`, a 42M-parameter Llama-architecture model (dim 512, 8 layers, 8 heads, head_dim 64), and generates text. The prompt prefill runs through `attention_flash_fast`; each generated token attends over the KV cache through `attention_step_cpu`. RMSNorm, RoPE, SwiGLU, the matmuls, and the tokenizer all live in that one file, so the binary is self-contained and runs the same on an Orange Pi as on a laptop.

Fetch the weights (downloaded, not committed) and run:

```sh
mkdir -p edge
curl -L -o edge/stories42M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin  # ~167 MB
curl -L -o edge/tokenizer.bin  https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
make story
./build/story -p "Once upon a time" -n 256        # greedy, deterministic
./build/story -p "The robot" -t 0.9 -s 42         # temperature sampling
```

Sample output (greedy): *"Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, yellow flower in the garden. It was a sunflower!..."*

On an Apple M4 Pro, fp32:

| phase | rate | via |
|---|---|---|
| prefill | ~300 tok/s | `attention_flash_fast` |
| decode | ~410 tok/s (2.4 ms/token) | `attention_step_cpu` + pooled matmuls |

The first run also faults in the 167 MB of weights from disk, which shows up as a slower first token.

Building this demo was what showed where the time actually goes. Attention through the KV-cache step is cheap; most of a decode token is the projection and FFN matmuls. Those, and the attention kernel, now run on a shared persistent thread pool ([src/parallel.h](src/parallel.h)) so the decode loop pays no per-call thread-spawn cost. Threading every matmul turned out to be slower than threading none, because the small projections cost more to dispatch than they save; running only the FFN and the vocab-sized logits matmul in parallel took decode from about 290 to about 410 tok/s. The crossover is device-dependent, so `MATVEC_MIN` retunes it (worth doing on an Orange Pi). The next real win is a batched/KV-cache decode kernel, which this harness can check against llama2.c the same way.

Correctness is pinned to the reference: under greedy sampling the generated tokens are byte-identical to [llama2.c](https://github.com/karpathy/llama2.c)'s `run.c` on the same weights, verified across a set of prompts. Greedy decoding is deterministic, so a correct forward pass has to match. The generation path also runs clean under ThreadSanitizer and Address/UB sanitizers.

## Build & run

```sh
make test                  # CPU: tiled algorithm vs reference (no GPU needed)
make test-fast             # CPU: NEON + threaded kernel vs reference (no GPU needed)
make bench-cpu             # CPU: fast kernel vs scalar, ms + speedup + GFLOP/s
make story                 # CPU: run a real TinyStories model (see above for weights)
make cuda                  # compile kernels without running (what CI does)
make test-gpu ARCH=sm_80   # on a CUDA box: kernel vs reference
make bench  ARCH=sm_80     # wall time, TFLOP/s, and the N² traffic avoided
```

`ARCH` defaults to `sm_70`; set it to your GPU (`sm_80` A100, `sm_86` 3090, `sm_89` 4090). The bench prints ms, achieved TFLOP/s, and the HBM traffic a materialized attention matrix would have cost.

## Layout

```
src/
  attention.h      the one API the CPU implementations share
  reference.cpp    naive attention, double accumulation (ground truth)
  flash_cpu.cpp    Algorithm 1 on the CPU, validates the math locally
  reference_bwd.cpp   analytic gradients, double accumulation (ground truth)
  flash_cpu_bwd.cpp   Appendix B on the CPU: logsumexp recompute, D = rowsum(dO.O)
  flash_cpu_fast.cpp  NEON + query-tile blocking + pooled CPU kernel
  parallel.h/.cpp  persistent thread pool (parallel_for) shared by the CPU paths
  flash_gpu.h      host-side contract + CUDA_CHECK
  flash_fwd.cu     the forward kernel + dispatch
  flash_bwd.cu     the backward kernels: dQ pass + dK/dV pass, no atomics
  bench.cu         timing/TFLOPs harness (GPU)
  bench_cpu.cpp    CPU throughput: fast kernel vs scalar
tests/
  test_cpu.cpp      tiled vs reference, shapes chosen to hurt
  test_cpu_bwd.cpp  gradients: reference vs finite differences, tiled vs reference
  test_cpu_fast.cpp fast kernel vs reference, tile-boundary shapes
  test_gpu.cu       fwd and bwd kernels vs reference, same shapes + N=1024
pytorch/
  binding.cpp     torch extension: contract checks + current-stream launch
  flash_attn.py   JIT front door: flash_attention(q, k, v, causal)
  test_parity.py  vs float64 reference, SDPA as sanity anchor (+ bench)
triton/
  flash_attn_triton.py  the same algorithm through a compiler
  test_parity.py        three implementations, one float64 judge
edge/
  story.cpp        a real TinyStories transformer end to end on the CPU kernel
```

## Roadmap

- [x] Run the numbers on real hardware: Colab T4 table above, via the notebook
- [x] Fast CPU kernel (NEON + query-tile blocking + threads) for GPU-less and edge inference
- [x] End-to-end on-device: a real TinyStories transformer through the CPU kernel, verified against llama2.c
- [ ] KV-cache decode kernel: the recompute-free hot path the on-device demo points at next
- [x] Backward pass, CPU-first like the forward: [src/flash_cpu_bwd.cpp](src/flash_cpu_bwd.cpp) proves the algorithm (logsumexp recompute of `P`, `D = rowsum(dO∘O)`) against analytic double-precision gradients grounded in finite differences (~1e-6, `make test`); [src/flash_bwd.cu](src/flash_bwd.cu) is that algorithm on device -- a dQ pass over query rows and a dK/dV pass over key rows, opposite reductions so nothing needs an atomic (`make test-gpu` on real hardware)
- [ ] autograd.Function over the backward kernels, making the PyTorch op trainable
- [ ] Close the gap to SDPA's mem-efficient kernel: >1 query row per thread, vectorized (float4) loads, and occupancy/register tuning. The N=512 1.8× is the target
- [ ] Block size tuning (CUDA and the Triton ~3× gap are a config problem) + head dim 128
- [ ] fp16/bf16 with tensor-core matmuls on Ampere+ (the fp32 kernel is the correctness baseline; fp32 is also all a T4 can show)
- [ ] Block-sparse variant (Section 3.3)

## License

[MIT](LICENSE)
