# flash-attention-cuda

> [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) implemented from scratch in CUDA — exact attention computed tile-by-tile in on-chip SRAM, so the N×N score matrix **never exists in HBM**.

![ci](https://github.com/agupt0318/flash-attention-cuda/actions/workflows/ci.yml/badge.svg)
![cuda](https://img.shields.io/badge/CUDA-fp32%20forward-76b900)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

<p align="center">
  <img src="assets/tiling.svg" width="880" alt="Animated view of the kernel: a query tile pinned in registers while K/V tiles stream through SRAM; the online-softmax statistics update and the output sharpens tile by tile">
</p>

## The idea

Attention is `O = softmax(QKᵀ/√d)·V`. The standard implementation writes the N×N matrices `S` and `P` to GPU HBM and reads them back — at N=4K that's gigabytes of traffic per head-batch for intermediates nobody keeps. Most of "attention is slow" is that traffic: the operation is memory-bound, and HBM is an order of magnitude slower than on-chip SRAM.

FlashAttention restructures the computation so each K/V tile is loaded to SRAM once and fully consumed. The trick that makes tiling legal is the **online softmax**: softmax over a concatenation decomposes if you carry a running max `m` and normalizer `ℓ`,

```
m' = max(m, s)          — new score s arrives
ℓ' = ℓ·e^(m−m') + e^(s−m')
acc' = acc·e^(m−m') + e^(s−m')·v
```

so the output row is just `acc/ℓ` after the last tile — exact, not approximate, with `O(N)` extra memory instead of `O(N²)`. HBM accesses drop from `Θ(Nd + N²)` to `Θ(N²d²M⁻¹)` (Theorem 2 of the paper).

## The kernel

[src/flash_fwd.cu](src/flash_fwd.cu) — fp32 forward, head dims 32/64, optional causal mask:

- **Grid**: one CTA per (query tile, batch·head); one thread per query row. The row's `q`, running `m`/`ℓ`, and unnormalized accumulator live in **registers** for the entire pass — the only HBM writes are the final `O` row and the logsumexp.
- **K/V stream through shared memory** in 64-row tiles, loaded cooperatively by the whole CTA. Score matrix rows exist one register at a time.
- **Softmax statistics** update per-key (Algorithm 1 with `Bc=1` for the stats; the IO tiling is the paper's) — a fresh max rescales history by `e^(m−m')` before the key folds in.
- **Causal masking** costs what it saves: tiles past a CTA's last row are never loaded, and each row breaks out of the key stream at the diagonal.
- **Logsumexp per row** is written out — the statistic the backward pass recomputes `P` from, so the forward is already backward-shaped.

## Watch the recurrence work

One query row streaming through K/V tiles — the green window is the tile currently in SRAM. The weights renormalize as the running max updates, and the output snaps to **exact** (‖error‖ ≈ 1e-7) the moment the last key folds in. No approximation anywhere; the animation is generated from the same recurrence the kernel runs ([tools/make_convergence_gif.py](tools/make_convergence_gif.py), deterministic).

<p align="center">
  <img src="assets/convergence.gif" width="640" alt="Attention weights revealed tile by tile, with the max-norm error against exact attention dropping to 1e-7 at the end of the stream">
</p>

## Correctness without owning a GPU

This was built on a machine with no NVIDIA hardware, so correctness is layered:

1. [src/reference.cpp](src/reference.cpp) — naive attention, double accumulation: the ground truth.
2. [src/flash_cpu.cpp](src/flash_cpu.cpp) — **the flash algorithm on the CPU**, same tiling, same float precision the kernel uses. Validated against the reference across ragged shapes (N = 33, 257, 300…), both mask modes: `make test`, errors ~1e-6.
3. [src/flash_fwd.cu](src/flash_fwd.cu) mirrors the validated recurrence; CI compiles it with nvcc on every push, and [tests/test_gpu.cu](tests/test_gpu.cu) checks it against the reference at 5e-5 on real hardware (`make test-gpu`).

The algorithm's math never had to be debugged through a device boundary — by the time CUDA entered, only CUDA could be wrong.

<p align="center">
  <img src="assets/tests.svg" width="740" alt="Animated terminal: make test running the tiled algorithm against the naive reference across ten shapes, all passing at ~1e-6">
</p>

## Build & run

```sh
make test                  # CPU: tiled algorithm vs reference (no GPU needed)
make cuda                  # compile kernels without running (what CI does)
make test-gpu ARCH=sm_80   # on a CUDA box: kernel vs reference
make bench  ARCH=sm_80     # wall time, TFLOP/s, and the N² traffic avoided
```

`ARCH` defaults to `sm_70`; set it to your GPU (`sm_80` A100, `sm_86` 3090, `sm_89` 4090). GPU numbers pending a hardware run — the bench prints ms, achieved TFLOP/s, and the HBM traffic a materialized attention matrix would have cost.

## Layout

```
src/
  attention.h     the one API both CPU implementations share
  reference.cpp   naive attention, double accumulation (ground truth)
  flash_cpu.cpp   Algorithm 1 on the CPU — validates the math locally
  flash_gpu.h     host-side contract + CUDA_CHECK
  flash_fwd.cu    the kernel + dispatch
  bench.cu        timing/TFLOPs harness
tests/
  test_cpu.cpp    tiled vs reference, shapes chosen to hurt
  test_gpu.cu     kernel vs reference, same shapes + N=1024
```

## Roadmap

- [ ] Run the numbers on real hardware (A100/4090) and publish them here
- [ ] Backward pass — recompute `P` from `O` + logsumexp per the paper's Appendix B
- [ ] fp16/bf16 with tensor-core matmuls (the fp32 kernel is the correctness baseline)
- [ ] Block size tuning + head dim 128
- [ ] Block-sparse variant (Section 3.3)

## License

[MIT](LICENSE)
