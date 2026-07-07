# flash-attention-cuda

A from-scratch CUDA implementation of [FlashAttention](https://arxiv.org/abs/2205.14135)
(Dao et al., 2022) — exact attention computed tile-by-tile in on-chip SRAM,
never materializing the N×N attention matrix in HBM.

Work in progress; see the commit history for the build-up.
