#pragma once
#include <vector>
#include "util.h"

// O = softmax(QK^T / sqrt(d)) V, one call per problem.
// Both implementations write O ([batch, heads, seq, d]) and, if lse is
// non-null, the per-row logsumexp L = m + log(l) that a backward pass
// (or a test) can consume.

void attention_reference(const Shape &s, const std::vector<float> &Q,
                         const std::vector<float> &K,
                         const std::vector<float> &V, bool causal,
                         std::vector<float> &O, std::vector<float> *lse);

void attention_flash_cpu(const Shape &s, const std::vector<float> &Q,
                         const std::vector<float> &K,
                         const std::vector<float> &V, bool causal,
                         std::vector<float> &O, std::vector<float> *lse);

// Same algorithm as attention_flash_cpu, tuned for throughput: NEON-
// vectorized inner loops (scalar fallback elsewhere), query-tile cache
// blocking, and the independent work spread across cores. Same numerics,
// same contract.
void attention_flash_fast(const Shape &s, const std::vector<float> &Q,
                          const std::vector<float> &K,
                          const std::vector<float> &V, bool causal,
                          std::vector<float> &O, std::vector<float> *lse);

// Backward pass, naive: analytic gradients of O = softmax(QK^T/sqrt(d))V
// with the whole row materialized and double accumulation throughout.
// Recomputes its own softmax from Q/K (trusts no float forward product).
// Ground truth for the tiled backward, the role reference.cpp plays for
// the forward.
void attention_backward_reference(const Shape &s, const std::vector<float> &Q,
                                  const std::vector<float> &K,
                                  const std::vector<float> &V, bool causal,
                                  const std::vector<float> &dO,
                                  std::vector<float> &dQ,
                                  std::vector<float> &dK,
                                  std::vector<float> &dV);

// Backward pass, the kernel's way (paper Appendix B): consumes the
// forward's O and logsumexp, recomputes P one tile-row at a time as
// exp(s - L), and collapses dP's row coupling into the precomputed
// scalar D = rowsum(dO . O). Nothing NxN ever exists. Float on purpose,
// like attention_flash_cpu: these are the CUDA kernel's numerics.
void attention_flash_cpu_bwd(const Shape &s, const std::vector<float> &Q,
                             const std::vector<float> &K,
                             const std::vector<float> &V, bool causal,
                             const std::vector<float> &O,
                             const std::vector<float> &lse,
                             const std::vector<float> &dO,
                             std::vector<float> &dQ, std::vector<float> &dK,
                             std::vector<float> &dV);

// Single-query decode step: one query attends over a KV cache of n_keys
// past positions. q and out are [n_heads*head_dim]; Kcache/Vcache are
// [n_keys][n_heads*head_dim] (row t, then head, then dim). Same online-
// softmax and NEON inner loops as the prefill kernel; serial (the per-
// step work is tiny, and it is called once per layer per generated token).
void attention_step_cpu(const float *q, const float *Kcache,
                        const float *Vcache, int n_heads, int head_dim,
                        int n_keys, float *out);
