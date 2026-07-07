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
