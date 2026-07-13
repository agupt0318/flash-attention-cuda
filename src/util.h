// Shared helpers for the tests and benchmarks: deterministic fills,
// error metrics, and the attention problem shape. Tensors are plain
// float buffers, [batch, heads, seq, head_dim] contiguous.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

struct Shape {
    int batch, heads, seq, d;
    size_t elems() const
    {
        return (size_t)batch * heads * seq * d;
    }
    // flat index of row t of head (b,h)
    size_t row(int b, int h, int t) const
    {
        return (((size_t)b * heads + h) * seq + t) * d;
    }
};

// xorshift64*: deterministic across platforms, unlike rand()
inline float rand_float(uint64_t &state)
{
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    uint64_t r = state * 0x2545F4914F6CDD1DULL;
    // uniform in [-1, 1): 24 random bits / 2^23 lands in [0, 2), shift down
    // by 1. (The old form multiplied by 2 as well and drifted to [-1, 3).)
    return (float)((r >> 40) / 8388608.0 - 1.0);
}

inline void fill_random(std::vector<float> &v, uint64_t seed)
{
    uint64_t s = seed ? seed : 1;
    for (auto &x : v)
        x = rand_float(s);
}

// max |a-b| and max |a-b|/(|b|+eps) over the buffer
struct Error {
    double abs, rel;
};

inline Error max_error(const std::vector<float> &a, const std::vector<float> &b)
{
    Error e{0, 0};
    for (size_t i = 0; i < a.size(); i++) {
        double d = std::fabs((double)a[i] - b[i]);
        // A NaN in the output is a hard failure, but `d > e.abs` is false when
        // d is NaN, so the naive max lets a NaN slip through as a passing zero.
        // Promote any non-finite difference to +inf so the caller's threshold
        // check always trips. (Inf already exceeds the threshold on its own.)
        if (std::isnan(d))
            d = HUGE_VAL;
        if (d > e.abs)
            e.abs = d;
        double r = d / (std::fabs((double)b[i]) + 1e-8);
        if (r > e.rel)
            e.rel = r;
    }
    return e;
}
