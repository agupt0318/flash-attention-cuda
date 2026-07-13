// CPU throughput: the fast kernel (NEON + blocking + threads) against the
// scalar reference flash implementation, on the same shapes and inputs.
// Both compute exact attention; this measures how much the three levers
// buy on a machine with no GPU. FLOP count is the two matmuls
// (QK^T and PV): 4 * B*H*N*N*d, halved for causal.

#include <chrono>
#include <cstdio>
#include <vector>
#include "attention.h"

using clk = std::chrono::steady_clock;

static double time_ms(void (*fn)(const Shape &, const std::vector<float> &,
                                 const std::vector<float> &,
                                 const std::vector<float> &, bool,
                                 std::vector<float> &, std::vector<float> *),
                      const Shape &s, const std::vector<float> &Q,
                      const std::vector<float> &K, const std::vector<float> &V,
                      bool causal, std::vector<float> &O, int iters)
{
    fn(s, Q, K, V, causal, O, nullptr);         // warm up
    auto t0 = clk::now();
    for (int i = 0; i < iters; i++)
        fn(s, Q, K, V, causal, O, nullptr);
    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main()
{
    const bool causal = false;
    const int iters = 20;
    printf("B=1 H=8 d=64, non-causal, %d iters\n\n", iters);
    printf("%-6s %12s %12s %9s %10s\n",
           "N", "scalar ms", "fast ms", "speedup", "fast GFLOP/s");

    for (int N : { 128, 256, 512, 1024, 2048 }) {
        Shape s{ 1, 8, N, 64 };
        std::vector<float> Q(s.elems()), K(s.elems()), V(s.elems()), O(s.elems());
        fill_random(Q, 1); fill_random(K, 2); fill_random(V, 3);

        double slow = time_ms(attention_flash_cpu, s, Q, K, V, causal, O, iters);
        double fast = time_ms(attention_flash_fast, s, Q, K, V, causal, O, iters);

        double flops = 4.0 * s.batch * s.heads * (double)N * N * s.d;
        if (causal) flops *= 0.5;
        double gflops = flops / (fast * 1e-3) / 1e9;
        printf("%-6d %12.3f %12.3f %8.1fx %10.1f\n",
               N, slow, fast, slow / fast, gflops);
    }
    return 0;
}
