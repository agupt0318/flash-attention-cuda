// Forward-pass benchmark: wall time, effective TFLOP/s, and what the
// same problem would have cost in HBM traffic had the N x N attention
// matrix been materialized (the number FlashAttention exists to avoid).

#include <cstdio>
#include <vector>
#include "util.h"
#include "flash_gpu.h"

static void bench(int batch, int heads, int seq, int d, bool causal)
{
    Shape s{batch, heads, seq, d};
    size_t bytes = s.elems() * sizeof(float);
    std::vector<float> host(s.elems());
    fill_random(host, 42);

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, bytes));
    CUDA_CHECK(cudaMalloc(&dK, bytes));
    CUDA_CHECK(cudaMalloc(&dV, bytes));
    CUDA_CHECK(cudaMalloc(&dO, bytes));
    CUDA_CHECK(cudaMemcpy(dQ, host.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, host.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, host.data(), bytes, cudaMemcpyHostToDevice));

    const int iters = 30;
    flash_forward(batch, heads, seq, d, dQ, dK, dV, causal, dO, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());        // warm up + fault in

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++)
        flash_forward(batch, heads, seq, d, dQ, dK, dV, causal, dO, nullptr);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= iters;

    // 2 matmuls, 2 flops/MAC; causal touches half the score matrix
    double flops = 4.0 * batch * heads * (double)seq * seq * d;
    if (causal)
        flops /= 2;
    double n2_gb = (double)batch * heads * seq * (double)seq * 4 * 2 / 1e9;
    printf("B=%-2d H=%-2d N=%-5d d=%-3d %s  %8.3f ms  %6.2f TFLOP/s"
           "  (skipped N^2 traffic: ~%.1f GB)\n",
           batch, heads, seq, d, causal ? "causal" : "full  ", ms,
           flops / (ms * 1e9), n2_gb);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

int main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s\n", prop.name);
    for (int seq : { 512, 1024, 2048, 4096 }) {
        bench(4, 8, seq, 64, false);
        bench(4, 8, seq, 64, true);
    }
    return 0;
}
