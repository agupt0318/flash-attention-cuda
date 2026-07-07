// Host-side API for the CUDA kernels. All pointers are device memory,
// [batch, heads, seq, d] contiguous; lse (logsumexp per row, for a
// future backward pass) may be null.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                  \
                    cudaGetErrorString(err_), __FILE__, __LINE__);        \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

void flash_forward(int batch, int heads, int seq, int d, const float *Q,
                   const float *K, const float *V, bool causal, float *O,
                   float *lse, cudaStream_t stream = nullptr);
