// The FlashAttention forward kernel.
//
// IO structure per the paper: K and V stream through shared memory in
// BLOCK_N-row tiles; the N x N score matrix never exists anywhere. One
// CTA owns a BLOCK_M-row tile of queries, one thread per query row,
// with that row's q, running max m, normalizer l, and unnormalized
// output accumulator living in registers for the whole pass.
//
// The softmax statistics use the online update at per-key granularity
// (Algorithm 1 with Bc = 1 for the stats; the IO tiling is unchanged):
// a new max rescales history by exp(m_old - m_new), then the key folds
// in. Same recurrence the CPU model validated against the reference.

#include <math.h>
#include "flash_gpu.h"

#define BLOCK_M 128     // query rows per CTA == threads per CTA
#define BLOCK_N 64      // K/V rows staged in shared memory at a time

template <int HEAD_DIM>
__global__ void flash_fwd_kernel(const float *__restrict__ Q,
                                 const float *__restrict__ K,
                                 const float *__restrict__ V,
                                 float *__restrict__ O,
                                 float *__restrict__ lse, int seq,
                                 float scale, int causal)
{
    __shared__ __align__(16) float Ks[BLOCK_N][HEAD_DIM];
    __shared__ __align__(16) float Vs[BLOCK_N][HEAD_DIM];

    const int q0 = blockIdx.x * BLOCK_M;
    const int t = q0 + threadIdx.x;             // this thread's query row
    const bool live = t < seq;                  // dead threads still load
    const size_t base = (size_t)blockIdx.y * seq * HEAD_DIM;

    float q[HEAD_DIM], acc[HEAD_DIM];
    float m = -1e30f, l = 0.0f;
    if (live) {
#pragma unroll
        for (int c = 0; c < HEAD_DIM; c++) {
            q[c] = Q[base + (size_t)t * HEAD_DIM + c];
            acc[c] = 0.0f;
        }
    }

    // causal: no key tile past this CTA's last query row is ever needed
    const int kv_end = causal ? min(seq, q0 + BLOCK_M) : seq;
    for (int j0 = 0; j0 < kv_end; j0 += BLOCK_N) {
        const int jn = min(BLOCK_N, seq - j0);

        // float4-wide cooperative load: rows are 128/256 B so tiles are
        // 16 B-aligned end to end; 4x fewer load instructions than the
        // scalar loop and the same coalescing.
        {
            const float4 *K4 = (const float4 *)(K + base +
                                                (size_t)j0 * HEAD_DIM);
            const float4 *V4 = (const float4 *)(V + base +
                                                (size_t)j0 * HEAD_DIM);
            float4 *Ks4 = (float4 *)&Ks[0][0];
            float4 *Vs4 = (float4 *)&Vs[0][0];
            for (int i = threadIdx.x; i < jn * HEAD_DIM / 4;
                 i += blockDim.x) {
                Ks4[i] = K4[i];
                Vs4[i] = V4[i];
            }
        }
        __syncthreads();

        if (live) {
            for (int j = 0; j < jn; j++) {
                if (causal && j0 + j > t)
                    break;                      // keys are ordered: done
                // Four independent partial sums: a single accumulator
                // makes the dot a serial chain of dependent FMAs (~4
                // cycles each on Turing), and nvcc won't reassociate
                // floats on its own — the chain gates the whole kernel.
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma unroll
                for (int c = 0; c < HEAD_DIM; c += 4) {
                    s0 += q[c + 0] * Ks[j][c + 0];
                    s1 += q[c + 1] * Ks[j][c + 1];
                    s2 += q[c + 2] * Ks[j][c + 2];
                    s3 += q[c + 3] * Ks[j][c + 3];
                }
                float s = ((s0 + s1) + (s2 + s3)) * scale;

                if (s > m) {                    // rescale history
                    const float corr = __expf(m - s);
                    l *= corr;
#pragma unroll
                    for (int c = 0; c < HEAD_DIM; c++)
                        acc[c] *= corr;
                    m = s;
                }
                const float e = __expf(s - m);
                l += e;
#pragma unroll
                for (int c = 0; c < HEAD_DIM; c++)
                    acc[c] += e * Vs[j][c];
            }
        }
        __syncthreads();
    }

    if (live) {
#pragma unroll
        for (int c = 0; c < HEAD_DIM; c++)
            O[base + (size_t)t * HEAD_DIM + c] = acc[c] / l;
        if (lse)
            lse[(size_t)blockIdx.y * seq + t] = m + logf(l);
    }
}

// stream: where to enqueue the launch — callers embedded in a runtime
// (e.g. the PyTorch binding) pass their current stream so ordering with
// surrounding ops holds; standalone callers let it default to stream 0.
void flash_forward(int batch, int heads, int seq, int d, const float *Q,
                   const float *K, const float *V, bool causal, float *O,
                   float *lse, cudaStream_t stream)
{
    dim3 grid((seq + BLOCK_M - 1) / BLOCK_M, batch * heads);
    const float scale = 1.0f / sqrtf((float)d);

    switch (d) {
    case 32:
        flash_fwd_kernel<32><<<grid, BLOCK_M, 0, stream>>>(Q, K, V, O, lse,
                                                           seq, scale, causal);
        break;
    case 64:
        flash_fwd_kernel<64><<<grid, BLOCK_M, 0, stream>>>(Q, K, V, O, lse,
                                                           seq, scale, causal);
        break;
    default:
        fprintf(stderr, "flash_forward: unsupported head dim %d\n", d);
        exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}
