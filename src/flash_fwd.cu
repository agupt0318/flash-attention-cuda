// The FlashAttention forward kernel.
//
// IO structure per the paper: K and V stream through shared memory in
// BLOCK_N-row tiles; the N x N score matrix never exists anywhere. One
// CTA owns a BLOCK_M-row tile of queries, and each row is split across
// a PAIR of threads — each thread carries half the row's q and half its
// accumulator in registers, and the pair joins dot-product halves with
// one warp shuffle per key.
//
// Why pairs: the one-thread-per-row layout cost 167 registers, capping
// the SM at 12 warps, and Nsight showed schedulers finding an eligible
// warp only 31% of cycles (1.76 active warps/scheduler). Halving the
// per-thread state (~100 regs) buys ~16 warps/SM plus twice the CTAs
// per problem — latency hiding, which the profile said was the wall.
//
// The softmax statistics use the online update at per-key granularity
// (Algorithm 1 with Bc = 1 for the stats; the IO tiling is unchanged):
// a new max rescales history by exp(m_old - m_new), then the key folds
// in. Both threads of a pair track m and l redundantly — cheaper than
// communicating them. Causal masking is a predicate, not a break: the
// shuffle requires pairs to execute in lockstep, so the loop bound
// stays uniform and masked keys contribute e = 0.

#include <math.h>
#include "flash_gpu.h"

#define BLOCK_M  64     // query rows per CTA (2 threads each)
#define BLOCK_N  32     // K/V rows staged in shared memory at a time
#define NTHREADS (BLOCK_M * 2)

template <int HEAD_DIM, bool CAUSAL>
__global__ void __launch_bounds__(NTHREADS, 4)
flash_fwd_kernel(const float *__restrict__ Q, const float *__restrict__ K,
                 const float *__restrict__ V, float *__restrict__ O,
                 float *__restrict__ lse, int seq, float scale)
{
    constexpr int HALF = HEAD_DIM / 2;
    __shared__ __align__(16) float Ks[BLOCK_N][HEAD_DIM];
    __shared__ __align__(16) float Vs[BLOCK_N][HEAD_DIM];

    const int row = threadIdx.x >> 1;
    const int c0 = (threadIdx.x & 1) * HALF;    // this thread's half of d
    const int q0 = blockIdx.x * BLOCK_M;
    const int t = q0 + row;
    const bool live = t < seq;      // dead rows compute (lockstep for the
    const size_t base = (size_t)blockIdx.y * seq * HEAD_DIM;   // shuffle),
                                                // but never load or store
    float q[HALF], acc[HALF];
    float m = -1e30f, l = 0.0f;
#pragma unroll
    for (int c = 0; c < HALF; c++) {
        q[c] = live ? Q[base + (size_t)t * HEAD_DIM + c0 + c] : 0.0f;
        acc[c] = 0.0f;
    }

    // causal: no key tile past this CTA's last query row is ever needed
    const int kv_end = CAUSAL ? min(seq, q0 + BLOCK_M) : seq;
    for (int j0 = 0; j0 < kv_end; j0 += BLOCK_N) {
        const int jn = min(BLOCK_N, seq - j0);

        {   // float4-wide cooperative load (tiles are 16 B-aligned)
            const float4 *K4 = (const float4 *)(K + base +
                                                (size_t)j0 * HEAD_DIM);
            const float4 *V4 = (const float4 *)(V + base +
                                                (size_t)j0 * HEAD_DIM);
            float4 *Ks4 = (float4 *)&Ks[0][0];
            float4 *Vs4 = (float4 *)&Vs[0][0];
            for (int i = threadIdx.x; i < jn * HEAD_DIM / 4; i += NTHREADS) {
                Ks4[i] = K4[i];
                Vs4[i] = V4[i];
            }
        }
        __syncthreads();

#pragma unroll 4
        for (int j = 0; j < jn; j++) {
            // half-dot in four independent partial sums, float4 smem reads
            const float4 *k4 = (const float4 *)&Ks[j][c0];
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma unroll
            for (int c = 0; c < HALF / 4; c++) {
                const float4 kk = k4[c];
                s0 += q[4 * c + 0] * kk.x;
                s1 += q[4 * c + 1] * kk.y;
                s2 += q[4 * c + 2] * kk.z;
                s3 += q[4 * c + 3] * kk.w;
            }
            const float part = (s0 + s1) + (s2 + s3);
            float s = (part + __shfl_xor_sync(0xffffffffu, part, 1)) * scale;
            if (CAUSAL && j0 + j > t)
                s = -1e30f;                     // masked: never raises m,
                                                // and contributes e = 0
            if (s > m) {                        // rescale history
                const float corr = __expf(m - s);
                l *= corr;
#pragma unroll
                for (int c = 0; c < HALF; c++)
                    acc[c] *= corr;
                m = s;
            }
            const float e = __expf(s - m);
            l += e;
            const float4 *v4 = (const float4 *)&Vs[j][c0];
#pragma unroll
            for (int c = 0; c < HALF / 4; c++) {
                const float4 vv = v4[c];
                acc[4 * c + 0] += e * vv.x;
                acc[4 * c + 1] += e * vv.y;
                acc[4 * c + 2] += e * vv.z;
                acc[4 * c + 3] += e * vv.w;
            }
        }
        __syncthreads();
    }

    if (live) {
        const float inv_l = 1.0f / l;           // one division, not HALF
#pragma unroll
        for (int c = 0; c < HALF; c++)
            O[base + (size_t)t * HEAD_DIM + c0 + c] = acc[c] * inv_l;
        if (lse && c0 == 0)
            lse[(size_t)blockIdx.y * seq + t] = m + logf(l);
    }
}

template <int HEAD_DIM>
static void launch(dim3 grid, cudaStream_t stream, const float *Q,
                   const float *K, const float *V, bool causal, float *O,
                   float *lse, int seq, float scale)
{
    if (causal)
        flash_fwd_kernel<HEAD_DIM, true><<<grid, NTHREADS, 0, stream>>>(
            Q, K, V, O, lse, seq, scale);
    else
        flash_fwd_kernel<HEAD_DIM, false><<<grid, NTHREADS, 0, stream>>>(
            Q, K, V, O, lse, seq, scale);
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
        launch<32>(grid, stream, Q, K, V, causal, O, lse, seq, scale);
        break;
    case 64:
        launch<64>(grid, stream, Q, K, V, causal, O, lse, seq, scale);
        break;
    default:
        fprintf(stderr, "flash_forward: unsupported head dim %d\n", d);
        exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}
