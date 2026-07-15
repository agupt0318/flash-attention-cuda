// The FlashAttention backward kernels: the CPU-validated algorithm in
// src/flash_cpu_bwd.cpp (paper Appendix B), split so nothing needs an
// atomic. dQ rows depend on a sum over keys, dK/dV rows on a sum over
// queries -- opposite reductions -- so one kernel owns query rows and
// streams K/V tiles (like the forward), and a second owns key rows and
// streams Q/dO tiles. P is recomputed per score from the forward's
// logsumexp, p = exp(q.k * scale - L_t): no second softmax, no N x N
// anything. D_t = rowsum(dO_t o O_t) is precomputed by a trivial kernel
// since the dK/dV pass needs every row's value.
//
// Threads use the forward's pair split (half of each register vector
// per thread, dot halves joined by one warp shuffle) -- the backward
// carries 3-4 half-vectors per thread, so the register relief matters
// even more here. Correctness-first: __launch_bounds__ asks for 2 CTAs
// per SM, not the forward's 4; occupancy tuning is roadmap work.
//
// Causal masking is a predicate, as in the forward: a masked score is
// -1e30, so p underflows to exactly 0 and the pair's dK/dV/dQ
// contributions vanish, keeping the shuffle in lockstep. Tiles wholly
// on the wrong side of the diagonal are never streamed at all.

#include <math.h>
#include "flash_gpu.h"

#define BLOCK_M  64     // rows a CTA owns (2 threads each)
#define BLOCK_N  32     // rows staged in shared memory at a time
#define NTHREADS (BLOCK_M * 2)

// D_t = dO_t . O_t, one thread per row
__global__ void bwd_dot_kernel(const float *__restrict__ dO,
                               const float *__restrict__ O,
                               float *__restrict__ D, int seq, int d)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= seq)
        return;
    const size_t r = ((size_t)blockIdx.y * seq + t) * d;
    float acc = 0.0f;
    for (int c = 0; c < d; c++)
        acc += dO[r + c] * O[r + c];
    D[(size_t)blockIdx.y * seq + t] = acc;
}

// dQ_t = sum_j ds_tj k_j * scale: one pair of threads per query row,
// K/V tiles streaming through shared memory, exactly the forward's IO.
template <int HEAD_DIM, bool CAUSAL>
__global__ void __launch_bounds__(NTHREADS, 2)
flash_bwd_dq_kernel(const float *__restrict__ Q, const float *__restrict__ K,
                    const float *__restrict__ V,
                    const float *__restrict__ lse,
                    const float *__restrict__ Dv,
                    const float *__restrict__ dO, float *__restrict__ dQ,
                    int seq, float scale)
{
    constexpr int HALF = HEAD_DIM / 2;
    __shared__ __align__(16) float Ks[BLOCK_N][HEAD_DIM];
    __shared__ __align__(16) float Vs[BLOCK_N][HEAD_DIM];

    const int row = threadIdx.x >> 1;
    const int c0 = (threadIdx.x & 1) * HALF;
    const int q0 = blockIdx.x * BLOCK_M;
    const int t = q0 + row;
    const bool live = t < seq;
    const size_t base = (size_t)blockIdx.y * seq * HEAD_DIM;

    float q[HALF], dOt[HALF], dq[HALF];
    const float L = live ? lse[(size_t)blockIdx.y * seq + t] : 0.0f;
    const float D = live ? Dv[(size_t)blockIdx.y * seq + t] : 0.0f;
#pragma unroll
    for (int c = 0; c < HALF; c++) {
        q[c] = live ? Q[base + (size_t)t * HEAD_DIM + c0 + c] : 0.0f;
        dOt[c] = live ? dO[base + (size_t)t * HEAD_DIM + c0 + c] : 0.0f;
        dq[c] = 0.0f;
    }

    const int kv_end = CAUSAL ? min(seq, q0 + BLOCK_M) : seq;
    for (int j0 = 0; j0 < kv_end; j0 += BLOCK_N) {
        const int jn = min(BLOCK_N, seq - j0);

        {   // float4-wide cooperative load, as in the forward
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
            // two half-dots, joined by one shuffle each
            float sdot = 0.0f, pdot = 0.0f;
#pragma unroll
            for (int c = 0; c < HALF; c++) {
                sdot += q[c] * Ks[j][c0 + c];
                pdot += dOt[c] * Vs[j][c0 + c];
            }
            float s = (sdot + __shfl_xor_sync(0xffffffffu, sdot, 1)) * scale;
            const float dp = pdot + __shfl_xor_sync(0xffffffffu, pdot, 1);
            if (CAUSAL && j0 + j > t)
                s = -1e30f;                     // p underflows to 0
            const float p = __expf(s - L);
            const float ds = p * (dp - D) * scale;
#pragma unroll
            for (int c = 0; c < HALF; c++)
                dq[c] += ds * Ks[j][c0 + c];
        }
        __syncthreads();
    }

    if (live) {
#pragma unroll
        for (int c = 0; c < HALF; c++)
            dQ[base + (size_t)t * HEAD_DIM + c0 + c] = dq[c];
    }
}

// dK_j = sum_t ds_tj q_t * scale, dV_j = sum_t p_tj dO_t: one pair of
// threads per KEY row, Q/dO tiles (plus their L and D rows) streaming.
template <int HEAD_DIM, bool CAUSAL>
__global__ void __launch_bounds__(NTHREADS, 2)
flash_bwd_dkv_kernel(const float *__restrict__ Q, const float *__restrict__ K,
                     const float *__restrict__ V,
                     const float *__restrict__ lse,
                     const float *__restrict__ Dv,
                     const float *__restrict__ dO, float *__restrict__ dK,
                     float *__restrict__ dV, int seq, float scale)
{
    constexpr int HALF = HEAD_DIM / 2;
    __shared__ __align__(16) float Qs[BLOCK_N][HEAD_DIM];
    __shared__ __align__(16) float dOs[BLOCK_N][HEAD_DIM];
    __shared__ float Ls[BLOCK_N], Ds[BLOCK_N];

    const int row = threadIdx.x >> 1;
    const int c0 = (threadIdx.x & 1) * HALF;
    const int k0 = blockIdx.x * BLOCK_M;
    const int j = k0 + row;
    const bool live = j < seq;
    const size_t base = (size_t)blockIdx.y * seq * HEAD_DIM;
    const size_t rows = (size_t)blockIdx.y * seq;

    float k[HALF], v[HALF], dk[HALF], dv[HALF];
#pragma unroll
    for (int c = 0; c < HALF; c++) {
        k[c] = live ? K[base + (size_t)j * HEAD_DIM + c0 + c] : 0.0f;
        v[c] = live ? V[base + (size_t)j * HEAD_DIM + c0 + c] : 0.0f;
        dk[c] = 0.0f;
        dv[c] = 0.0f;
    }

    // causal: queries before this CTA's first key row see none of it
    const int t_start = CAUSAL ? (k0 / BLOCK_N) * BLOCK_N : 0;
    for (int t0 = t_start; t0 < seq; t0 += BLOCK_N) {
        const int tn = min(BLOCK_N, seq - t0);

        {
            const float4 *Q4 = (const float4 *)(Q + base +
                                                (size_t)t0 * HEAD_DIM);
            const float4 *dO4 = (const float4 *)(dO + base +
                                                 (size_t)t0 * HEAD_DIM);
            float4 *Qs4 = (float4 *)&Qs[0][0];
            float4 *dOs4 = (float4 *)&dOs[0][0];
            for (int i = threadIdx.x; i < tn * HEAD_DIM / 4; i += NTHREADS) {
                Qs4[i] = Q4[i];
                dOs4[i] = dO4[i];
            }
            for (int i = threadIdx.x; i < tn; i += NTHREADS) {
                Ls[i] = lse[rows + t0 + i];
                Ds[i] = Dv[rows + t0 + i];
            }
        }
        __syncthreads();

#pragma unroll 4
        for (int i = 0; i < tn; i++) {
            float sdot = 0.0f, pdot = 0.0f;
#pragma unroll
            for (int c = 0; c < HALF; c++) {
                sdot += Qs[i][c0 + c] * k[c];
                pdot += dOs[i][c0 + c] * v[c];
            }
            float s = (sdot + __shfl_xor_sync(0xffffffffu, sdot, 1)) * scale;
            const float dp = pdot + __shfl_xor_sync(0xffffffffu, pdot, 1);
            if (CAUSAL && t0 + i < j)
                s = -1e30f;                     // query above the diagonal
            const float p = __expf(s - Ls[i]);
            const float ds = p * (dp - Ds[i]) * scale;
#pragma unroll
            for (int c = 0; c < HALF; c++) {
                dk[c] += ds * Qs[i][c0 + c];
                dv[c] += p * dOs[i][c0 + c];
            }
        }
        __syncthreads();
    }

    if (live) {
#pragma unroll
        for (int c = 0; c < HALF; c++) {
            dK[base + (size_t)j * HEAD_DIM + c0 + c] = dk[c];
            dV[base + (size_t)j * HEAD_DIM + c0 + c] = dv[c];
        }
    }
}

template <int HEAD_DIM>
static void launch_bwd(dim3 grid, cudaStream_t stream, const float *Q,
                       const float *K, const float *V, const float *lse,
                       const float *D, const float *dO, bool causal,
                       float *dQ, float *dK, float *dV, int seq, float scale)
{
    if (causal) {
        flash_bwd_dq_kernel<HEAD_DIM, true><<<grid, NTHREADS, 0, stream>>>(
            Q, K, V, lse, D, dO, dQ, seq, scale);
        flash_bwd_dkv_kernel<HEAD_DIM, true><<<grid, NTHREADS, 0, stream>>>(
            Q, K, V, lse, D, dO, dK, dV, seq, scale);
    } else {
        flash_bwd_dq_kernel<HEAD_DIM, false><<<grid, NTHREADS, 0, stream>>>(
            Q, K, V, lse, D, dO, dQ, seq, scale);
        flash_bwd_dkv_kernel<HEAD_DIM, false><<<grid, NTHREADS, 0, stream>>>(
            Q, K, V, lse, D, dO, dK, dV, seq, scale);
    }
}

void flash_backward(int batch, int heads, int seq, int d, const float *Q,
                    const float *K, const float *V, const float *O,
                    const float *lse, const float *dO, bool causal,
                    float *dQ, float *dK, float *dV, cudaStream_t stream)
{
    const size_t rows = (size_t)batch * heads * seq;
    const float scale = 1.0f / sqrtf((float)d);

    // scratch for D = rowsum(dO o O). cudaFree synchronizes the device,
    // which is fine for the test/bench callers; an autograd.Function
    // will hold this buffer through torch's allocator instead.
    float *D;
    CUDA_CHECK(cudaMalloc(&D, rows * sizeof(float)));
    dim3 dgrid((seq + 127) / 128, batch * heads);
    bwd_dot_kernel<<<dgrid, 128, 0, stream>>>(dO, O, D, seq, d);

    dim3 grid((seq + BLOCK_M - 1) / BLOCK_M, batch * heads);
    switch (d) {
    case 32:
        launch_bwd<32>(grid, stream, Q, K, V, lse, D, dO, causal,
                       dQ, dK, dV, seq, scale);
        break;
    case 64:
        launch_bwd<64>(grid, stream, Q, K, V, lse, D, dO, causal,
                       dQ, dK, dV, seq, scale);
        break;
    default:
        fprintf(stderr, "flash_backward: unsupported head dim %d\n", d);
        exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(D));
}
