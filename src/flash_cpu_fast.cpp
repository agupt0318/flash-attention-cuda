// Fast CPU FlashAttention: the same online-softmax algorithm as
// flash_cpu.cpp, rebuilt to actually be quick on a CPU with no GPU in
// sight. Three levers:
//
//   1. NEON (AArch64) vectorizes the two per-head-dim inner loops that
//      dominate the work: the q.k dot product and the weighted-V
//      accumulate. head_dim (32/64) is a multiple of 4, so the vector
//      path has no ragged tail on the shapes this repo supports.
//   2. Query-tile blocking keeps a small block of query rows resident
//      while each K/V tile streams past once, so a loaded K/V tile is
//      reused across the whole block instead of being reread per row.
//   3. The independent (head, query-tile) units are spread across cores.
//
// A portable scalar path keeps it building and correct on x86 CI, where
// the compiler still auto-vectorizes the same loops.
//
// Why this exists: attention on edge and ARM hardware (phones, SBCs like
// the RK3588) has no CUDA to reach for. This is a from-scratch, oracle-
// verified answer to "how do you run attention fast where there is no
// GPU," and it turns the repo's CPU path from a correctness reference
// into a usable kernel.

#include <cmath>
#include <vector>
#include <algorithm>
#include "attention.h"
#include "parallel.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace {

constexpr int QBLK = 8;     // query rows kept resident per tile
constexpr int KBLK = 64;    // K/V rows streamed per tile (stays cache-hot)

// sum_c q[c] * k[c], four independent accumulators for instruction-level
// parallelism through the FMA latency.
inline float dotf(const float *q, const float *k, int d)
{
#if defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    int c = 0;
    for (; c + 16 <= d; c += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(q + c),      vld1q_f32(k + c));
        a1 = vfmaq_f32(a1, vld1q_f32(q + c + 4),  vld1q_f32(k + c + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(q + c + 8),  vld1q_f32(k + c + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(q + c + 12), vld1q_f32(k + c + 12));
    }
    for (; c + 4 <= d; c += 4)
        a0 = vfmaq_f32(a0, vld1q_f32(q + c), vld1q_f32(k + c));
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; c < d; c++)
        s += q[c] * k[c];
    return s;
#else
    float s = 0;
    for (int c = 0; c < d; c++)
        s += q[c] * k[c];
    return s;
#endif
}

// acc[c] += e * v[c]
inline void axpy(float *acc, const float *v, float e, int d)
{
#if defined(__ARM_NEON)
    int c = 0;
    for (; c + 4 <= d; c += 4)
        vst1q_f32(acc + c, vfmaq_n_f32(vld1q_f32(acc + c), vld1q_f32(v + c), e));
    for (; c < d; c++)
        acc[c] += e * v[c];
#else
    for (int c = 0; c < d; c++)
        acc[c] += e * v[c];
#endif
}

// acc[c] *= f
inline void scal(float *acc, float f, int d)
{
#if defined(__ARM_NEON)
    const float32x4_t fv = vdupq_n_f32(f);
    int c = 0;
    for (; c + 4 <= d; c += 4)
        vst1q_f32(acc + c, vmulq_f32(vld1q_f32(acc + c), fv));
    for (; c < d; c++)
        acc[c] *= f;
#else
    for (int c = 0; c < d; c++)
        acc[c] *= f;
#endif
}

// One (head, query-tile) unit: rows [r0, r1) of head (Qh/Kh/Vh point at
// this head's row 0). Streams each needed K/V tile once, updating the
// per-row online-softmax state for every row in the block against it.
void process_tile(const float *Qh, const float *Kh, const float *Vh,
                  float *Oh, float *Lh, int seq, int d, float scale,
                  bool causal, int r0, int r1)
{
    const int nq = r1 - r0;
    float m[QBLK], l[QBLK];
    float acc[QBLK * 64];               // 64 = max supported head_dim

    for (int i = 0; i < nq; i++) {
        m[i] = -1e30f;
        l[i] = 0.0f;
        for (int c = 0; c < d; c++)
            acc[i * d + c] = 0.0f;
    }

    // Causal: the block never needs a key past its own last row (r1-1).
    const int kv_end = causal ? r1 : seq;
    for (int j0 = 0; j0 < kv_end; j0 += KBLK) {
        const int jn = std::min(KBLK, seq - j0);
        for (int i = 0; i < nq; i++) {
            const int row = r0 + i;
            const float *q = Qh + (size_t)row * d;
            float *acci = acc + i * d;
            for (int j = 0; j < jn; j++) {
                const int key = j0 + j;
                if (causal && key > row)
                    break;              // keys ascend; the rest are masked too
                const float *k = Kh + (size_t)key * d;
                const float s = dotf(q, k, d) * scale;
                if (s > m[i]) {         // new max: rescale this row's history
                    const float corr = expf(m[i] - s);
                    l[i] *= corr;
                    scal(acci, corr, d);
                    m[i] = s;
                }
                const float e = expf(s - m[i]);
                l[i] += e;
                axpy(acci, Vh + (size_t)key * d, e, d);
            }
        }
    }

    for (int i = 0; i < nq; i++) {
        const int row = r0 + i;
        const float inv = 1.0f / l[i];
        float *o = Oh + (size_t)row * d;
        for (int c = 0; c < d; c++)
            o[c] = acc[i * d + c] * inv;
        if (Lh)
            Lh[row] = m[i] + logf(l[i]);
    }
}

}   // namespace

// Single-query decode step (see attention.h). The KV cache row stride is
// n_heads*head_dim; each head reads its own slice.
void attention_step_cpu(const float *q, const float *Kcache,
                        const float *Vcache, int n_heads, int head_dim,
                        int n_keys, float *out)
{
    const int D = n_heads * head_dim;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    for (int h = 0; h < n_heads; h++) {
        const float *qh = q + h * head_dim;
        float *o = out + h * head_dim;
        float m = -1e30f, l = 0.0f;
        for (int c = 0; c < head_dim; c++)
            o[c] = 0.0f;
        for (int t = 0; t < n_keys; t++) {
            const float *k = Kcache + (size_t)t * D + h * head_dim;
            const float s = dotf(qh, k, head_dim) * scale;
            if (s > m) {
                const float corr = expf(m - s);
                l *= corr;
                scal(o, corr, head_dim);
                m = s;
            }
            const float e = expf(s - m);
            l += e;
            axpy(o, Vcache + (size_t)t * D + h * head_dim, e, head_dim);
        }
        scal(o, 1.0f / l, head_dim);
    }
}

void attention_flash_fast(const Shape &s, const std::vector<float> &Q,
                          const std::vector<float> &K,
                          const std::vector<float> &V, bool causal,
                          std::vector<float> &O, std::vector<float> *lse)
{
    const float scale = 1.0f / std::sqrt((float)s.d);
    const int nh = s.batch * s.heads;
    const int qtiles = (s.seq + QBLK - 1) / QBLK;
    const long total = (long)nh * qtiles;

    auto run = [&](long w) {
        const int head = (int)(w / qtiles);
        const int qt = (int)(w % qtiles);
        const int r0 = qt * QBLK;
        const int r1 = std::min(s.seq, r0 + QBLK);
        const size_t hbase = (size_t)head * s.seq * s.d;
        const size_t lbase = (size_t)head * s.seq;
        process_tile(Q.data() + hbase, K.data() + hbase, V.data() + hbase,
                     O.data() + hbase, lse ? lse->data() + lbase : nullptr,
                     s.seq, s.d, scale, causal, r0, r1);
    };

    // One (head, query-tile) unit per index; the pool splits them across
    // cores (FLASH_CPU_THREADS caps the count). Each unit is independent, so
    // the split is race-free.
    parallel_for(total, /*min_chunk=*/1, [&](int64_t w0, int64_t w1) {
        for (int64_t w = w0; w < w1; w++)
            run(w);
    });
}
