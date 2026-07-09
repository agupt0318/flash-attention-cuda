// FlashAttention's algorithm, minus the GPU: process K/V in tiles of
// TILE_N, keeping only a running row max m, normalizer l, and an
// unnormalized accumulator per query row (Algorithm 1 of the paper,
// with the loop order the CUDA kernel uses, so each query row streams
// through all K/V tiles). Deliberately float, not double: this is the
// numerics the kernel will have, validated against the double-accum
// reference before any CUDA exists.

#include <cmath>
#include "attention.h"

#define TILE_N 64

void attention_flash_cpu(const Shape &s, const std::vector<float> &Q,
                         const std::vector<float> &K,
                         const std::vector<float> &V, bool causal,
                         std::vector<float> &O, std::vector<float> *lse)
{
    const float scale = 1.0f / std::sqrt((float)s.d);
    std::vector<float> acc(s.d);
    float p[TILE_N];

    for (int b = 0; b < s.batch; b++)
    for (int h = 0; h < s.heads; h++)
    for (int t = 0; t < s.seq; t++) {
        const float *q = &Q[s.row(b, h, t)];
        float m = -1e30f, l = 0.0f;
        acc.assign(s.d, 0.0f);

        for (int j0 = 0; j0 < s.seq; j0 += TILE_N) {
            if (causal && j0 > t)
                break;                      // tile fully above the diagonal
            int jn = s.seq - j0 < TILE_N ? s.seq - j0 : TILE_N;

            // scores for this tile, tracking its max
            float mt = -1e30f;
            for (int j = 0; j < jn; j++) {
                if (causal && j0 + j > t) {
                    p[j] = -1e30f;          // masked: exp() underflows to 0
                    continue;
                }
                const float *k = &K[s.row(b, h, j0 + j)];
                float dot = 0;
                for (int c = 0; c < s.d; c++)
                    dot += q[c] * k[c];
                p[j] = dot * scale;
                if (p[j] > mt)
                    mt = p[j];
            }

            // online softmax: rescale history to the new max, fold tile in
            float mn = m > mt ? m : mt;
            float corr = std::exp(m - mn);
            l *= corr;
            for (int c = 0; c < s.d; c++)
                acc[c] *= corr;
            for (int j = 0; j < jn; j++) {
                float e = std::exp(p[j] - mn);
                l += e;
                const float *v = &V[s.row(b, h, j0 + j)];
                for (int c = 0; c < s.d; c++)
                    acc[c] += e * v[c];
            }
            m = mn;
        }

        float *o = &O[s.row(b, h, t)];
        for (int c = 0; c < s.d; c++)
            o[c] = acc[c] / l;
        if (lse)
            (*lse)[((size_t)b * s.heads + h) * s.seq + t] = m + std::log(l);
    }
}
