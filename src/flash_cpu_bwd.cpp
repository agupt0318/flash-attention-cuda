// The backward pass the way the kernel will run it (Appendix B of the
// paper): nothing NxN ever exists. Each query row streams through K/V
// tiles again, P is recomputed one tile at a time from the forward's
// logsumexp -- p_j = exp(s_j - L_t), no second softmax, no saved score
// matrix -- and the softmax Jacobian's row coupling collapses into one
// precomputed scalar per row,
//
//   D_t = sum_j P_tj dP_tj = rowsum(dO_t o O_t),
//
// which is why the forward saving O and L is all the backward needs.
// Deliberately float, like flash_cpu.cpp: these are the numerics the
// CUDA kernel will have, judged against the double-accum reference.

#include <cmath>
#include "attention.h"

#define TILE_N 64

void attention_flash_cpu_bwd(const Shape &s, const std::vector<float> &Q,
                             const std::vector<float> &K,
                             const std::vector<float> &V, bool causal,
                             const std::vector<float> &O,
                             const std::vector<float> &lse,
                             const std::vector<float> &dO,
                             std::vector<float> &dQ, std::vector<float> &dK,
                             std::vector<float> &dV)
{
    const float scale = 1.0f / std::sqrt((float)s.d);
    dQ.assign(s.elems(), 0.0f);
    dK.assign(s.elems(), 0.0f);
    dV.assign(s.elems(), 0.0f);

    for (int b = 0; b < s.batch; b++)
    for (int h = 0; h < s.heads; h++)
    for (int t = 0; t < s.seq; t++) {
        const float *q = &Q[s.row(b, h, t)];
        const float *o = &O[s.row(b, h, t)];
        const float *dOt = &dO[s.row(b, h, t)];
        const float L = lse[((size_t)b * s.heads + h) * s.seq + t];

        // the Appendix B identity: D_t needs only this row of dO and O
        float D = 0;
        for (int c = 0; c < s.d; c++)
            D += dOt[c] * o[c];

        float *dq = &dQ[s.row(b, h, t)];
        for (int j0 = 0; j0 < s.seq; j0 += TILE_N) {
            if (causal && j0 > t)
                break;                  // tile fully above the diagonal
            int jn = s.seq - j0 < TILE_N ? s.seq - j0 : TILE_N;

            for (int j = 0; j < jn; j++) {
                if (causal && j0 + j > t)
                    break;              // rest of the tile is masked
                const float *k = &K[s.row(b, h, j0 + j)];
                const float *v = &V[s.row(b, h, j0 + j)];
                float *dk = &dK[s.row(b, h, j0 + j)];
                float *dv = &dV[s.row(b, h, j0 + j)];

                // recompute this score, then p from the saved logsumexp
                float dot = 0, dp = 0;
                for (int c = 0; c < s.d; c++) {
                    dot += q[c] * k[c];
                    dp += dOt[c] * v[c];
                }
                float p = std::exp(dot * scale - L);
                float ds = p * (dp - D) * scale;

                for (int c = 0; c < s.d; c++) {
                    dq[c] += ds * k[c];
                    dk[c] += ds * q[c];
                    dv[c] += p * dOt[c];
                }
            }
        }
    }
}
