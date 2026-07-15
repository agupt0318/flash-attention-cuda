// Analytic gradients of attention, the naive way: rebuild each row's
// softmax in double, then apply the chain rule with everything
// materialized,
//
//   dV = P^T dO
//   dP = dO V^T
//   dS = P o (dP - D),   D_i = sum_j P_ij dP_ij
//   dQ = dS K * scale,   dK = dS^T Q * scale.
//
// Double accumulation throughout, so float rounding in the tiled
// backward shows up as *its* error, not the reference's.

#include <cmath>
#include "attention.h"

void attention_backward_reference(const Shape &s, const std::vector<float> &Q,
                                  const std::vector<float> &K,
                                  const std::vector<float> &V, bool causal,
                                  const std::vector<float> &dO,
                                  std::vector<float> &dQ,
                                  std::vector<float> &dK,
                                  std::vector<float> &dV)
{
    const double scale = 1.0 / std::sqrt((double)s.d);
    std::vector<double> p(s.seq), dp(s.seq);
    std::vector<double> dQa(s.elems(), 0.0), dKa(s.elems(), 0.0),
        dVa(s.elems(), 0.0);

    for (int b = 0; b < s.batch; b++)
    for (int h = 0; h < s.heads; h++)
    for (int t = 0; t < s.seq; t++) {
        const float *q = &Q[s.row(b, h, t)];
        const float *dOt = &dO[s.row(b, h, t)];
        const int kmax = causal ? t + 1 : s.seq;

        // the row's softmax, recomputed in double
        double m = -1e30;
        for (int j = 0; j < kmax; j++) {
            const float *k = &K[s.row(b, h, j)];
            double dot = 0;
            for (int c = 0; c < s.d; c++)
                dot += (double)q[c] * k[c];
            p[j] = dot * scale;
            if (p[j] > m)
                m = p[j];
        }
        double l = 0;
        for (int j = 0; j < kmax; j++) {
            p[j] = std::exp(p[j] - m);
            l += p[j];
        }
        for (int j = 0; j < kmax; j++)
            p[j] /= l;

        // dP row and its P-weighted sum D
        double D = 0;
        for (int j = 0; j < kmax; j++) {
            const float *v = &V[s.row(b, h, j)];
            double dot = 0;
            for (int c = 0; c < s.d; c++)
                dot += (double)dOt[c] * v[c];
            dp[j] = dot;
            D += p[j] * dot;
        }

        // scatter this row's contributions
        double *dq = &dQa[s.row(b, h, t)];
        for (int j = 0; j < kmax; j++) {
            const float *k = &K[s.row(b, h, j)];
            double *dk = &dKa[s.row(b, h, j)];
            double *dv = &dVa[s.row(b, h, j)];
            double ds = p[j] * (dp[j] - D) * scale;
            for (int c = 0; c < s.d; c++) {
                dq[c] += ds * k[c];
                dk[c] += ds * q[c];
                dv[c] += p[j] * dOt[c];
            }
        }
    }

    for (size_t i = 0; i < s.elems(); i++) {
        dQ[i] = (float)dQa[i];
        dK[i] = (float)dKa[i];
        dV[i] = (float)dVa[i];
    }
}
