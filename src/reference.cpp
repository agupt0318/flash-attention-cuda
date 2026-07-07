// Naive attention: the ground truth everything else is judged against.
// One query row at a time — scores, stable softmax, weighted V sum —
// with double accumulation throughout so float rounding in the fast
// implementations shows up as *their* error, not the reference's.

#include <cmath>
#include "attention.h"

void attention_reference(const Shape &s, const std::vector<float> &Q,
                         const std::vector<float> &K,
                         const std::vector<float> &V, bool causal,
                         std::vector<float> &O, std::vector<float> *lse)
{
    const double scale = 1.0 / std::sqrt((double)s.d);
    std::vector<double> sc(s.seq), acc(s.d);

    for (int b = 0; b < s.batch; b++)
    for (int h = 0; h < s.heads; h++)
    for (int t = 0; t < s.seq; t++) {
        const float *q = &Q[s.row(b, h, t)];
        const int kmax = causal ? t + 1 : s.seq;

        double m = -1e30;
        for (int j = 0; j < kmax; j++) {
            const float *k = &K[s.row(b, h, j)];
            double dot = 0;
            for (int c = 0; c < s.d; c++)
                dot += (double)q[c] * k[c];
            sc[j] = dot * scale;
            if (sc[j] > m)
                m = sc[j];
        }

        double l = 0;
        for (int j = 0; j < kmax; j++) {
            sc[j] = std::exp(sc[j] - m);
            l += sc[j];
        }

        acc.assign(s.d, 0.0);
        for (int j = 0; j < kmax; j++) {
            const float *v = &V[s.row(b, h, j)];
            for (int c = 0; c < s.d; c++)
                acc[c] += sc[j] * v[c];
        }
        float *o = &O[s.row(b, h, t)];
        for (int c = 0; c < s.d; c++)
            o[c] = (float)(acc[c] / l);
        if (lse)
            (*lse)[((size_t)b * s.heads + h) * s.seq + t] =
                (float)(m + std::log(l));
    }
}
