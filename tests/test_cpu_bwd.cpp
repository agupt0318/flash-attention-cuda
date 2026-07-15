// Backward-pass correctness, two layers deep:
//
// 1. The analytic reference itself is checked against central finite
//    differences of the forward on a tiny shape -- the gradients are
//    grounded in the definition of a derivative, not in a second copy
//    of the same chain-rule algebra.
// 2. The tiled backward (float, logsumexp recompute, Appendix B) is
//    checked against that reference across the same shapes-chosen-to-
//    hurt as the forward tests, both mask modes.

#include <cstdio>
#include <vector>
#include "attention.h"

// d/dx of sum(O o W) at x = one input element, by central difference
static double fd_grad(const Shape &s, std::vector<float> Q,
                      std::vector<float> K, std::vector<float> V, bool causal,
                      const std::vector<float> &W, int which, size_t i)
{
    const float h = 1e-3f;
    std::vector<float> *X[3] = { &Q, &K, &V };
    std::vector<float> O(s.elems());
    double loss[2];
    for (int sgn = 0; sgn < 2; sgn++) {
        float saved = (*X[which])[i];
        (*X[which])[i] = saved + (sgn ? -h : h);
        attention_reference(s, Q, K, V, causal, O, nullptr);
        (*X[which])[i] = saved;
        loss[sgn] = 0;
        for (size_t j = 0; j < O.size(); j++)
            loss[sgn] += (double)O[j] * W[j];
    }
    return (loss[0] - loss[1]) / (2.0 * h);
}

// ground the analytic reference in finite differences
static int check_fd(bool causal)
{
    Shape s{ 1, 1, 8, 4 };
    std::vector<float> Q(s.elems()), K(s.elems()), V(s.elems()), W(s.elems());
    std::vector<float> dQ(s.elems()), dK(s.elems()), dV(s.elems());
    fill_random(Q, 0xA11CE);
    fill_random(K, 0xB0B);
    fill_random(V, 0xC0FFEE);
    fill_random(W, 0xD00D); // loss = sum(O o W), so dO = W

    attention_backward_reference(s, Q, K, V, causal, W, dQ, dK, dV);

    std::vector<float> *G[3] = { &dQ, &dK, &dV };
    double worst = 0;
    for (int which = 0; which < 3; which++)
        for (size_t i = 0; i < s.elems(); i += 5) {
            double fd = fd_grad(s, Q, K, V, causal, W, which, i);
            double err = std::fabs(fd - (double)(*G[which])[i]);
            if (err > worst)
                worst = err;
        }
    // h^2 truncation + float-forward noise through /2h; loose on purpose
    int ok = worst < 5e-3;
    printf("%-30s B=%d H=%d N=%-4d d=%-3d %s  worst fd err %.2e\n",
           causal ? "reference vs finite-diff (c)" : "reference vs finite-diff",
           s.batch, s.heads, s.seq, s.d, ok ? "ok  " : "FAIL", worst);
    return ok ? 0 : 1;
}

static int check(Shape s, bool causal)
{
    std::vector<float> Q(s.elems()), K(s.elems()), V(s.elems());
    std::vector<float> O(s.elems()), dO(s.elems());
    size_t rows = (size_t)s.batch * s.heads * s.seq;
    std::vector<float> L(rows);
    std::vector<float> dQr(s.elems()), dKr(s.elems()), dVr(s.elems());
    std::vector<float> dQf(s.elems()), dKf(s.elems()), dVf(s.elems());

    fill_random(Q, 0xA11CE);
    fill_random(K, 0xB0B);
    fill_random(V, 0xC0FFEE);
    fill_random(dO, 0xD00D);

    // the tiled backward consumes the flash forward's own O and lse,
    // exactly what the CUDA kernel will have on hand
    attention_flash_cpu(s, Q, K, V, causal, O, &L);
    attention_flash_cpu_bwd(s, Q, K, V, causal, O, L, dO, dQf, dKf, dVf);
    attention_backward_reference(s, Q, K, V, causal, dO, dQr, dKr, dVr);

    Error eq = max_error(dQf, dQr), ek = max_error(dKf, dKr),
          ev = max_error(dVf, dVr);
    // dK/dV rows accumulate over all queries in float; scale with seq
    double tol = 2e-5 * (1.0 + s.seq / 64.0);
    int ok = eq.abs < tol && ek.abs < tol && ev.abs < tol;
    printf("%-30s B=%d H=%d N=%-4d d=%-3d %s  dQ %.2e  dK %.2e  dV %.2e\n",
           causal ? "flash_bwd vs ref (causal)" : "flash_bwd vs ref",
           s.batch, s.heads, s.seq, s.d, ok ? "ok  " : "FAIL",
           eq.abs, ek.abs, ev.abs);
    return ok ? 0 : 1;
}

int main()
{
    int fails = 0;
    fails += check_fd(false);
    fails += check_fd(true);
    const Shape shapes[] = {
        { 1, 1, 64, 32 },       // one tile exactly
        { 2, 3, 128, 64 },      // multiple tiles, multiple heads
        { 1, 2, 257, 64 },      // ragged: 257 = 4*64 + 1
        { 1, 1, 33, 64 },       // shorter than one tile
        { 2, 2, 300, 32 },      // ragged again, small head dim
    };
    for (const Shape &s : shapes) {
        fails += check(s, false);
        fails += check(s, true);
    }
    printf(fails ? "FAILED (%d)\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
