// GPU kernel vs. the double-accum CPU reference. Same hostile shapes
// as the CPU test, plus one long-sequence case, both mask modes.

#include <cstdio>
#include <vector>
#include "attention.h"
#include "flash_gpu.h"

static int check(Shape s, bool causal)
{
    std::vector<float> Q(s.elems()), K(s.elems()), V(s.elems());
    std::vector<float> Oref(s.elems()), Ogpu(s.elems());
    size_t rows = (size_t)s.batch * s.heads * s.seq;
    std::vector<float> Lref(rows), Lgpu(rows);

    fill_random(Q, 0xA11CE);
    fill_random(K, 0xB0B);
    fill_random(V, 0xC0FFEE);
    attention_reference(s, Q, K, V, causal, Oref, &Lref);

    float *dQ, *dK, *dV, *dO, *dL;
    size_t bytes = s.elems() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dQ, bytes));
    CUDA_CHECK(cudaMalloc(&dK, bytes));
    CUDA_CHECK(cudaMalloc(&dV, bytes));
    CUDA_CHECK(cudaMalloc(&dO, bytes));
    CUDA_CHECK(cudaMalloc(&dL, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), bytes, cudaMemcpyHostToDevice));

    flash_forward(s.batch, s.heads, s.seq, s.d, dQ, dK, dV, causal, dO, dL);
    CUDA_CHECK(cudaMemcpy(Ogpu.data(), dO, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Lgpu.data(), dL, rows * sizeof(float),
                          cudaMemcpyDeviceToHost));
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);

    Error eo = max_error(Ogpu, Oref);
    Error el = max_error(Lgpu, Lref);
    int ok = eo.abs < 5e-5 && el.abs < 5e-5;    // __expf is ~2 ulp
    printf("%-26s B=%d H=%d N=%-4d d=%-3d %s  O err %.2e  lse err %.2e\n",
           causal ? "gpu vs ref (causal)" : "gpu vs ref", s.batch, s.heads,
           s.seq, s.d, ok ? "ok  " : "FAIL", eo.abs, el.abs);
    return ok ? 0 : 1;
}

// Backward: the device kernels consume the device forward's own O and
// lse, judged against the analytic CPU reference (which test_cpu_bwd
// grounds in finite differences).
static int check_bwd(Shape s, bool causal)
{
    std::vector<float> Q(s.elems()), K(s.elems()), V(s.elems());
    std::vector<float> dOh(s.elems());
    std::vector<float> dQr(s.elems()), dKr(s.elems()), dVr(s.elems());
    std::vector<float> dQg(s.elems()), dKg(s.elems()), dVg(s.elems());
    size_t rows = (size_t)s.batch * s.heads * s.seq;

    fill_random(Q, 0xA11CE);
    fill_random(K, 0xB0B);
    fill_random(V, 0xC0FFEE);
    fill_random(dOh, 0xD00D);
    attention_backward_reference(s, Q, K, V, causal, dOh, dQr, dKr, dVr);

    float *q, *k, *v, *o, *l, *dO, *dq, *dk, *dv;
    size_t bytes = s.elems() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&q, bytes));
    CUDA_CHECK(cudaMalloc(&k, bytes));
    CUDA_CHECK(cudaMalloc(&v, bytes));
    CUDA_CHECK(cudaMalloc(&o, bytes));
    CUDA_CHECK(cudaMalloc(&dO, bytes));
    CUDA_CHECK(cudaMalloc(&dq, bytes));
    CUDA_CHECK(cudaMalloc(&dk, bytes));
    CUDA_CHECK(cudaMalloc(&dv, bytes));
    CUDA_CHECK(cudaMalloc(&l, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(q, Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(k, K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v, V.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dO, dOh.data(), bytes, cudaMemcpyHostToDevice));

    flash_forward(s.batch, s.heads, s.seq, s.d, q, k, v, causal, o, l);
    flash_backward(s.batch, s.heads, s.seq, s.d, q, k, v, o, l, dO, causal,
                   dq, dk, dv);
    CUDA_CHECK(cudaMemcpy(dQg.data(), dq, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dKg.data(), dk, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dVg.data(), dv, bytes, cudaMemcpyDeviceToHost));
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o); cudaFree(l);
    cudaFree(dO); cudaFree(dq); cudaFree(dk); cudaFree(dv);

    Error eq = max_error(dQg, dQr), ek = max_error(dKg, dKr),
          ev = max_error(dVg, dVr);
    // float accumulation over seq rows, plus __expf's ~2 ulp
    double tol = 5e-5 * (1.0 + s.seq / 64.0);
    int ok = eq.abs < tol && ek.abs < tol && ev.abs < tol;
    printf("%-26s B=%d H=%d N=%-4d d=%-3d %s  dQ %.2e  dK %.2e  dV %.2e\n",
           causal ? "gpu bwd vs ref (causal)" : "gpu bwd vs ref", s.batch,
           s.heads, s.seq, s.d, ok ? "ok  " : "FAIL", eq.abs, ek.abs, ev.abs);
    return ok ? 0 : 1;
}

int main()
{
    int fails = 0;
    const Shape shapes[] = {
        { 1, 1, 64, 32 },   { 2, 3, 128, 64 }, { 1, 2, 257, 64 },
        { 1, 1, 33, 64 },   { 2, 2, 300, 32 }, { 1, 4, 1024, 64 },
    };
    for (const Shape &s : shapes) {
        fails += check(s, false);
        fails += check(s, true);
        fails += check_bwd(s, false);
        fails += check_bwd(s, true);
    }
    printf(fails ? "FAILED (%d)\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
