// PyTorch binding for the CUDA kernels: forward -> (O, lse) and
// backward -> (dQ, dK, dV). Validates the contract the kernels assume,
// launches on torch's current stream, and hands back fresh tensors. The
// forward returns the per-row logsumexp too (not just O like the old
// inference path), because the autograd.Function saves it for a backward
// that recomputes P from O + lse (paper Appendix B) instead of an N×N.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

void flash_forward(int batch, int heads, int seq, int d, const float *Q,
                   const float *K, const float *V, bool causal, float *O,
                   float *lse, cudaStream_t stream);

void flash_backward(int batch, int heads, int seq, int d, const float *Q,
                    const float *K, const float *V, const float *O,
                    const float *lse, const float *dO, bool causal, float *dQ,
                    float *dK, float *dV, cudaStream_t stream);

static void check_contract(const torch::Tensor &q, const torch::Tensor &k,
                           const torch::Tensor &v)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "flash_attention: q/k/v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32,
                "flash_attention: fp32 only (fp16 is roadmap)");
    TORCH_CHECK(q.dim() == 4, "flash_attention: expected "
                "[batch, heads, seq, head_dim], got ", q.dim(), " dims");
    TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(),
                "flash_attention: q/k/v shapes must match");
    const int d = (int)q.size(3);
    TORCH_CHECK(d == 32 || d == 64,
                "flash_attention: head_dim must be 32 or 64, got ", d);
}

// forward -> (O, lse). lse is [batch, heads, seq], required by backward.
static std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k,
                                          torch::Tensor v, bool causal)
{
    check_contract(q, k, v);
    const at::cuda::CUDAGuard guard(q.device());
    auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
    auto o = torch::empty_like(qc);
    auto lse = torch::empty({ q.size(0), q.size(1), q.size(2) },
                            qc.options());

    flash_forward((int)q.size(0), (int)q.size(1), (int)q.size(2),
                  (int)q.size(3), qc.data_ptr<float>(), kc.data_ptr<float>(),
                  vc.data_ptr<float>(), causal, o.data_ptr<float>(),
                  lse.data_ptr<float>(), at::cuda::getCurrentCUDAStream());
    return { o, lse };
}

// backward -> (dQ, dK, dV), consuming the forward's own O and lse.
static std::vector<torch::Tensor> backward(torch::Tensor q, torch::Tensor k,
                                           torch::Tensor v, torch::Tensor o,
                                           torch::Tensor lse, torch::Tensor dO,
                                           bool causal)
{
    check_contract(q, k, v);
    const at::cuda::CUDAGuard guard(q.device());
    auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous(),
         oc = o.contiguous(), lc = lse.contiguous(), dc = dO.contiguous();
    auto dQ = torch::empty_like(qc), dK = torch::empty_like(qc),
         dV = torch::empty_like(qc);

    flash_backward((int)q.size(0), (int)q.size(1), (int)q.size(2),
                   (int)q.size(3), qc.data_ptr<float>(), kc.data_ptr<float>(),
                   vc.data_ptr<float>(), oc.data_ptr<float>(),
                   lc.data_ptr<float>(), dc.data_ptr<float>(), causal,
                   dQ.data_ptr<float>(), dK.data_ptr<float>(),
                   dV.data_ptr<float>(), at::cuda::getCurrentCUDAStream());
    return { dQ, dK, dV };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &forward, "FlashAttention forward (fp32, CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
    m.def("backward", &backward, "FlashAttention backward (fp32, CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("o"),
          py::arg("lse"), py::arg("dO"), py::arg("causal") = false);
}
