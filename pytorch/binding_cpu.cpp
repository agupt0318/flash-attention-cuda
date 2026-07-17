// CPU backend for the trainable FlashAttention op. Same online-softmax
// algorithm as the CUDA kernel, run through the validated C++ code
// (src/flash_cpu.cpp forward, src/flash_cpu_bwd.cpp backward). Its
// reason to exist: the autograd.Function needs a backend on a machine
// with no GPU, so torch.autograd.gradcheck can prove the whole
// forward+backward against finite differences with no CUDA in the loop.
// Correctness without a GPU, extended to the gradient.

#include <torch/extension.h>
#include <vector>
#include "attention.h"

static Shape shape_of(const torch::Tensor &q)
{
    return Shape{ (int)q.size(0), (int)q.size(1), (int)q.size(2),
                  (int)q.size(3) };
}

static std::vector<float> to_vec(const torch::Tensor &t)
{
    auto c = t.contiguous();
    const float *p = c.data_ptr<float>();
    return std::vector<float>(p, p + c.numel());
}

static torch::Tensor to_tensor(const std::vector<float> &v,
                               torch::IntArrayRef sizes)
{
    return torch::from_blob((void *)v.data(), sizes,
                            torch::kFloat32).clone();
}

static void check_contract(const torch::Tensor &q, const torch::Tensor &k,
                           const torch::Tensor &v)
{
    TORCH_CHECK(q.device().is_cpu() && k.device().is_cpu() &&
                    v.device().is_cpu(),
                "flash_attention cpu backend: q/k/v must be CPU tensors");
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

// forward -> (O, lse). lse is [batch, heads, seq]; the backward needs it,
// so unlike the inference path we always compute and return it.
static std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k,
                                          torch::Tensor v, bool causal)
{
    check_contract(q, k, v);
    const Shape s = shape_of(q);
    std::vector<float> Q = to_vec(q), K = to_vec(k), V = to_vec(v);
    std::vector<float> O(s.elems());
    std::vector<float> L((size_t)s.batch * s.heads * s.seq);

    attention_flash_cpu(s, Q, K, V, causal, O, &L);

    return { to_tensor(O, q.sizes()),
             to_tensor(L, { s.batch, s.heads, s.seq }) };
}

// backward -> (dQ, dK, dV), consuming the forward's own O and lse.
static std::vector<torch::Tensor> backward(torch::Tensor q, torch::Tensor k,
                                           torch::Tensor v, torch::Tensor o,
                                           torch::Tensor lse, torch::Tensor dO,
                                           bool causal)
{
    check_contract(q, k, v);
    const Shape s = shape_of(q);
    std::vector<float> Q = to_vec(q), K = to_vec(k), V = to_vec(v),
                       O = to_vec(o), L = to_vec(lse), dOut = to_vec(dO);
    std::vector<float> dQ(s.elems()), dK(s.elems()), dV(s.elems());

    attention_flash_cpu_bwd(s, Q, K, V, causal, O, L, dOut, dQ, dK, dV);

    return { to_tensor(dQ, q.sizes()), to_tensor(dK, q.sizes()),
             to_tensor(dV, q.sizes()) };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &forward, "FlashAttention forward (fp32, CPU)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
    m.def("backward", &backward, "FlashAttention backward (fp32, CPU)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("o"),
          py::arg("lse"), py::arg("dO"), py::arg("causal") = false);
}
