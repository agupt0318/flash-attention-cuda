// PyTorch binding: flash_attention_cuda_ext.forward(q, k, v, causal).
// Validates the contract the kernel assumes, launches on torch's
// current stream, and hands back a fresh output tensor. Inference-only
// for now: no autograd node; the backward kernel is on the roadmap.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

void flash_forward(int batch, int heads, int seq, int d, const float *Q,
                   const float *K, const float *V, bool causal, float *O,
                   float *lse, cudaStream_t stream);

static torch::Tensor forward(torch::Tensor q, torch::Tensor k,
                             torch::Tensor v, bool causal)
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

    const at::cuda::CUDAGuard guard(q.device());
    auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
    auto o = torch::empty_like(qc);

    flash_forward((int)q.size(0), (int)q.size(1), (int)q.size(2), d,
                  qc.data_ptr<float>(), kc.data_ptr<float>(),
                  vc.data_ptr<float>(), causal, o.data_ptr<float>(),
                  nullptr, at::cuda::getCurrentCUDAStream());
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &forward, "FlashAttention forward (fp32, CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("causal") = false);
}
