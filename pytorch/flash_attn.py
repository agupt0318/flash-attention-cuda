"""PyTorch front door for the kernel.

    from flash_attn import flash_attention
    o = flash_attention(q, k, v, causal=True)

q/k/v: [batch, heads, seq, head_dim] fp32 CUDA tensors, head_dim 32 or
64 — the same contract as torch's scaled_dot_product_attention with
explicit head layout. The extension JIT-compiles on first import
(needs nvcc + a CUDA build of torch); nothing to install.

Inference-only: the op does not record an autograd node. The backward
kernel (recompute from O + logsumexp, paper Appendix B) is roadmap.
"""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent.parent
_ext = None


def _module():
    global _ext
    if _ext is None:
        _ext = load(
            name="flash_attention_cuda_ext",
            sources=[
                str(_ROOT / "pytorch" / "binding.cpp"),
                str(_ROOT / "src" / "flash_fwd.cu"),
            ],
            extra_include_paths=[str(_ROOT / "src")],
            extra_cuda_cflags=["-O3", "-lineinfo"],
        )
    return _ext


@torch.no_grad()
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    causal: bool = False) -> torch.Tensor:
    """softmax(q·kᵀ/√d)·v without materializing the N×N score matrix."""
    return _module().forward(q, k, v, causal)


__all__ = ["flash_attention"]
