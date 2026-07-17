"""PyTorch front door for the kernel.

    from flash_attn import flash_attention
    o = flash_attention(q, k, v, causal=True)   # differentiable

q/k/v: [batch, heads, seq, head_dim] fp32 tensors, head_dim 32 or 64,
the same contract as torch's scaled_dot_product_attention with explicit
head layout. CUDA and CPU are both backends: CUDA tensors run the hand
kernel (src/flash_fwd.cu + flash_bwd.cu), CPU tensors run the validated
C++ algorithm (src/flash_cpu.cpp + flash_cpu_bwd.cpp). The right
extension JIT-compiles on first use; the CPU one needs only a host
compiler, so the op runs (and gradchecks) with no GPU in the loop.

The op is a torch.autograd.Function: it records a backward node, so it
is trainable and drops into an nn.Module like SDPA. The backward is the
paper's Appendix B (recompute P from O + logsumexp, D = rowsum(dO.O)).
"""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent.parent
_cuda_ext = None
_cpu_ext = None


def _cuda_module():
    global _cuda_ext
    if _cuda_ext is None:
        _cuda_ext = load(
            name="flash_attention_cuda_ext",
            sources=[
                str(_ROOT / "pytorch" / "binding.cpp"),
                str(_ROOT / "src" / "flash_fwd.cu"),
                str(_ROOT / "src" / "flash_bwd.cu"),
            ],
            extra_include_paths=[str(_ROOT / "src")],
            extra_cuda_cflags=["-O3", "-lineinfo"],
        )
    return _cuda_ext


def _cpu_module():
    global _cpu_ext
    if _cpu_ext is None:
        _cpu_ext = load(
            name="flash_attention_cpu_ext",
            sources=[
                str(_ROOT / "pytorch" / "binding_cpu.cpp"),
                str(_ROOT / "src" / "flash_cpu.cpp"),
                str(_ROOT / "src" / "flash_cpu_bwd.cpp"),
            ],
            extra_include_paths=[str(_ROOT / "src")],
            extra_cflags=["-O2"],
        )
    return _cpu_ext


def _backend(t: torch.Tensor):
    return _cuda_module() if t.is_cuda else _cpu_module()


class _FlashAttention(torch.autograd.Function):
    """Records the backward node. Forward saves O and the per-row
    logsumexp; backward recomputes P from them (Appendix B), so no NxN
    intermediate is ever stored, on the forward or the backward."""

    @staticmethod
    def forward(ctx, q, k, v, causal):
        ext = _backend(q)
        o, lse = ext.forward(q, k, v, causal)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, o, lse = ctx.saved_tensors
        ext = _backend(q)
        dq, dk, dv = ext.backward(q, k, v, o, lse, grad_out.contiguous(),
                                  ctx.causal)
        return dq, dk, dv, None


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    causal: bool = False) -> torch.Tensor:
    """softmax(q·kᵀ/√d)·v without materializing the N×N score matrix.
    Differentiable: use it in a training loop or an nn.Module."""
    return _FlashAttention.apply(q, k, v, causal)


__all__ = ["flash_attention"]
