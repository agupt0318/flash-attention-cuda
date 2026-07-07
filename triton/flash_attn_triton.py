"""FlashAttention forward in Triton — the answer to the paper's own
"Compiling to CUDA" future-work paragraph (Section 5): the same
IO-aware algorithm written in a high-level DSL and compiled, instead
of hand-scheduled in CUDA.

Same recurrence as src/flash_fwd.cu, different granularity: Triton
programs work on whole tiles, so the online-softmax statistics update
per K/V tile (Algorithm 1 exactly) with tl.dot doing the tile matmuls
— what the CUDA kernel spells out in registers and __syncthreads, the
compiler schedules here.

    from flash_attn_triton import flash_attention_triton
    o = flash_attention_triton(q, k, v, causal=True)

Contract matches the CUDA binding: [batch, heads, seq, head_dim],
fp32, CUDA tensors. tl.dot runs with allow_tf32=False so the parity
tolerance stays honest; flip it for speed once numbers matter more
than digits.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(Q, K, V, O, Lse, seq_len, scale,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)                # query-tile index
    pid_bh = tl.program_id(1)               # batch*heads index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    base = pid_bh.to(tl.int64) * seq_len * HEAD_DIM

    q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                mask=offs_m[:, None] < seq_len, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)
    else:
        hi = seq_len

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < seq_len

        k = tl.load(K + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k), allow_tf32=False) * scale
        s = tl.where(offs_n[None, :] < seq_len, s, float("-inf"))
        if CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s,
                         float("-inf"))

        # Algorithm 1's tile-granular online update
        m_new = tl.maximum(m_i, tl.max(s, 1))
        corr = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * corr + tl.sum(p, 1)

        v = tl.load(V + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=kv_mask, other=0.0)
        acc = acc * corr[:, None] + tl.dot(p, v, allow_tf32=False)
        m_i = m_new

    live = offs_m[:, None] < seq_len
    tl.store(O + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
             acc / l_i[:, None], mask=live)
    tl.store(Lse + pid_bh.to(tl.int64) * seq_len + offs_m,
             m_i + tl.log(l_i), mask=offs_m < seq_len)


def flash_attention_triton(q: torch.Tensor, k: torch.Tensor,
                           v: torch.Tensor, causal: bool = False,
                           block_m: int = 64, block_n: int = 64):
    """softmax(q·kᵀ/√d)·v, tiled through SRAM by the Triton compiler."""
    assert q.is_cuda and q.dtype == torch.float32, "fp32 CUDA tensors"
    assert q.dim() == 4 and q.shape == k.shape == v.shape
    b, h, n, d = q.shape
    assert d in (32, 64), f"head_dim must be 32 or 64, got {d}"

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    lse = torch.empty(b, h, n, device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(n, block_m), b * h)
    _fwd_kernel[grid](q, k, v, o, lse, n, 1.0 / d ** 0.5,
                      BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=d,
                      CAUSAL=causal)
    return o


__all__ = ["flash_attention_triton"]
