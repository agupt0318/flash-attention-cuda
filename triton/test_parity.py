"""Triton kernel vs a float64 reference, with the hand-written CUDA
kernel alongside when its extension is importable — three ways to
compute the same attention, one judge. Needs a CUDA box.

    python3 triton/test_parity.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch"))
from flash_attn_triton import flash_attention_triton

torch.manual_seed(7)


def exact(q, k, v, causal):
    q, k, v = (t.double() for t in (q, k, v))
    s = q @ k.transpose(-1, -2) / (q.size(-1) ** 0.5)
    if causal:
        n = s.size(-1)
        s = s.masked_fill(torch.ones(n, n, device=s.device,
                                     dtype=torch.bool).triu(1), -torch.inf)
    return (s.softmax(-1) @ v).float()


def check(b, h, n, d, causal, cuda_impl):
    q, k, v = (torch.randn(b, h, n, d, device="cuda") for _ in range(3))
    ref = exact(q, k, v, causal)
    e_tri = (flash_attention_triton(q, k, v, causal) - ref).abs().max().item()
    line = (f"B={b} H={h} N={n:<5} d={d:<3} causal={int(causal)} "
            f"{'ok  ' if e_tri < 5e-5 else 'FAIL'} triton {e_tri:.2e}")
    if cuda_impl:
        e_cu = (cuda_impl(q, k, v, causal) - ref).abs().max().item()
        line += f"  cuda {e_cu:.2e}"
    print(line)
    return 0 if e_tri < 5e-5 else 1


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("no CUDA device — parity test needs a GPU")
        sys.exit(0)
    try:
        from flash_attn import flash_attention as cuda_impl
    except Exception:
        cuda_impl = None                    # nvcc-less box: triton only
    fails = 0
    for shape in [(1, 1, 64, 32), (2, 3, 128, 64), (1, 2, 257, 64),
                  (1, 1, 33, 64), (2, 2, 300, 32), (1, 4, 1024, 64)]:
        for causal in (False, True):
            fails += check(*shape, causal, cuda_impl)
    print("FAILED" if fails else "all passed")
    sys.exit(1 if fails else 0)
