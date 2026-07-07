"""Parity vs torch: our kernel and F.scaled_dot_product_attention are
both checked against a float64 einsum reference, so a torch-side
shortcut can't hide a bug of ours. Needs a CUDA box.

    python3 pytorch/test_parity.py            # correctness
    python3 pytorch/test_parity.py bench      # + timing vs SDPA
"""

import sys
import torch
import torch.nn.functional as F
from flash_attn import flash_attention

torch.manual_seed(7)


def exact(q, k, v, causal):
    q, k, v = (t.double() for t in (q, k, v))
    s = q @ k.transpose(-1, -2) / (q.size(-1) ** 0.5)
    if causal:
        n = s.size(-1)
        s = s.masked_fill(torch.ones(n, n, device=s.device,
                                     dtype=torch.bool).triu(1), -torch.inf)
    return (s.softmax(-1) @ v).float()


def check(b, h, n, d, causal):
    q, k, v = (torch.randn(b, h, n, d, device="cuda") for _ in range(3))
    ref = exact(q, k, v, causal)
    ours = flash_attention(q, k, v, causal)
    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    e_ours = (ours - ref).abs().max().item()
    e_sdpa = (sdpa - ref).abs().max().item()
    ok = e_ours < 5e-5
    print(f"B={b} H={h} N={n:<5} d={d:<3} causal={int(causal)} "
          f"{'ok  ' if ok else 'FAIL'} ours {e_ours:.2e}  sdpa {e_sdpa:.2e}")
    return 0 if ok else 1


def bench(b, h, n, d):
    q, k, v = (torch.randn(b, h, n, d, device="cuda") for _ in range(3))
    for fn, name in ((lambda: flash_attention(q, k, v), "ours"),
                     (lambda: F.scaled_dot_product_attention(q, k, v), "sdpa")):
        fn(); torch.cuda.synchronize()
        t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
        t0.record()
        for _ in range(30):
            fn()
        t1.record(); torch.cuda.synchronize()
        print(f"  N={n:<5} {name}: {t0.elapsed_time(t1) / 30:7.3f} ms")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("no CUDA device — parity test needs a GPU"); sys.exit(0)
    fails = 0
    for shape in [(1, 1, 64, 32), (2, 3, 128, 64), (1, 2, 257, 64),
                  (1, 1, 33, 64), (2, 2, 300, 32), (1, 4, 1024, 64)]:
        for causal in (False, True):
            fails += check(*shape, causal)
    print("FAILED" if fails else "all passed")
    if "bench" in sys.argv[1:]:
        for n in (512, 1024, 2048, 4096):
            bench(4, 8, n, 64)
    sys.exit(1 if fails else 0)
