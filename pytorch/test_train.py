"""The op is trainable: it records a backward node, so gradients flow.
This checks that three ways, and unlike parity it needs no GPU -- the
CPU backend runs the same algorithm, so the whole forward+backward is
proven here with no CUDA in the loop.

    python3 pytorch/test_train.py

1. gradcheck: finite differences of our forward vs our analytic
   backward, the derivative grounded in its own definition.
2. gradient parity: dQ/dK/dV vs autograd through a float64 exact
   attention, so our backward can't hide a bug the forward shares.
3. a training loop that fits Q to a target output, asserting the loss
   actually falls -- the end the whole op exists for.

Runs on CPU, and on CUDA too when a device is present.
"""

import sys
import warnings

import torch
import torch.nn.functional as F
from flash_attn import flash_attention

# gradcheck warns that a fp32 op is not fp64; that is the point here (the
# op is fp32-only), and the tolerances are sized for it.
warnings.filterwarnings("ignore", message=".*not a double precision.*")

torch.manual_seed(7)


def exact(q, k, v, causal):
    """float64 ground-truth attention, differentiable by torch autograd."""
    s = q @ k.transpose(-1, -2) / (q.size(-1) ** 0.5)
    if causal:
        n = s.size(-1)
        s = s.masked_fill(torch.ones(n, n, device=s.device,
                                     dtype=torch.bool).triu(1), -torch.inf)
    return s.softmax(-1) @ v


def rand_qkv(b, h, n, d, device, dtype, requires_grad=False):
    return [torch.randn(b, h, n, d, device=device, dtype=dtype,
                        requires_grad=requires_grad) for _ in range(3)]


def test_gradcheck(device):
    # small and fp64-eps on a fp32 op: eps/tolerances sized for float.
    q, k, v = rand_qkv(1, 2, 16, 32, device, torch.float32, requires_grad=True)
    for causal in (False, True):
        ok = torch.autograd.gradcheck(
            lambda a, b_, c: flash_attention(a, b_, c, causal),
            (q, k, v), eps=1e-3, atol=2e-2, rtol=2e-2, nondet_tol=1e-4)
        print(f"  gradcheck  causal={int(causal)}  {'ok' if ok else 'FAIL'}")
        if not ok:
            return 1
    return 0


def test_grad_parity(device):
    fails = 0
    for b, h, n, d in [(1, 1, 64, 32), (2, 3, 128, 64), (1, 2, 257, 64)]:
        for causal in (False, True):
            q, k, v = rand_qkv(b, h, n, d, device, torch.float32)
            g = torch.randn(b, h, n, d, device=device)   # upstream grad dO

            qo, ko, vo = (t.detach().requires_grad_(True) for t in (q, k, v))
            flash_attention(qo, ko, vo, causal).backward(g)

            qr, kr, vr = (t.detach().double().requires_grad_(True)
                          for t in (q, k, v))
            exact(qr, kr, vr, causal).backward(g.double())

            errs = [(o.grad - r.grad.float()).abs().max().item()
                    for o, r in ((qo, qr), (ko, kr), (vo, vr))]
            tol = 2e-5 * (1.0 + n / 64.0)
            ok = all(e < tol for e in errs)
            fails += 0 if ok else 1
            print(f"  grad vs f64  B={b} H={h} N={n:<4} d={d:<3} "
                  f"causal={int(causal)}  {'ok  ' if ok else 'FAIL'}"
                  f"  dQ {errs[0]:.2e}  dK {errs[1]:.2e}  dV {errs[2]:.2e}")
    return fails


def test_training_loop(device):
    # Fit q/k/v so attention output matches a target, and the target is
    # reachable (it is the op's own output at a true q*/k*/v*), so a
    # correct gradient has to drive the loss down. Optimizing toward a
    # random target instead would be ill-posed: attention output is a
    # convex combination of V rows, so most targets are unreachable and
    # the loss would plateau for reasons that say nothing about the grad.
    b, h, n, d = 1, 2, 64, 64
    qt, kt, vt = rand_qkv(b, h, n, d, device, torch.float32)
    target = flash_attention(qt, kt, vt).detach()

    params = [(t + 0.3 * torch.randn_like(t)).requires_grad_(True)
              for t in (qt, kt, vt)]
    opt = torch.optim.Adam(params, lr=5e-2)

    first = last = None
    for step in range(300):
        opt.zero_grad()
        loss = F.mse_loss(flash_attention(*params), target)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    ok = last < first * 0.05
    print(f"  train loop   loss {first:.4f} -> {last:.4f}  "
          f"({first / max(last, 1e-12):.0f}x)  {'ok' if ok else 'FAIL'}")
    return 0 if ok else 1


def run(device):
    print(f"[{device}]")
    fails = 0
    fails += test_gradcheck(device)
    fails += test_grad_parity(device)
    fails += test_training_loop(device)
    return fails


if __name__ == "__main__":
    fails = run("cpu")
    if torch.cuda.is_available():
        fails += run("cuda")
    print("FAILED" if fails else "all passed")
    sys.exit(1 if fails else 0)
