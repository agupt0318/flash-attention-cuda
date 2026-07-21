"""Measure the memory FlashAttention's streaming saves, on GPT-2.

Two attention implementations run on identical [B, H, N, d] tensors:

  standard  -- materializes the full [B, H, N, N] score matrix, the way a
               textbook (and GPT-2's default "eager") attention does.
  streamed  -- FlashAttention's recurrence: process keys in tiles with an
               online softmax, never allocating [B, H, N, N]. Same algorithm
               as this repo's CUDA kernel, written in plain PyTorch so the
               memory behaviour can be measured on any device.

Part A: capture real pretrained GPT-2's q/k/v activations on a prompt and check
        the streamed output matches standard attention (exact, up to fp32).
Part B: sweep sequence length at GPT-2-small dimensions and measure peak
        attention memory for each implementation.

    python3 pytorch/gpt2_memory.py            # both parts
    python3 pytorch/gpt2_memory.py --sweep    # Part B only (no model download)

Results in the repo (Apple M-series GPU, 24 GB unified memory) are in
gpt2_memory_results.json next to this file.
"""
import argparse, math, time, gc, json
import torch


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def allocated_mb(dev):
    if dev.type == "cuda":
        return torch.cuda.memory_allocated() / 1e6
    if dev.type == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return 0.0


def sync(dev):
    if dev.type == "cuda": torch.cuda.synchronize()
    elif dev.type == "mps": torch.mps.synchronize()


def reset(dev):
    gc.collect()
    if dev.type == "cuda": torch.cuda.empty_cache()
    elif dev.type == "mps" and hasattr(torch.mps, "empty_cache"): torch.mps.empty_cache()


# ---- the two attention implementations ------------------------------------
def standard_attention(q, k, v, dev):
    B, H, N, d = q.shape
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)   # [B,H,N,N]
    mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), 1)
    scores = scores.masked_fill(mask, float("-inf"))
    peak = allocated_mb(dev)                                       # scores alive
    out = torch.matmul(torch.softmax(scores, dim=-1), v)
    return out, peak


def streamed_attention(q, k, v, dev, block=512):
    B, H, N, d = q.shape
    scale = 1.0 / math.sqrt(d)
    O = torch.zeros_like(q)
    m = torch.full((B, H, N, 1), float("-inf"), device=q.device, dtype=q.dtype)
    l = torch.zeros(B, H, N, 1, device=q.device, dtype=q.dtype)
    qi = torch.arange(N, device=q.device).view(1, 1, N, 1)
    peak = 0.0
    for j0 in range(0, N, block):
        kb = k[:, :, j0:j0 + block]
        vb = v[:, :, j0:j0 + block]
        s = torch.matmul(q, kb.transpose(-1, -2)) * scale          # [B,H,N,blk]
        kj = torch.arange(j0, j0 + kb.shape[2], device=q.device).view(1, 1, 1, -1)
        s = s.masked_fill(kj > qi, float("-inf"))
        peak = max(peak, allocated_mb(dev))
        m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
        corr = torch.exp(m - m_new)
        p = torch.exp(s - m_new)
        l = l * corr + p.sum(dim=-1, keepdim=True)
        O = O * corr + torch.matmul(p, vb)
        m = m_new
    return O / l, peak


# ---- Part A: exactness on real GPT-2 activations ---------------------------
def part_a(dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", attn_implementation="eager", dtype=torch.float32).to(dev).eval()
    cfg = model.config
    H, d = cfg.n_head, cfg.n_embd // cfg.n_head

    prompt = "The key idea behind attention is that every token"
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)

    caps = {}
    def mk(i):
        def hook(mod, inp, out):
            B, N, _ = out.shape
            q, k, v = out.split(cfg.n_embd, dim=2)
            r = lambda t: t.view(B, N, H, d).permute(0, 2, 1, 3).contiguous()
            caps[i] = (r(q), r(k), r(v))
        return hook
    hs = [model.transformer.h[i].attn.c_attn.register_forward_hook(mk(i))
          for i in range(cfg.n_layer)]
    with torch.no_grad():
        model(ids)
    for h in hs: h.remove()

    worst = 0.0
    for i in range(cfg.n_layer):
        q, k, v = caps[i]
        a, _ = standard_attention(q, k, v, dev)
        b, _ = streamed_attention(q, k, v, dev, block=128)
        worst = max(worst, (a - b).abs().max().item())
    print(f"[Part A] real GPT-2, {ids.shape[1]}-token prompt, {cfg.n_layer} layers "
          f"(H={H}, d={d}): worst |diff| standard vs streamed = {worst:.2e}")
    return {"prompt": prompt, "tokens": int(ids.shape[1]), "layers": cfg.n_layer,
            "worst_maxdiff": worst}


# ---- Part B: memory sweep at GPT-2-small dimensions ------------------------
def part_b(dev):
    H, d, B = 12, 64, 1
    rows = []
    for N in [512, 1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576]:
        row = {"N": N, "scores_matrix_mb": round(B * H * N * N * 4 / 1e6, 1)}
        q = torch.randn(B, H, N, d, device=dev)
        k = torch.randn(B, H, N, d, device=dev)
        v = torch.randn(B, H, N, d, device=dev)
        sync(dev)
        out_s = None
        try:
            reset(dev); base = allocated_mb(dev); t = time.time()
            out_s, peak = standard_attention(q, k, v, dev); sync(dev)
            row["standard_peak_mb"] = round(peak - base, 1)
            row["standard_ms"] = round((time.time() - t) * 1000, 1)
        except RuntimeError:
            row["standard_peak_mb"] = "OOM"; row["standard_ms"] = None; reset(dev)

        reset(dev); base = allocated_mb(dev); t = time.time()
        out_f, peakf = streamed_attention(q, k, v, dev, block=512); sync(dev)
        row["streamed_peak_mb"] = round(peakf - base, 1)
        row["streamed_ms"] = round((time.time() - t) * 1000, 1)
        row["maxdiff"] = (f"{(out_s - out_f).abs().max().item():.2e}"
                          if out_s is not None else "n/a (standard OOM)")
        print("[Part B] " + json.dumps(row))
        rows.append(row)
        del q, k, v, out_f
        if out_s is not None: del out_s
        reset(dev)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="Part B only (no model download)")
    args = ap.parse_args()
    dev = pick_device()
    print(f"torch {torch.__version__} | device: {dev}")
    out = {"device": str(dev), "torch": torch.__version__}
    if not args.sweep:
        out["part_a"] = part_a(dev)
    out["part_b"] = part_b(dev)
    json.dump(out, open("pytorch/gpt2_memory_results.json", "w"), indent=2)
    print("wrote pytorch/gpt2_memory_results.json")
