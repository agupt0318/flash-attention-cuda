#!/usr/bin/env python3
"""Render assets/convergence.gif: one query row streaming through K/V
tiles under the online-softmax recurrence (the exact update flash_cpu
and the CUDA kernel run), showing the attention weights as they are
revealed and the partial output converging to the exact answer.
Deterministic; regenerate with  python3 tools/make_convergence_gif.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

N, D, TILE = 256, 64, 16
rng = np.random.default_rng(7)
q = rng.standard_normal(D).astype(np.float32)
K = rng.standard_normal((N, D)).astype(np.float32)
V = rng.standard_normal((N, D)).astype(np.float32)
s = (K @ q) / np.sqrt(D)

# exact answer for the error curve
p_exact = np.exp(s - s.max()); p_exact /= p_exact.sum()
o_exact = p_exact @ V

# run the recurrence, recording state after every tile
states = []
m, l, acc = -np.inf, 0.0, np.zeros(D, np.float32)
for j0 in range(0, N, TILE):
    st = s[j0:j0 + TILE]
    mn = max(m, st.max())
    corr = np.exp(m - mn)
    l = l * corr + np.exp(st - mn).sum()
    acc = acc * corr + np.exp(st - mn) @ V[j0:j0 + TILE]
    m = mn
    w = np.zeros(N); w[:j0 + TILE] = np.exp(s[:j0 + TILE] - m) / l
    err = np.abs(acc / l - o_exact).max()
    states.append((j0 + TILE, w, m, l, err))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 3.6), dpi=100,
                               height_ratios=[1, 1.4])
fig.patch.set_facecolor("#0d1117")
for ax in (ax1, ax2):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#30363d")

def draw(f):
    n, w, m, l, err = states[min(f, len(states) - 1)]
    ax1.clear(); ax1.set_facecolor("#161b22")
    ax1.bar(np.arange(N), w, width=1.0, color="#79c0ff")
    ax1.axvspan(n - TILE, n, color="#7ee787", alpha=0.25)
    ax1.set_xlim(0, N); ax1.set_ylim(0, 0.035); ax1.set_yticks([])
    ax1.set_title(f"softmax weights after {n}/{N} keys   "
                  f"m={m:.2f}  ℓ={l:.1f}", color="#c9d1d9", fontsize=9)
    errs = [st[4] for st in states[:min(f, len(states) - 1) + 1]]
    ax2.clear(); ax2.set_facecolor("#161b22")
    ax2.semilogy(np.arange(1, len(errs) + 1) * TILE, errs, color="#7ee787")
    ax2.set_xlim(0, N); ax2.set_ylim(1e-8, 10)
    ax2.set_ylabel("‖O − O_exact‖∞", color="#8b949e", fontsize=8)
    ax2.set_xlabel("keys processed — exact at the end of the stream, no approximation",
                   color="#8b949e", fontsize=8)
    ax1.tick_params(colors="#8b949e", labelsize=7)
    ax2.tick_params(colors="#8b949e", labelsize=7)

frames = len(states) + 6                    # hold the final frame
anim = FuncAnimation(fig, draw, frames=frames, interval=350)
anim.save("assets/convergence.gif", writer=PillowWriter(fps=3),
          savefig_kwargs={"facecolor": "#0d1117"})
print("wrote assets/convergence.gif")
