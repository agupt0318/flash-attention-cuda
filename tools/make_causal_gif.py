#!/usr/bin/env python3
"""Render assets/causal_tiles.gif: which K/V tiles each query tile
actually loads under causal masking — the same skip logic flash_fwd.cu
runs (kv_end = min(seq, q0 + BLOCK_M); rows break at the diagonal).
Deterministic; regenerate with  python3 tools/make_causal_gif.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

N, BM, BN = 512, 128, 64
QT, KT = N // BM, N // BN

# tile states per query tile, mirroring the kernel's loop bounds
def tiles_for(qi):
    q_last = qi * BM + BM - 1
    out = []
    for kj in range(KT):
        j0 = kj * BN
        if j0 > q_last:
            out.append("skip")              # never loaded
        elif j0 + BN - 1 <= qi * BM:
            out.append("full")              # entirely below the diagonal
        else:
            out.append("edge")              # loaded, partially masked
    return out

COLORS = {"skip": "#21262d", "full": "#238636", "edge": "#e3b341"}

fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)
fig.patch.set_facecolor("#0d1117")

def draw(f):
    qi = min(f, QT - 1)
    ax.clear()
    ax.set_facecolor("#0d1117")
    loaded = total = 0
    for row in range(qi + 1):
        st = tiles_for(row)
        for kj, s in enumerate(st):
            alpha = 1.0 if row == qi else 0.45
            ax.add_patch(plt.Rectangle((kj, QT - 1 - row), 0.92, 0.92,
                                       color=COLORS[s], alpha=alpha))
    for row in range(QT):                   # count over the full pass
        st = tiles_for(row)
        loaded += sum(s != "skip" for s in st[:KT if row <= qi else 0])
        total += KT if row <= qi else 0
    ax.set_xlim(-0.1, KT); ax.set_ylim(-0.1, QT)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"causal tiling, N={N}: query tile {qi} — K/V tiles "
                 f"loaded {loaded}/{total}", color="#c9d1d9", fontsize=10)
    ax.text(0, -0.55, "green: streamed in full   amber: streamed, diagonal-"
            "masked   dark: never leaves HBM", color="#8b949e", fontsize=8)

anim = FuncAnimation(fig, draw, frames=QT + 3, interval=900)
anim.save("assets/causal_tiles.gif", writer=PillowWriter(fps=1.2),
          savefig_kwargs={"facecolor": "#0d1117"})
print("wrote assets/causal_tiles.gif")
