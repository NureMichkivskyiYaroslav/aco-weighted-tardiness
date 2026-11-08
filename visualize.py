"""Visualisation of ACS for 1||sum wT while it runs.

    python visualize.py                       # live window, wt40 instance 60
    python visualize.py --set 40 --k 1 --gif results/aco_wt40-001.gif
    python visualize.py --small               # task example (A,B,C) + mixed n=8
    python visualize.py --png results/frame.png --frame 25

Panels: (1) pheromone matrix tau[position, job] (columns in the order of the
final best schedule, so convergence shows up as a bright diagonal);
(2) Gantt chart of the best-so-far schedule, tardy jobs in red, due dates as
ticks; (3) convergence: best-so-far / iteration-best / iteration-mean TWT;
(4) TWT of every ant of the current iteration.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import orlib
from aco_twt import Params, acs, schedule
from experiments import small_instances


def record(inst, prm: Params):
    frames = []

    def cb(it, tau, best_perm, best_val, vals):
        frames.append((it, tau.copy(), list(best_perm), int(best_val), vals.copy()))

    res = acs(inst, prm, callback=cb)
    return res, frames


def build(inst, res, frames):
    order = np.asarray(res.best_perm)              # column order for the heatmap
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.6, 1], height_ratios=[1, 1])
    ax_tau = fig.add_subplot(gs[:, 0])
    ax_gantt = fig.add_subplot(gs[0, 1:])
    ax_conv = fig.add_subplot(gs[1, 1])
    ax_ants = fig.add_subplot(gs[1, 2])

    n = inst.n
    tau0 = frames[0][1]
    im = ax_tau.imshow(tau0[:, order], aspect="auto", cmap="viridis", interpolation="nearest")
    ax_tau.set_title("Феромон τ[позиція, робота]")
    ax_tau.set_xlabel("робота (у порядку фінального розкладу)")
    ax_tau.set_ylabel("позиція в розкладі")
    if n <= 12:
        ax_tau.set_xticks(range(n), [inst.label(j) for j in order])
        ax_tau.set_yticks(range(n), [str(i + 1) for i in range(n)])
    best_dots, = ax_tau.plot([], [], "o", ms=4 if n > 12 else 9, mfc="none", mec="red", mew=1.2)
    fig.colorbar(im, ax=ax_tau, fraction=0.046)

    hist_best = [f[3] for f in frames]
    hist_it = [int(f[4].min()) for f in frames]
    hist_mean = [float(f[4].mean()) for f in frames]
    its = [f[0] for f in frames]
    ax_conv.set_title("Збіжність")
    ax_conv.set_xlabel("ітерація")
    ax_conv.set_ylabel("TWT")
    ax_conv.set_xlim(1, its[-1])
    lo, hi = min(hist_best), max(hist_mean)
    if inst.best_known is not None:
        lo = min(lo, inst.best_known)
    ax_conv.set_ylim(lo - 0.05 * (hi - lo + 1), hi + 0.05 * (hi - lo + 1))
    l_best, = ax_conv.plot([], [], lw=2, label="найкраще")
    l_it, = ax_conv.plot([], [], lw=1, label="найкраще ітерації")
    l_mean, = ax_conv.plot([], [], lw=1, alpha=.6, label="середнє ітерації")
    if inst.best_known is not None:
        ax_conv.axhline(inst.best_known, ls="--", c="gray", lw=1, label=f"best-known {inst.best_known}")
    ax_conv.legend(fontsize=8)
    ax_conv.grid(alpha=.3)

    title = fig.suptitle("")
    P = int(inst.p.sum())

    def draw_gantt(perm, val):
        ax_gantt.clear()
        s, c, t = schedule(inst, perm)
        colors = ["#d62728" if x > 0 else "#2ca02c" for x in t]
        ax_gantt.barh(0, inst.p[perm], left=s, color=colors, edgecolor="white", height=0.6)
        for pos, j in enumerate(perm):
            ax_gantt.plot([inst.d[j]] * 2, [0.35, 0.55], c="black", lw=0.8)
            if n <= 50:
                ax_gantt.text(s[pos] + inst.p[j] / 2, 0, inst.label(j), ha="center", va="center",
                              fontsize=8 if n <= 12 else 5, color="white")
        ax_gantt.set_xlim(0, max(P, int(inst.d.max())) * 1.02)
        ax_gantt.set_ylim(-0.5, 0.8)
        ax_gantt.set_yticks([])
        ax_gantt.set_xlabel("час")
        late = int((t > 0).sum())
        ax_gantt.set_title(f"Найкращий розклад: TWT = {val}, із запізненням {late}/{n}\n"
                           "зелений: вчасно, червоний: із запізненням, |: дедлайни", fontsize=10)

    def update(k):
        it, tau, perm, val, vals = frames[k]
        im.set_data(tau[:, order])
        im.set_clim(tau.min(), tau.max())
        col = {j: c for c, j in enumerate(order)}
        best_dots.set_data([col[j] for j in perm], list(range(n)))
        draw_gantt(perm, val)
        l_best.set_data(its[:k + 1], hist_best[:k + 1])
        l_it.set_data(its[:k + 1], hist_it[:k + 1])
        l_mean.set_data(its[:k + 1], hist_mean[:k + 1])
        ax_ants.clear()
        ax_ants.bar(range(1, len(vals) + 1), vals, color="#1f77b4")
        ax_ants.axhline(val, c="red", lw=1, label="найкраще")
        ax_ants.set_title(f"TWT кожної мурахи на ітерації {it}")
        ax_ants.set_xlabel("мураха")
        ax_ants.set_ylim(min(lo, vals.min()) * 0.98, max(vals.max(), lo + 1) * 1.02)
        gap = "" if not inst.best_known else f", відхилення {100 * (val - inst.best_known) / inst.best_known:.2f}%"
        title.set_text(f"ACS для 1||ΣwT, {inst.name} (n={n}), ітерація {it}/{its[-1]}, "
                       f"найкраще TWT {val}{gap}")
        return im,

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, default=40, choices=(40, 50, 100))
    ap.add_argument("--k", type=int, default=60, help="instance number 1..125")
    ap.add_argument("--small", type=int, nargs="?", const=0, default=None,
                    help="index of a small hand-made instance (0 = task example)")
    ap.add_argument("--iterations", type=int, default=120)
    ap.add_argument("--ls", action="store_true", help="enable local search")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--gif", help="save animation as GIF instead of showing a window")
    ap.add_argument("--png", help="save one frame as PNG")
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--interval", type=int, default=120, help="ms between frames")
    a = ap.parse_args()

    if a.gif or a.png:
        matplotlib.use("Agg")
    if a.small is not None:
        inst = small_instances()[a.small]
        inst.best_known = None
        prm = Params(ants=8, iterations=min(a.iterations, 40), seed=a.seed, q0=0.5)
    else:
        inst = orlib.load(a.set)[a.k - 1]
        prm = Params(iterations=a.iterations, seed=a.seed, local_search=a.ls)
    res, frames = record(inst, prm)
    fig, update = build(inst, res, frames)
    if a.png:
        update(a.frame if a.frame >= 0 else len(frames) - 1)
        fig.savefig(a.png, dpi=110)
        return
    anim = FuncAnimation(fig, update, frames=len(frames), interval=a.interval, blit=False, repeat=False)
    if a.gif:
        anim.save(a.gif, writer=PillowWriter(fps=max(1, 1000 // a.interval)), dpi=70)
    else:
        plt.show()


if __name__ == "__main__":
    main()
