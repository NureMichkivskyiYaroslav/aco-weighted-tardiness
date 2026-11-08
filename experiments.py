"""All experiments of the lab. Results go to results/*.csv / *.json / *.png.

    python experiments.py small      # tiny instances vs brute force
    python experiments.py params     # parameter study (one factor at a time)
    python experiments.py bench      # OR-Library wt40 / wt50 / wt100
    python experiments.py conv       # convergence curves
    python experiments.py all
"""
from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

import orlib
from aco_twt import Instance, Params, acs, brute_force, greedy, twt, schedule

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
WORKERS = max(1, (os.cpu_count() or 2) - 2)


def dev(v, best):
    if best == 0:
        return 0.0 if v == 0 else float("inf")
    return 100.0 * (v - best) / best


# --------------------------------------------------------------------------- #
# 1. Small hand-made instances, checked by full enumeration
# --------------------------------------------------------------------------- #
def small_instances() -> list[Instance]:
    L = lambda s: list(s)
    return [
        # example from the individual task (topic 11)
        Instance(np.array([2, 4, 1]), np.array([5, 1, 3]), np.array([3, 6, 2]),
                 "task-example (A,B,C)", labels=L("ABC")),
        # everything can be on time -> optimum 0
        Instance(np.array([2, 3, 1, 4]), np.array([1, 2, 3, 1]), np.array([3, 6, 7, 10]),
                 "all-on-time", labels=L("ABCD")),
        # all jobs are late anyway -> pure w/p ordering (WSPT) is optimal
        Instance(np.array([4, 2, 6, 3, 5]), np.array([1, 4, 3, 2, 5]), np.array([0, 0, 0, 0, 0]),
                 "all-late (WSPT)", labels=L("ABCDE")),
        # conflicting weights and due dates, n = 8
        Instance(np.array([5, 9, 3, 7, 2, 6, 4, 8]), np.array([3, 1, 4, 2, 5, 2, 3, 1]),
                 np.array([10, 30, 8, 20, 6, 25, 15, 40]), "mixed n=8", labels=L("ABCDEFGH")),
        # tight due dates, heavy weights on long jobs, n = 9
        Instance(np.array([8, 3, 6, 2, 9, 4, 7, 5, 1]), np.array([9, 1, 6, 2, 10, 3, 7, 4, 1]),
                 np.array([12, 9, 15, 6, 20, 10, 18, 14, 5]), "tight n=9", labels=L("ABCDEFGHI")),
    ]


def run_small():
    rows = []
    for inst in small_instances():
        opt_perm, opt = brute_force(inst)
        r = acs(inst, Params(ants=10, iterations=50, seed=1))
        s, c, t = schedule(inst, r.best_perm)
        rows.append({
            "instance": inst.name, "n": inst.n,
            "p": inst.p.tolist(), "w": inst.w.tolist(), "d": inst.d.tolist(),
            "greedy_atc": " ".join(inst.label(j) for j in greedy(inst, "atc")),
            "greedy_value": twt(inst, greedy(inst, "atc")),
            "aco_order": " ".join(inst.label(j) for j in r.best_perm),
            "aco_value": r.best_value, "found_at_iter": r.found_at,
            "optimum_order": " ".join(inst.label(j) for j in opt_perm),
            "optimum": opt,
            "C": c.tolist(), "T": t.tolist(),
        })
        print(f"{inst.name:24s} ACO {r.best_value:5d}  optimum {opt:5d}  "
              f"order {rows[-1]['aco_order']}  (iter {r.found_at})")
    (OUT / "small.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


# --------------------------------------------------------------------------- #
# 2. Parameter study
# --------------------------------------------------------------------------- #
PARAM_INSTANCES = [("wt40", 40, 1), ("wt40", 40, 60), ("wt40", 40, 87), ("wt100", 100, 10)]
BASE = Params(ants=20, iterations=200, alpha=1.0, beta=2.0, q0=0.9, rho=0.1, xi=0.1,
              heuristic="atc", sum_rule=True, local_search=False)
GRID = {
    "q0": [0.0, 0.5, 0.75, 0.9, 0.98],
    "beta": [0.0, 1.0, 2.0, 3.0, 5.0],
    "rho": [0.01, 0.05, 0.1, 0.3, 0.6],
    "ants": [5, 10, 20, 40],
    "heuristic": ["none", "edd", "mdd", "wmdd", "atc"],
    "sum_rule": [False, True],
    "local_search": [False, True],
}
SEEDS = range(10)


def _job(args):
    (n, k, factor, value, seed) = args
    inst = orlib.load(n)[k - 1]
    prm = replace(BASE, **{factor: value}, seed=seed)
    if n == 100:
        prm = replace(prm, iterations=150)
    r = acs(inst, prm)
    return {"instance": inst.name, "factor": factor, "value": value, "seed": seed,
            "best": r.best_value, "best_known": inst.best_known,
            "dev": dev(r.best_value, inst.best_known), "found_at": r.found_at,
            "seconds": round(r.seconds, 3)}


def run_params():
    jobs = [(n, k, f, v, s) for (_, n, k) in PARAM_INSTANCES for f, vals in GRID.items()
            for v in vals for s in SEEDS]
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(_job, jobs, chunksize=4))
    _write_csv(OUT / "params_raw.csv", rows)
    # aggregate
    agg = []
    for (_, n, k) in PARAM_INSTANCES:
        name = f"wt{n}-{k:03d}"
        for f, vals in GRID.items():
            for v in vals:
                sel = [r for r in rows if r["instance"] == name and r["factor"] == f and r["value"] == v]
                d = np.array([r["dev"] for r in sel])
                agg.append({"instance": name, "factor": f, "value": v,
                            "mean_dev": round(d.mean(), 3), "min_dev": round(d.min(), 3),
                            "max_dev": round(d.max(), 3),
                            "hits": int(sum(r["best"] <= r["best_known"] for r in sel)),
                            "runs": len(sel),
                            "mean_iter": round(np.mean([r["found_at"] for r in sel]), 1),
                            "mean_sec": round(np.mean([r["seconds"] for r in sel]), 3)})
    _write_csv(OUT / "params.csv", agg)
    _plot_params(agg)
    print("params done:", len(rows), "runs")


def _plot_params(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    insts = sorted({a["instance"] for a in agg})
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, f in zip(axes.flat, GRID):
        vals = GRID[f]
        x = np.arange(len(vals))
        width = 0.8 / len(insts)
        for i, name in enumerate(insts):
            y = [next(a["mean_dev"] for a in agg if a["instance"] == name and a["factor"] == f and a["value"] == v)
                 for v in vals]
            ax.bar(x + i * width - 0.4 + width / 2, y, width, label=name)
        ax.set_xticks(x, [str(v) for v in vals])
        ax.set_title(f"параметр: {f}")
        ax.set_ylabel("сер. відхилення, %")
        ax.grid(axis="y", alpha=.3)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Вплив параметрів ACS (10 запусків на конфігурацію, решта параметрів базові)")
    fig.tight_layout()
    fig.savefig(OUT / "params.png", dpi=120)


# --------------------------------------------------------------------------- #
# 2b. Stage 2: combinations of the values that were best one at a time
# --------------------------------------------------------------------------- #
COMBOS = {
    "A: база (atc, сум., q0=.9, β=2)": {},
    "B: wmdd": {"heuristic": "wmdd"},
    "C: wmdd, без сум.": {"heuristic": "wmdd", "sum_rule": False},
    "D: wmdd, без сум., q0=.75": {"heuristic": "wmdd", "sum_rule": False, "q0": 0.75},
    "E: wmdd, без сум., q0=.75, ρ=.3": {"heuristic": "wmdd", "sum_rule": False, "q0": 0.75, "rho": 0.3},
    "F: wmdd, без сум., β=1, ρ=.3": {"heuristic": "wmdd", "sum_rule": False, "beta": 1.0, "rho": 0.3},
    "G: atc, без сум., β=1, ρ=.3": {"heuristic": "atc", "sum_rule": False, "beta": 1.0, "rho": 0.3},
}


def _combo_job(args):
    n, k, label, seed = args
    inst = orlib.load(n)[k - 1]
    prm = replace(BASE, **COMBOS[label], seed=seed, iterations=200 if n < 100 else 150)
    r = acs(inst, prm)
    return {"instance": inst.name, "combo": label, "seed": seed, "best": r.best_value,
            "best_known": inst.best_known, "dev": dev(r.best_value, inst.best_known),
            "seconds": round(r.seconds, 3)}


def run_tune():
    jobs = [(n, k, c, s) for (_, n, k) in PARAM_INSTANCES for c in COMBOS for s in SEEDS]
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(_combo_job, jobs, chunksize=4))
    _write_csv(OUT / "tune_raw.csv", rows)
    agg = []
    for c in COMBOS:
        row = {"combo": c}
        for (_, n, k) in PARAM_INSTANCES:
            name = f"wt{n}-{k:03d}"
            row[name] = round(float(np.mean([r["dev"] for r in rows if r["combo"] == c and r["instance"] == name])), 3)
        row["mean"] = round(float(np.mean([row[f"wt{n}-{k:03d}"] for (_, n, k) in PARAM_INSTANCES])), 3)
        agg.append(row)
        print(row)
    _write_csv(OUT / "tune.csv", agg)


# --------------------------------------------------------------------------- #
# 3. OR-Library benchmark
# --------------------------------------------------------------------------- #
TUNED = replace(BASE, heuristic="wmdd", sum_rule=False)    # combo C, best mean deviation
FINAL = replace(TUNED, local_search=True)
BENCH_SEEDS = range(3)


VARIANTS = {"ACS": None, "ACS+LS (обмін)": "swap", "ACS+LS (обмін+вставка)": "swap+insert"}


def _bench_job(args):
    n, k, variant, seed = args
    inst = orlib.load(n)[k - 1]
    moves = VARIANTS[variant]
    prm = replace(FINAL, local_search=moves is not None, ls_moves=moves or "swap", seed=seed,
                  iterations=200 if n < 100 else 150)
    r = acs(inst, prm, target=inst.best_known if moves else None)
    tf, rdd = orlib.classes(k)
    return {"instance": inst.name, "n": n, "k": k, "TF": tf, "RDD": rdd, "variant": variant,
            "seed": seed, "greedy": twt(inst, greedy(inst, "wmdd")), "best": r.best_value,
            "best_known": inst.best_known, "dev": dev(r.best_value, inst.best_known),
            "hit": r.best_value <= inst.best_known, "found_at": r.found_at,
            "seconds": round(r.seconds, 3)}


def run_bench(variants=tuple(VARIANTS)):
    """Runs the given variants and merges them into results/bench_raw.csv."""
    jobs = [(n, k, v, s) for n in (40, 50, 100) for k in range(1, 126)
            for v in variants for s in BENCH_SEEDS]
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(_bench_job, jobs, chunksize=2))
    raw = OUT / "bench_raw.csv"
    if raw.exists():
        with open(raw, encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f) if r["variant"] not in variants]
        rows = old + rows
    _write_csv(raw, rows)
    summarize_bench()


def _stats(sel, all_sel):
    f = lambda r, key: float(r[key])
    devs = [f(r, "dev") for r in sel]
    return {"runs": len(all_sel),
            "greedy_mean_dev": round(float(np.mean([dev(f(r, "greedy"), f(r, "best_known")) for r in sel])), 2),
            "mean_dev": round(float(np.mean(devs)), 3),
            "median_dev": round(float(np.median(devs)), 3),
            "total_dev": round(100 * sum(f(r, "best") - f(r, "best_known") for r in sel)
                               / sum(f(r, "best_known") for r in sel), 3),
            "max_dev": round(float(np.max(devs)), 2),
            "hit_rate_%": round(100 * np.mean([str(r["hit"]) == "True" for r in all_sel]), 1),
            "mean_sec": round(float(np.mean([f(r, "seconds") for r in all_sel])), 2)}


def summarize_bench():
    with open(OUT / "bench_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summ = []
    for n in ("40", "50", "100"):
        for v in VARIANTS:
            all_sel = [r for r in rows if r["n"] == n and r["variant"] == v]
            if not all_sel:
                continue
            sel = [r for r in all_sel if float(r["best_known"]) > 0]
            summ.append({"set": f"wt{n}", "variant": v, **_stats(sel, all_sel)})
            for tf in orlib.TF_VALUES:
                a2 = [r for r in all_sel if float(r["TF"]) == tf]
                s2 = [r for r in a2 if float(r["best_known"]) > 0]
                summ.append({"set": f"wt{n} TF={tf}", "variant": v, **_stats(s2, a2)})
    _write_csv(OUT / "bench.csv", summ)
    for s in summ:
        if "TF" not in s["set"]:
            print(s)


# --------------------------------------------------------------------------- #
# 4. Convergence curves
# --------------------------------------------------------------------------- #
def run_conv():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    inst = orlib.load(100)[10 - 1]
    variants = {
        "A: базова конфігурація (atc, сумування)": BASE,
        "C: налаштована (wmdd, без сумування)": TUNED,
        "C, q0=0.5 (більше дослідження)": replace(TUNED, q0=0.5),
        "C, β=0 (без евристики)": replace(TUNED, beta=0.0),
        "C + локальний пошук": FINAL,
    }
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, prm in variants.items():
        curves = [acs(inst, replace(prm, iterations=150, seed=s)).history_best for s in range(5)]
        y = np.mean([[dev(v, inst.best_known) for v in c] for c in curves], axis=0)
        ax.plot(np.arange(1, len(y) + 1), y, label=label)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("ітерація")
    ax.set_ylabel("відхилення найкращого від best-known, % (середнє з 5 запусків)")
    ax.set_title(f"Збіжність ACS на {inst.name} (n=100, best-known {inst.best_known})")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=120)
    print("conv done")


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("small", "all"):
        run_small()
    if what in ("params", "all"):
        run_params()
    if what in ("tune", "all"):
        run_tune()
    if what in ("bench", "all"):
        run_bench()
    if what == "bench-insert":
        run_bench(("ACS+LS (обмін+вставка)",))
    if what in ("conv", "all"):
        run_conv()
