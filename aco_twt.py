"""Ant Colony System (ACS) for the single-machine total weighted tardiness
problem 1||sum w_j T_j.

Model (problem -> ACO):
  * solution      - a permutation pi of the n jobs (processing order);
  * construction  - an ant fills positions 1..n one by one, at each step it
                    picks one of the jobs not yet scheduled;
  * pheromone     - tau[i, j]: desirability of putting job j at position i;
                    with the summation rule (Merkle & Middendorf) an ant at
                    position i uses sum_{k<=i} tau[k, j], i.e. "job j should
                    be scheduled no later than i" - pheromone effectively
                    encodes the relation "j before the jobs placed after it";
  * heuristic eta - a dispatching rule evaluated at the current time T
                    (default: weighted modified due date, WMDD);
  * quality       - TWT(pi) = sum_j w_j * max(0, C_j - d_j), minimised.

ACS specifics (Dorigo & Gambardella 1997; den Besten, Stuetzle & Dorigo 2000):
  pseudo-random proportional rule with q0, local pheromone update after every
  step, global update by the best-so-far ant only, optional local search
  (pairwise interchange) applied to the iteration-best ant.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field, asdict

import numpy as np


# --------------------------------------------------------------------------- #
# Problem
# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    p: np.ndarray            # processing times
    w: np.ndarray            # weights
    d: np.ndarray            # due dates
    name: str = "instance"
    best_known: int | None = None
    labels: list[str] | None = None

    @property
    def n(self) -> int:
        return len(self.p)

    def label(self, j: int) -> str:
        return self.labels[j] if self.labels else f"J{j + 1}"


def twt(inst: Instance, perm) -> int:
    """Total weighted tardiness of one permutation."""
    perm = np.asarray(perm)
    c = np.cumsum(inst.p[perm])
    return int(np.sum(inst.w[perm] * np.maximum(0, c - inst.d[perm])))


def twt_batch(inst: Instance, perms: np.ndarray) -> np.ndarray:
    """TWT of many permutations at once, perms has shape (k, n)."""
    c = np.cumsum(inst.p[perms], axis=1)
    return np.sum(inst.w[perms] * np.maximum(0, c - inst.d[perms]), axis=1)


def schedule(inst: Instance, perm):
    """Start, completion and tardiness of every job in processing order."""
    perm = np.asarray(perm)
    c = np.cumsum(inst.p[perm])
    s = c - inst.p[perm]
    t = np.maximum(0, c - inst.d[perm])
    return s, c, t


def brute_force(inst: Instance):
    """Exact optimum by full enumeration (only for n <= 9)."""
    assert inst.n <= 9, "brute force is only for tiny instances"
    perms = np.array(list(itertools.permutations(range(inst.n))))
    vals = twt_batch(inst, perms)
    k = int(np.argmin(vals))
    return perms[k].tolist(), int(vals[k])


# --------------------------------------------------------------------------- #
# Heuristic information eta(T, j)
# --------------------------------------------------------------------------- #
def heuristic(inst: Instance, kind: str, T: np.ndarray) -> np.ndarray:
    """eta for every ant (rows, current time T[a]) and every job (cols)."""
    p, w, d = inst.p[None, :], inst.w[None, :], inst.d[None, :]
    T = T[:, None]
    if kind == "none":
        return np.ones((T.shape[0], inst.n))
    if kind == "edd":        # earliest due date
        return np.broadcast_to(1.0 / np.maximum(d, 1), (T.shape[0], inst.n)).copy()
    if kind == "mdd":        # modified due date (unweighted)
        return 1.0 / np.maximum(T + p, d)
    if kind == "wmdd":       # weighted modified due date (Kanet & Li)
        return w / np.maximum(p, d - T)
    if kind == "atc":        # apparent tardiness cost, k = 2
        pbar = inst.p.mean()
        return (w / p) * np.exp(-np.maximum(d - p - T, 0) / (2.0 * pbar))
    raise ValueError(f"unknown heuristic {kind!r}")


def greedy(inst: Instance, kind: str = "wmdd") -> list[int]:
    """Dispatching rule: always take the job with the best eta."""
    left = np.ones(inst.n, bool)
    T, perm = 0, []
    for _ in range(inst.n):
        eta = heuristic(inst, kind, np.array([T], float))[0]
        eta[~left] = -np.inf
        j = int(np.argmax(eta))
        perm.append(j)
        left[j] = False
        T += inst.p[j]
    return perm


# --------------------------------------------------------------------------- #
# Local search: best-improvement pairwise interchange (vectorised)
# --------------------------------------------------------------------------- #
def _swap_pairs(n: int):
    i, j = np.triu_indices(n, 1)
    return i, j


_INSERT_CACHE: dict[int, np.ndarray] = {}


def _insert_index(n: int) -> np.ndarray:
    """Rows of position indices: job at position i moved to position k (i != k)."""
    if n not in _INSERT_CACHE:
        rows = []
        base = np.arange(n)
        for i in range(n):
            rest = np.delete(base, i)
            for k in range(n):
                if k != i:
                    rows.append(np.insert(rest, k, i))
        _INSERT_CACHE[n] = np.array(rows)
    return _INSERT_CACHE[n]


def local_search(inst: Instance, perm, max_rounds: int = 50, moves: str = "swap"):
    """Best-improvement descent. moves: 'swap' (pairwise interchange) or
    'swap+insert' (also moves one job to another position)."""
    perm = np.asarray(perm).copy()
    best = twt(inst, perm)
    ii, jj = _swap_pairs(inst.n)
    rows = np.arange(len(ii))
    ins = _insert_index(inst.n) if moves == "swap+insert" else None
    for _ in range(max_rounds):
        cand = np.tile(perm, (len(ii), 1))
        cand[rows, ii], cand[rows, jj] = perm[jj], perm[ii]
        if ins is not None:
            cand = np.vstack([cand, perm[ins]])
        vals = twt_batch(inst, cand)
        k = int(np.argmin(vals))
        if vals[k] >= best:
            break
        perm, best = cand[k], int(vals[k])
    return perm, best


# --------------------------------------------------------------------------- #
# ACS
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    ants: int = 20
    iterations: int = 300
    alpha: float = 1.0
    beta: float = 2.0
    q0: float = 0.9          # exploitation probability
    rho: float = 0.1         # global evaporation
    xi: float = 0.1          # local evaporation
    heuristic: str = "wmdd"       # best in the parameter study (results/tune.csv)
    sum_rule: bool = False
    local_search: bool = False
    ls_moves: str = "swap"            # 'swap' or 'swap+insert'
    seed: int | None = None
    time_limit: float | None = None   # seconds, optional
    snapshot_every: int = 0           # store pheromone matrix every k iterations (for visualisation)


@dataclass
class Result:
    best_perm: list[int]
    best_value: int
    found_at: int                     # iteration where best was found
    seconds: float
    history_best: list[int] = field(default_factory=list)
    history_iter_best: list[int] = field(default_factory=list)
    history_iter_mean: list[float] = field(default_factory=list)
    snapshots: list[tuple[int, np.ndarray, list[int], int]] = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("snapshots")
        return d


def acs(inst: Instance, prm: Params, target: int | None = None, callback=None) -> Result:
    rng = np.random.default_rng(prm.seed)
    n, m = inst.n, prm.ants

    g = greedy(inst, prm.heuristic if prm.heuristic != "none" else "wmdd")
    g_val = twt(inst, g)
    tau0 = 1.0 / (n * (1.0 + g_val))
    tau = np.full((n, n), tau0)

    best_perm, best_val, found_at = list(g), g_val, 0
    res = Result(best_perm, best_val, 0, 0.0)
    t_start = time.perf_counter()

    for it in range(1, prm.iterations + 1):
        perms = np.empty((m, n), dtype=int)
        left = np.ones((m, n), bool)
        T = np.zeros(m)
        ants = np.arange(m)
        for i in range(n):
            tau_eff = tau[: i + 1].sum(0) if prm.sum_rule else tau[i]
            eta = heuristic(inst, prm.heuristic, T)
            score = (tau_eff[None, :] ** prm.alpha) * (eta ** prm.beta)
            score = np.where(left, score, 0.0)
            # pseudo-random proportional rule
            greedy_pick = np.argmax(score, axis=1)
            cum = np.cumsum(score, axis=1)
            r = rng.random(m) * cum[:, -1]
            roulette_pick = np.minimum((cum < r[:, None]).sum(1), n - 1)
            exploit = rng.random(m) < prm.q0
            pick = np.where(exploit, greedy_pick, roulette_pick)
            # numerical safety: never pick an already scheduled job
            bad = ~left[ants, pick]
            if bad.any():
                pick[bad] = greedy_pick[bad]
            perms[:, i] = pick
            left[ants, pick] = False
            T += inst.p[pick]
            # ACS local pheromone update
            u = np.unique(pick)
            tau[i, u] = (1 - prm.xi) * tau[i, u] + prm.xi * tau0

        vals = twt_batch(inst, perms)
        k = int(np.argmin(vals))
        it_perm, it_val = perms[k], int(vals[k])
        if prm.local_search:
            it_perm, it_val = local_search(inst, it_perm, moves=prm.ls_moves)
        if it_val < best_val:
            best_perm, best_val, found_at = it_perm.tolist(), it_val, it

        # ACS global update: only the best-so-far ant deposits
        pos = np.arange(n)
        bp = np.asarray(best_perm)
        tau[pos, bp] = (1 - prm.rho) * tau[pos, bp] + prm.rho / (1.0 + best_val)

        res.history_best.append(best_val)
        res.history_iter_best.append(it_val)
        res.history_iter_mean.append(float(vals.mean()))
        if prm.snapshot_every and (it == 1 or it % prm.snapshot_every == 0):
            res.snapshots.append((it, tau.copy(), list(best_perm), best_val))
        if callback:
            callback(it, tau, best_perm, best_val, vals)

        if target is not None and best_val <= target:
            break
        if prm.time_limit and time.perf_counter() - t_start > prm.time_limit:
            break

    res.best_perm, res.best_value, res.found_at = list(map(int, best_perm)), int(best_val), found_at
    res.seconds = time.perf_counter() - t_start
    if prm.snapshot_every and (not res.snapshots or res.snapshots[-1][0] != it):
        res.snapshots.append((it, tau.copy(), list(best_perm), best_val))
    return res
