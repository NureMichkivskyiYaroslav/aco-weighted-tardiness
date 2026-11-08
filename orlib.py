"""Reader for the OR-Library weighted tardiness sets (wt40, wt50, wt100).

Each file holds 125 instances one after another: n processing times,
n weights, n due dates. Optimal / best-known values: wtopt40, wtopt50,
wtbest100b (one value per instance, same order).
Source: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/wtinfo.html
"""
from pathlib import Path

import numpy as np

from aco_twt import Instance

DATA = Path(__file__).resolve().parent / "data"
BEST_FILE = {40: "wtopt40.txt", 50: "wtopt50.txt", 100: "wtbest100b.txt"}

# RDD and TF of instance k (1-based): 5 instances per (RDD, TF) pair; RDD is
# the outer loop (every 25 instances), TF the inner one (every 5) - checked
# against the data: TF ~ 1 - mean(d)/P, RDD ~ (max d - min d)/P.
RDD_VALUES = (0.2, 0.4, 0.6, 0.8, 1.0)
TF_VALUES = (0.2, 0.4, 0.6, 0.8, 1.0)


def load(n: int) -> list[Instance]:
    nums = np.array((DATA / f"wt{n}.txt").read_text().split(), dtype=int)
    best = np.array((DATA / BEST_FILE[n]).read_text().split(), dtype=int)
    block = 3 * n
    out = []
    for k in range(len(nums) // block):
        chunk = nums[k * block:(k + 1) * block]
        out.append(Instance(p=chunk[:n], w=chunk[n:2 * n], d=chunk[2 * n:],
                            name=f"wt{n}-{k + 1:03d}", best_known=int(best[k])))
    return out


def classes(k: int) -> tuple[float, float]:
    """(TF, RDD) of 1-based instance k, inferred from the generator order."""
    group = (k - 1) // 5            # 0..24
    return TF_VALUES[group % 5], RDD_VALUES[group // 5]
