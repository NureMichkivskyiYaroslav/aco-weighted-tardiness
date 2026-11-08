# ACO for Total Weighted Tardiness (1||ΣwⱼTⱼ)

Lab work #1, course "Models and Algorithms of Swarm Intelligence" (KhNURE, master's, 121 Software Engineering).
Ant Colony System (ACS) for the single-machine total weighted tardiness problem, tested on the OR-Library sets wt40 / wt50 / wt100.

## Problem

n jobs, each with processing time pⱼ, weight wⱼ and due date dⱼ, are processed one at a time on a single machine.
Find a processing order π minimising `TWT(π) = Σ wⱼ · max(0, Cⱼ − dⱼ)`, where Cⱼ is the completion time of job j.

## Model (problem → ACO)

| ACO element | Here |
|---|---|
| solution | permutation of jobs, built position by position |
| pheromone | `τ[i, j]`: job j at position i; summation rule `Σ_{k≤i} τ[k, j]` (Merkle & Middendorf) |
| heuristic η | dispatching rule at the current time T: WMDD (default), ATC, MDD, EDD or none |
| choice | ACS pseudo-random proportional rule: argmax with prob. q0, else roulette over τ^α·η^β |
| local update | `τ ← (1−ξ)τ + ξτ₀` on every chosen (position, job) |
| global update | best-so-far ant only: `τ ← (1−ρ)τ + ρ/(1+TWT_best)` |
| local search (optional) | best-improvement descent on the iteration-best ant: pairwise interchange, or interchange + insertion (`ls_moves="swap+insert"`, the final variant) |

Final configuration (from the parameter study): m = 20, α = 1, β = 2, q0 = 0.9, ρ = ξ = 0.1, heuristic WMDD, no summation rule, + local search (interchange + insertion).
On all 375 OR-Library instances: median deviation 0-0.003 %, total deviation 0.03-0.07 %, optimum reached in 81 % (wt40), 73 % (wt50), 53 % (wt100) of runs.

## Files

| File | What |
|---|---|
| `aco_twt.py` | problem, heuristics, greedy rule, local search, ACS |
| `orlib.py` | reader of the OR-Library files in `data/` |
| `experiments.py` | `small` (brute-force check), `params` (parameter study), `bench` (wt40/50/100), `conv` (convergence) |
| `visualize.py` | animation of a run: pheromone matrix, Gantt chart, convergence, ants of the iteration |
| `results/` | CSV / PNG / GIF produced by the scripts |

## Run

```bash
pip install -r requirements.txt
python experiments.py all                 # ~10-15 min on 16+ cores
python visualize.py --set 40 --k 60       # live window
python visualize.py --small 4 --gif results/aco_small_tight9.gif
```

Data: J. E. Beasley, OR-Library, http://people.brunel.ac.uk/~mastjjb/jeb/orlib/wtinfo.html
