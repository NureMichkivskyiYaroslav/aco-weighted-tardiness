# ACO for Total Weighted Tardiness

Lab work 1, "Models and Algorithms of Swarm Intelligence", KhNURE.

Ant Colony System (ACS) for the single machine total weighted tardiness problem (1||sum wT):
n jobs with processing time p, weight w and due date d are processed on one machine,
the goal is the order with minimal sum of w * max(0, C - d).

Test data: OR-Library sets wt40, wt50, wt100 (http://people.brunel.ac.uk/~mastjjb/jeb/orlib/wtinfo.html).

## Files

- `aco_twt.py` - problem, heuristics, local search, ACS
- `orlib.py` - reads the OR-Library files from `data/`
- `experiments.py` - experiments: `small`, `params`, `tune`, `bench`, `conv`
- `visualize.py` - visualization of a run (window or GIF)
- `results/` - experiment results (CSV, PNG, GIF)

## Run

```
pip install -r requirements.txt
python experiments.py small
python experiments.py params
python visualize.py --set 40 --k 60
python visualize.py --set 40 --k 60 --gif results/aco_wt40-060.gif
```
