#  Replication of Calvano et al. (2020), "Artificial Intelligence, Algorithmic Pricing and Collusion"

Two Q-learning agents set prices in a repeated logit Bertrand duopoly. Without communicating,
they learn to charge prices well above the competitive level and to punish deviations.

## Results

Representative experiment (α = 0.15, β = 4×10⁻⁶, baseline economy), reproducing Table 1:

| Cycle type | Share | Profit gain Δ | Nash on path | Q-loss (path) | Q-loss (all) |
| --- | --- | --- | --- | --- | --- |
| 1, symmetric | 0.269 (0.277) | 0.873 (0.866) | 0.725 (0.686) | 0.001 (0.001) | 0.017 (0.018) |
| 1, asymmetric | 0.346 (0.366) | 0.865 (0.855) | 0.610 (0.661) | 0.001 (0.001) | 0.018 (0.018) |
| 2 | 0.261 (0.238) | 0.834 (0.846) | 0.264 (0.294) | 0.002 (0.002) | 0.018 (0.018) |
| ≥ 3 | 0.124 (0.119) | 0.792 (0.793) | 0.040 (0.025) | 0.005 (0.004) | 0.019 (0.018) |
| **All** | 1.000 | **0.850 (0.849)** | **0.480 (0.505)** | 0.002 (0.002) | 0.018 (0.018) |

1,000 sessions; paper values in brackets. All sessions converged (mean 1.88 million periods).

A 100,000-session run gives Δ = 0.849 and Nash = 0.493. Every gap from the paper is within
2 standard errors of the paper's own 1,000-session sampling noise.

## Scope

I replicated the paper's representative experiment, which is the whole of Table 1, in the results table above.
It demonstrates the profit gain, cycle types, equilibrium play and Q-loss.

My replication confirms that collusion emerges at the paper's chosen parameters,
but not the paper's further claim that it holds across a wide range of learning and exploration rates.
This was done in the paper across 10 million sessions, which I worked out to take roughly 40 days on
my laptop.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/representative.py          # Table 1, 1,000 sessions (~40 s on a MacBook Air)
python scripts/representative.py 20       # Quick check
```

Results are written to `results/representative.csv` (one row per session) and
`results/representative_strategies.npz` (learned strategies, for further analysis).
Sessions are seeded `0 … N−1`, so every run is exactly reproducible.

## Code

| File | Contents |
|---|---|
| `src/calvano/econ.py` | Logit demand (Eq. 5), Nash and monopoly prices, price grid, state encoding, profit table, `Params` / `build_env` |
| `src/calvano/agents.py` | Initial Q-matrix (Eq. 8), numba learning loop (Eqs. 4, 7), limit cycles and profit gain (Eq. 9), equilibrium check (Section III.D) |
| `scripts/representative.py` | Runs the sessions in parallel and prints Table 1 alongside the paper's values |
| `tests/` | Benchmarks checked against the paper (p^N, p^M, grid, Table 4 shares, fn. 21), plus learning and analysis tests |
