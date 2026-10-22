import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from calvano.econ import build_env
from calvano.agents import run_session, session_outcome

# Learning rate and exploration-decay, the papers representative baseline values.
ALPHA, BETA = 0.15, 4e-6
RESULTS = Path(__file__).resolve().parent.parent / "results"

# Table 1 of the paper, for comparison.
PAPER = pd.DataFrame({
    "frequency": [0.277, 0.366, 0.238, 0.119, 1.0],
    "mean gain": [0.866, 0.855, 0.846, 0.793, 0.849],
    "sd gain": [0.115, 0.114, 0.104, 0.097, 0.112]},
    index=["1-sym", "1-asym", "2", ">=3", "all"] 
)


def one_session(env, seed):
    """Train one pair of agents and summarise where they ended up."""

    # Follows the greedy strategies until a state repeats.
    Q, t_end, converged, last_state = run_session(env, ALPHA, BETA, seed=seed)
    out = session_outcome(env, Q, last_state)

    return dict(seed=seed, t_end=t_end, converged=converged, gain=out.gain,
                length=out.length, symmetric = out.symmetric, strategy=out.strategy)


def table1(df):
    """Session shares and profit gain laid out by cycle (like table 1)."""

    df = df.assign(type=np.select(
                   [(df["length"] == 1) & df["symmetric"], df["length"] == 1, df["length"] == 2],
                    ["1-sym", "1-asym", "2"], default=">=3"))
    by_type = df.groupby("type")["gain"].agg(["size", "mean", "std"])
    by_type.loc["all"] = [len(df), df["gain"].mean(), df["gain"].std()]
    by_type = by_type.reindex(PAPER.index)
    return pd.DataFrame({"frequency": by_type["size"] / len(df),
                        "mean gain": by_type["mean"],
                        "sd gain": by_type["std"]})


def main(n_sessions):
    env = build_env()

    start = time.time()
    rows = Parallel(n_jobs=-1)(delayed(one_session)(env, seed) for seed in range(n_sessions))
    print(f"{n_sessions} sessions in {time.time() - start:.0f}s.")

    strategies = np.stack([r.pop("strategy") for r in rows])
    df = pd.DataFrame(rows)

    print("\nThis replication:\n", table1(df).round(3))
    print("\nPaper: (Table 1):\n", PAPER)

    print(f"Converged: {df['converged'].mean():.1%}, "
          f"mean periods to converge: {df.loc[df['converged'], 't_end'].mean():.0f}")

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "representative.csv", index=False)
    np.savez_compressed(RESULTS / "representative_strategies.npz",
                        seeds=df["seed"].to_numpy(), strategies=strategies)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)


