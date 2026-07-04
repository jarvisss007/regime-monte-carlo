"""
run_ksweep_v2.py -- HMM state-count sweep on fresh S&P 500 data (1999-2026).

Gaussian HMM with K = 2..6 and 8 states, walk-forward as in the main study
but refitting every 126 days (a documented compute compromise -- the sweep
compares K against K under identical conditions, so the coarser refit grid
cancels out). Reports OOS QLIKE and mean log score per K.

Usage: python run_ksweep_v2.py       (~30-45 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hmm import HMMFilter, baum_welch

INIT_TRAIN = 1260
REFIT = 126
KS = [2, 3, 4, 5, 6, 8]


def main():
    eq = pd.read_csv("data/equities_1999_2026.csv", parse_dates=["date"],
                     index_col="date")
    rv = eq["SP500"].dropna().pct_change().dropna().values
    T = len(rv)
    rows = []
    for K in KS:
        t_wall = time.time()
        var_ = np.full(T, np.nan)
        ls_ = np.full(T, np.nan)
        for t0 in range(INIT_TRAIN, T, REFIT):
            fit = baum_welch(rv[:t0], n_states=K, n_iter=80)
            f = HMMFilter(fit["pi"], fit["A"], fit["mu"], fit["sigma"])
            for x in rv[:t0]:
                f.update(x)
            for t in range(t0, min(t0 + REFIT, T)):
                q = f.predict_state_probs()
                var_[t] = f.predict_next_var()
                z = (rv[t] - f.mu) / f.sigma
                pdf = np.exp(-0.5 * z * z) / (f.sigma * np.sqrt(2 * np.pi))
                ls_[t] = np.log(max((q * pdf).sum(), 1e-300))
                f.update(rv[t])
        oos = ~np.isnan(var_)
        r2 = rv[oos] ** 2 + 1e-12
        ql = (np.log(np.maximum(var_[oos], 1e-12))
              + r2 / np.maximum(var_[oos], 1e-12)).mean()
        rows.append(dict(K=K, QLIKE=ql, mean_logscore=ls_[oos].mean(),
                         oos_days=int(oos.sum())))
        print(f"K={K}: QLIKE {ql:.4f}, logscore {rows[-1]['mean_logscore']:.4f}"
              f" ({time.time()-t_wall:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv("results/v2_ksweep.csv", index=False)
    print("wrote results/v2_ksweep.csv")


if __name__ == "__main__":
    main()
