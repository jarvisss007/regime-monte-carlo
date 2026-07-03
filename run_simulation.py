"""
run_simulation.py -- end-to-end demo on real S&P 500 data (bundled in the
`arch` package, 1999-2018; no vendor access needed).

Pipeline:
  1. classify each day into calm/normal/turbulent regimes (vol terciles)
  2. estimate the Markov transition matrix by counting
  3. simulate 1-year-ahead price paths two ways:
       (a) textbook GBM calibrated to full-sample mean/vol
       (b) regime-switching Monte Carlo (this project)
  4. compare risk metrics (VaR/CVaR/drawdowns) and stylized facts
     (fat tails, volatility clustering) against the real series

Usage: python run_simulation.py [--paths 10000] [--horizon 252]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import arch.data.sp500
from regimes import (STATE_NAMES, classify_regimes, regime_return_pools,
                     stationary_distribution, transition_matrix)
from risk import (facts_from_paths, max_drawdown_per_path, stylized_facts,
                  summary_table)
from simulate import gbm_paths, horizon_returns, regime_mc_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=10_000)
    ap.add_argument("--horizon", type=int, default=252)
    args = ap.parse_args()

    px = arch.data.sp500.load()["Adj Close"].dropna()
    rets = px.pct_change().dropna().values
    print(f"S&P 500 daily returns: {len(rets)} obs "
          f"({px.index[0].date()} .. {px.index[-1].date()})")

    # 1-2. regimes + Markov chain
    states, edges = classify_regimes(rets)
    P = transition_matrix(states)
    pi = stationary_distribution(P)
    pools = regime_return_pools(rets, states)

    print("\nMarkov transition matrix (rows: from, cols: to "
          f"{STATE_NAMES}):\n{np.round(P, 3)}")
    print(f"stationary distribution: "
          f"{dict(zip(STATE_NAMES, np.round(pi, 3)))}")
    for k, name in enumerate(STATE_NAMES):
        print(f"  {name:9}: {len(pools[k]):5d} days | "
              f"ann.vol {pools[k].std()*np.sqrt(252):5.1%} | "
              f"mean daily {pools[k].mean()*1e4:+.1f} bp")

    # 3. simulate
    mu, sigma = rets.mean(), rets.std()
    start_state = int(states[states >= 0][-1])
    print(f"\nsimulating {args.paths} paths x {args.horizon} days "
          f"(starting regime: {STATE_NAMES[start_state]}) ...")
    gbm = gbm_paths(mu, sigma, args.horizon, args.paths, seed=1)
    rmc = regime_mc_paths(P, pools, start_state, args.horizon, args.paths,
                          seed=1)

    # 4. compare
    tables = []
    for label, paths in [("GBM (naive)", gbm), ("Regime-switching MC", rmc)]:
        tr = horizon_returns(paths)
        dd = max_drawdown_per_path(paths)
        tables.append(summary_table(tr, dd, label))
    table = pd.concat(tables, ignore_index=True)
    os.makedirs("results", exist_ok=True)
    table.to_csv("results/risk_comparison.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 140):
        print("\n=== 1-year-ahead risk comparison ===")
        print(table.to_string(index=False))

    real = stylized_facts(rets)
    facts = pd.DataFrame([
        {"series": "real S&P 500", **real},
        {"series": "GBM (naive)", **facts_from_paths(gbm)},
        {"series": "Regime-switching MC", **facts_from_paths(rmc)},
    ])
    facts.to_csv("results/stylized_facts.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print("\n=== stylized facts (simulators vs reality) ===")
        print(facts.to_string(index=False))

    print("\nwrote results/risk_comparison.csv, results/stylized_facts.csv")


if __name__ == "__main__":
    main()
