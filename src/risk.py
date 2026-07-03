"""
risk.py -- risk metrics computed on simulated paths, and stylized-facts
validation comparing simulator output against the real series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def var_cvar(total_returns, alpha=0.05):
    """Value-at-Risk and Conditional VaR (expected shortfall) at level alpha.

    Reported as losses (positive numbers): VaR_5% = 0.30 means a 5% chance of
    losing 30% or more over the horizon.
    """
    q = np.quantile(total_returns, alpha)
    cvar = total_returns[total_returns <= q].mean()
    return -q, -cvar


def max_drawdown_per_path(paths):
    """Maximum drawdown of each simulated path (negative numbers)."""
    peaks = np.maximum.accumulate(paths, axis=1)
    dd = paths / peaks - 1.0
    return dd.min(axis=1)


def summary_table(total_returns, drawdowns, label):
    """One-row DataFrame of the headline risk numbers for a simulator."""
    var5, cvar5 = var_cvar(total_returns, 0.05)
    var1, cvar1 = var_cvar(total_returns, 0.01)
    return pd.DataFrame([{
        "simulator": label,
        "mean_return": total_returns.mean(),
        "median_return": np.median(total_returns),
        "VaR_5%": var5, "CVaR_5%": cvar5,
        "VaR_1%": var1, "CVaR_1%": cvar1,
        "P(loss)": (total_returns < 0).mean(),
        "median_maxDD": np.median(drawdowns),
        "worst_maxDD": drawdowns.min(),
    }])


# ---------------------------------------------------------------------------
# stylized-facts validation
# ---------------------------------------------------------------------------
def stylized_facts(daily_returns):
    """Excess kurtosis, skew, and volatility clustering (lag-1 autocorrelation
    of |r|) -- the facts a good simulator should reproduce."""
    r = pd.Series(np.asarray(daily_returns, dtype=float)).dropna()
    abs_r = r.abs()
    return {
        "excess_kurtosis": float(r.kurtosis()),
        "skew": float(r.skew()),
        "volclust_acf1": float(abs_r.autocorr(1)),
    }


def facts_from_paths(paths, n_sample=200, seed=0):
    """Average stylized facts across a sample of simulated paths."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n_sample, len(paths)), replace=False)
    rows = []
    for i in idx:
        r = np.diff(paths[i]) / paths[i][:-1]
        rows.append(stylized_facts(r))
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
