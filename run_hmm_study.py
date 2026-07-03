"""
run_hmm_study.py -- Do HIDDEN Markov regimes add out-of-sample value?

The Renaissance-lineage question, tested honestly on S&P 500 daily data:

  Part 1  Fit a 3-state Gaussian HMM (Baum-Welch) to the full sample and
          describe what it finds; compare with the simple rule-based regimes.
  Part 2  Walk-forward volatility forecasting shoot-out -- every forecast at
          time t uses only data through t (HMM refit every 63 days on an
          expanding window; filter updated daily):
             HMM predictive mixture   vs   rule-based regime vol
             vs RiskMetrics EWMA(0.94) vs 21-day rolling variance.
          Scored by QLIKE loss; Diebold-Mariano test vs the EWMA benchmark.
  Part 3  Vol-targeting overlay (10% target) driven by each forecaster,
          2 bps/side costs on turnover -- does better forecasting cash out?

Usage: python run_hmm_study.py       (~1-2 min)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import arch.data.sp500
from hmm import HMMFilter, baum_welch, viterbi
from regimes import STATE_NAMES, classify_regimes, realised_vol

TRADING_DAYS = 252
INIT_TRAIN = 1260        # 5 years
REFIT_EVERY = 63         # quarterly
COST_PER_SIDE = 2e-4
TARGET_VOL_ANN = 0.10


def qlike(r2, v):
    return np.log(v) + r2 / v


def diebold_mariano(loss_a, loss_b, lag=5):
    """DM test, HAC (Bartlett) variance. Negative t => A better than B."""
    d = loss_a - loss_b
    n = len(d)
    dbar = d.mean()
    g0 = ((d - dbar) ** 2).mean()
    s = g0
    for l in range(1, lag + 1):
        cov = ((d[l:] - dbar) * (d[:-l] - dbar)).mean()
        s += 2 * (1 - l / (lag + 1)) * cov
    t = dbar / np.sqrt(s / n)
    return dbar, t


def main():
    px = arch.data.sp500.load()["Adj Close"].dropna()
    r = px.pct_change().dropna()
    rv = r.values
    T = len(rv)
    print(f"S&P 500: {T} daily returns "
          f"({r.index[0].date()} .. {r.index[-1].date()})")

    # ---------------- Part 1: what does the HMM find? ---------------------
    fit = baum_welch(rv, n_states=3)
    print(f"\n=== 3-state Gaussian HMM, full sample "
          f"(loglik {fit['loglik']:.0f}, {fit['n_iter_used']} EM iters) ===")
    for k, name in enumerate(STATE_NAMES):
        print(f"  state {k} ({name:9}): mu {fit['mu'][k]*1e4:+.1f} bp/day, "
              f"sigma {fit['sigma'][k]*np.sqrt(252):5.1%} ann, "
              f"self-persistence {fit['A'][k,k]:.3f}, "
              f"expected duration {1/(1-fit['A'][k,k]):5.1f} days")
    print(f"transition matrix:\n{np.round(fit['A'], 3)}")

    path = viterbi(rv, fit["pi"], fit["A"], fit["mu"], fit["sigma"])
    rule_states, _ = classify_regimes(rv)
    ok = rule_states >= 0
    agree = (path[ok] == rule_states[ok]).mean()
    print(f"\nViterbi vs rule-based labels: {agree:.1%} agreement "
          f"(the hidden states largely rediscover the vol regimes, "
          f"without being told what volatility is)")
    pd.DataFrame({"date": r.index, "hmm_state": path}).to_csv(
        "results/hmm_states.csv", index=False)

    # ---------------- Part 2: walk-forward vol forecasting ----------------
    print(f"\nwalk-forward forecasting from day {INIT_TRAIN} "
          f"(refit every {REFIT_EVERY} days) ...")
    pred = {m: np.full(T, np.nan) for m in ["hmm", "rule", "ewma", "roll"]}

    ewma_v = rv[:INIT_TRAIN].var()
    for t0 in range(INIT_TRAIN, T, REFIT_EVERY):
        train = rv[:t0]
        f_ = baum_welch(train, n_states=3)
        filt = HMMFilter(f_["pi"], f_["A"], f_["mu"], f_["sigma"])
        for x in train:
            filt.update(x)
        # rule-based: tercile edges + pool vars from train only
        _, edges = classify_regimes(train)
        tr_states, _ = classify_regimes(train, edges=edges)
        pool_var = np.array([train[tr_states == k].var() for k in range(3)])
        vol_hist = pd.Series(rv[:t0]).rolling(21).std()

        for t in range(t0, min(t0 + REFIT_EVERY, T)):
            pred["hmm"][t] = filt.predict_next_var()
            # today's rule regime from trailing vol (train edges, no lookahead)
            v21 = np.std(rv[t - 21:t], ddof=1)
            k = 0 if v21 <= edges[0] else (1 if v21 <= edges[1] else 2)
            pred["rule"][t] = pool_var[k]
            pred["ewma"][t] = ewma_v
            pred["roll"][t] = v21 ** 2
            # updates AFTER prediction (use r_t only once it is realised)
            filt.update(rv[t])
            ewma_v = 0.94 * ewma_v + 0.06 * rv[t] ** 2

    oos = ~np.isnan(pred["hmm"])
    r2 = rv[oos] ** 2 + 1e-12
    losses = {m: qlike(r2, np.maximum(pred[m][oos], 1e-10)) for m in pred}
    rows = []
    for m, label in [("hmm", "HMM predictive mixture"),
                     ("rule", "Rule-based regime vol"),
                     ("ewma", "EWMA(0.94) RiskMetrics"),
                     ("roll", "21-day rolling")]:
        tstat = (diebold_mariano(losses[m], losses["ewma"])[1]
                 if m != "ewma" else np.nan)
        rows.append({"forecaster": label, "QLIKE": losses[m].mean(),
                     "DM_vs_EWMA_t": tstat})
    tab = pd.DataFrame(rows)
    tab.to_csv("results/vol_forecast_qlike.csv", index=False)
    print(f"\n=== QLIKE (lower better), {oos.sum()} OOS days ===")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(tab.to_string(index=False))

    # ---------------- Part 3: vol-targeting overlay ------------------------
    tgt_v = (TARGET_VOL_ANN / np.sqrt(TRADING_DAYS)) ** 2
    rows = []
    ro = rv[oos]
    for m, label in [("hmm", "HMM"), ("rule", "Rule-based"),
                     ("ewma", "EWMA"), ("roll", "Rolling")]:
        w = np.minimum(1.0, np.sqrt(tgt_v / np.maximum(pred[m][oos], 1e-10)))
        turn = np.abs(np.diff(np.concatenate([[0.0], w])))
        strat = w * ro - COST_PER_SIDE * turn
        sh = strat.mean() / strat.std(ddof=1) * np.sqrt(TRADING_DAYS)
        eq = np.cumprod(1 + strat)
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        rows.append({"overlay": label,
                     "ann_ret": (eq[-1] ** (TRADING_DAYS / len(strat)) - 1),
                     "sharpe": sh, "max_dd": dd,
                     "realised_vol": strat.std(ddof=1) * np.sqrt(TRADING_DAYS)})
    bh = ro
    eq = np.cumprod(1 + bh)
    rows.append({"overlay": "Buy&Hold (no overlay)",
                 "ann_ret": eq[-1] ** (TRADING_DAYS / len(bh)) - 1,
                 "sharpe": bh.mean() / bh.std(ddof=1) * np.sqrt(TRADING_DAYS),
                 "max_dd": (eq / np.maximum.accumulate(eq) - 1).min(),
                 "realised_vol": bh.std(ddof=1) * np.sqrt(TRADING_DAYS)})
    ov = pd.DataFrame(rows)
    ov.to_csv("results/vol_target_overlay.csv", index=False)
    print(f"\n=== 10%-vol-target overlay, net of {COST_PER_SIDE*1e4:.0f} bps/side ===")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(ov.to_string(index=False))

    print("\nwrote results/hmm_states.csv, vol_forecast_qlike.csv, "
          "vol_target_overlay.csv")


if __name__ == "__main__":
    main()
