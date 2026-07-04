"""
run_full_study.py -- the deepened, widened study behind the working paper.

Widened : three assets (S&P 500, NASDAQ, WTI crude -- arch bundled data).
Deepened: five forecasters (HMM mixture, rule-based regimes, EWMA(0.94),
          21-day rolling, GARCH(1,1)) evaluated on THREE dimensions:
            1. variance point forecasts   (QLIKE, DM tests vs EWMA)
            2. full predictive density    (log score, DM tests vs EWMA)
            3. 1% Value-at-Risk           (Kupiec coverage + Christoffersen
                                           independence backtests)
          plus HMM state-count sensitivity (K = 2,3,4 on the S&P) and
          stationary-bootstrap significance for vol-target overlay Sharpes.

Protocol: expanding-window walk-forward, initial train 1260 days, refit all
models every 63 days, filters/recursions updated daily, zero look-ahead.

Usage: python run_full_study.py            (~10-15 min)
Writes: results/full_qlike.csv, full_logscore.csv, full_var.csv,
        full_overlay.csv, k_sensitivity.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import arch.data.nasdaq
import arch.data.sp500
import arch.data.wti
from evaluate import (christoffersen_independence, dm_test, garch_fit,
                      garch_next_var, gaussian_logscore, gaussian_var,
                      kupiec_pof, mixture_logscore, mixture_var,
                      sharpe_diff_bootstrap)
from hmm import HMMFilter, baum_welch
from regimes import classify_regimes

TRADING_DAYS = 252
INIT_TRAIN = 1260
REFIT_EVERY = 63
COST = 2e-4
TARGET_ANN = 0.10
VAR_ALPHA = 0.01
MODELS = ["hmm", "rule", "ewma", "roll", "garch"]
LABELS = {"hmm": "HMM mixture", "rule": "Rule-based regimes",
          "ewma": "EWMA(0.94)", "roll": "21-day rolling",
          "garch": "GARCH(1,1)"}


def load_assets():
    return {
        "SP500": arch.data.sp500.load()["Adj Close"].dropna().pct_change().dropna().values,
        "NASDAQ": arch.data.nasdaq.load()["Adj Close"].dropna().pct_change().dropna().values,
        "WTI": arch.data.wti.load()["DCOILWTICO"].dropna().pct_change().dropna().values,
    }


def walk_forward(rv, n_states=3):
    """Returns dict of per-day predvar, logscore, and VaR for each model,
    over the out-of-sample region."""
    T = len(rv)
    var_ = {m: np.full(T, np.nan) for m in MODELS}
    ls_ = {m: np.full(T, np.nan) for m in MODELS}
    var99 = {m: np.full(T, np.nan) for m in MODELS}

    ewma_v = rv[:INIT_TRAIN].var()
    garch_v = None
    for t0 in range(INIT_TRAIN, T, REFIT_EVERY):
        train = rv[:t0]
        hfit = baum_welch(train, n_states=n_states)
        filt = HMMFilter(hfit["pi"], hfit["A"], hfit["mu"], hfit["sigma"])
        for x in train:
            filt.update(x)
        gfit = garch_fit(train)
        if garch_v is None:
            garch_v = train.var()
        _, edges = classify_regimes(train)
        tr_states, _ = classify_regimes(train, edges=edges)
        pool_var = np.array([max(train[tr_states == k].var(), 1e-10)
                             for k in range(3)])

        for t in range(t0, min(t0 + REFIT_EVERY, T)):
            # HMM: full predictive mixture
            q = filt.predict_state_probs()
            var_["hmm"][t] = filt.predict_next_var()
            ls_["hmm"][t] = mixture_logscore(rv[t], q, filt.mu, filt.sigma)[0]
            var99["hmm"][t] = mixture_var(q, filt.mu, filt.sigma, VAR_ALPHA)
            # rule-based
            v21 = np.std(rv[t - 21:t], ddof=1)
            k = 0 if v21 <= edges[0] else (1 if v21 <= edges[1] else 2)
            var_["rule"][t] = pool_var[k]
            # ewma / roll / garch
            var_["ewma"][t] = ewma_v
            var_["roll"][t] = v21 ** 2
            var_["garch"][t] = garch_next_var(gfit, rv[t - 1], garch_v)
            for m in ["rule", "ewma", "roll", "garch"]:
                ls_[m][t] = gaussian_logscore(rv[t], var_[m][t])
                var99[m][t] = gaussian_var(var_[m][t], VAR_ALPHA)
            # post-observation updates
            filt.update(rv[t])
            ewma_v = 0.94 * ewma_v + 0.06 * rv[t] ** 2
            garch_v = var_["garch"][t]
    return var_, ls_, var99


def evaluate_asset(name, rv, results):
    var_, ls_, var99 = walk_forward(rv)
    oos = ~np.isnan(var_["hmm"])
    ro = rv[oos]
    r2 = ro ** 2 + 1e-12
    n = int(oos.sum())
    print(f"\n########## {name}: {n} OOS days ##########")

    # 1) QLIKE
    ql = {m: np.log(np.maximum(var_[m][oos], 1e-12)) + r2 / np.maximum(var_[m][oos], 1e-12)
          for m in MODELS}
    for m in MODELS:
        t = dm_test(ql[m], ql["ewma"]) if m != "ewma" else np.nan
        results["qlike"].append(dict(asset=name, model=LABELS[m],
                                     QLIKE=ql[m].mean(), DM_t_vs_EWMA=t))
    # 2) log score (loss = negative log score)
    for m in MODELS:
        t = dm_test(-ls_[m][oos], -ls_["ewma"][oos]) if m != "ewma" else np.nan
        results["logscore"].append(dict(asset=name, model=LABELS[m],
                                        mean_logscore=ls_[m][oos].mean(),
                                        DM_t_vs_EWMA=t))
    # 3) 1% VaR backtests
    for m in MODELS:
        hits = ro < var99[m][oos]
        lr_k, p_k, rate = kupiec_pof(hits, VAR_ALPHA)
        lr_c, p_c = christoffersen_independence(hits)
        results["var"].append(dict(asset=name, model=LABELS[m],
                                   hit_rate=rate, kupiec_p=p_k,
                                   christoffersen_p=p_c))
    # 4) overlay + bootstrap
    tgt = (TARGET_ANN / np.sqrt(TRADING_DAYS)) ** 2
    for m in MODELS:
        w = np.minimum(1.0, np.sqrt(tgt / np.maximum(var_[m][oos], 1e-10)))
        turn = np.abs(np.diff(np.concatenate([[0.0], w])))
        strat = w * ro - COST * turn
        sh = strat.mean() / strat.std(ddof=1) * np.sqrt(TRADING_DAYS)
        _, p = sharpe_diff_bootstrap(strat, ro, n_boot=1000, seed=1)
        eq = np.cumprod(1 + strat)
        results["overlay"].append(dict(
            asset=name, model=LABELS[m], sharpe=sh,
            ann_ret=eq[-1] ** (TRADING_DAYS / len(strat)) - 1,
            max_dd=(eq / np.maximum.accumulate(eq) - 1).min(),
            p_sharpe_vs_BH=p))
    bh_sh = ro.mean() / ro.std(ddof=1) * np.sqrt(TRADING_DAYS)
    eq = np.cumprod(1 + ro)
    results["overlay"].append(dict(asset=name, model="Buy&Hold", sharpe=bh_sh,
                                   ann_ret=eq[-1] ** (TRADING_DAYS / len(ro)) - 1,
                                   max_dd=(eq / np.maximum.accumulate(eq) - 1).min(),
                                   p_sharpe_vs_BH=np.nan))
    for key in ["qlike", "logscore", "var", "overlay"]:
        sub = pd.DataFrame([x for x in results[key] if x["asset"] == name])
        with pd.option_context("display.float_format", "{:.4f}".format,
                               "display.width", 130):
            print(f"--- {key} ---\n{sub.to_string(index=False)}")


def k_sensitivity(rv):
    rows = []
    for K in (2, 3, 4):
        var_, ls_, _ = walk_forward(rv, n_states=K)
        oos = ~np.isnan(var_["hmm"])
        r2 = rv[oos] ** 2 + 1e-12
        ql = (np.log(np.maximum(var_["hmm"][oos], 1e-12))
              + r2 / np.maximum(var_["hmm"][oos], 1e-12)).mean()
        rows.append(dict(K=K, QLIKE=ql, mean_logscore=ls_["hmm"][oos].mean()))
        print(f"K={K}: QLIKE {ql:.4f}, logscore {rows[-1]['mean_logscore']:.4f}")
    pd.DataFrame(rows).to_csv("results/k_sensitivity.csv", index=False)


def main():
    os.makedirs("results", exist_ok=True)
    assets = load_assets()
    results = {k: [] for k in ["qlike", "logscore", "var", "overlay"]}
    for name, rv in assets.items():
        evaluate_asset(name, rv, results)
    for key, fname in [("qlike", "full_qlike.csv"),
                       ("logscore", "full_logscore.csv"),
                       ("var", "full_var.csv"),
                       ("overlay", "full_overlay.csv")]:
        pd.DataFrame(results[key]).to_csv(f"results/{fname}", index=False)
    print("\n=== HMM state-count sensitivity (S&P 500) ===")
    k_sensitivity(assets["SP500"])
    print("\nwrote results/full_*.csv, k_sensitivity.csv")


if __name__ == "__main__":
    main()
