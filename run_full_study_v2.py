"""
run_full_study_v2.py -- the upgraded study: fair tails, fresh data, full
calibration toolkit.

vs run_full_study.py:
  * data through mid-2026 (dot-com, GFC, COVID-2020, 2022 all in sample),
    cached in data/ (yfinance ^GSPC/^IXIC; FRED DCOILWTICO)
  * SEVEN models: Gaussian HMM, Student-t HMM, rule-based regimes,
    EWMA(0.94), 21-day rolling, GARCH(1,1)-Normal, GARCH(1,1)-t
  * VaR at 1%/2.5%/5% (Kupiec + Christoffersen at each level)
  * Expected Shortfall calibration (realised/predicted ES ratio at 1%, 2.5%)
  * PIT + Berkowitz density-calibration test
  * Hansen-Lunde-Nason Model Confidence Sets on QLIKE and log-score losses
  * crisis-window VaR coverage (GFC, COVID-2020, 2022)
  * 10% vol-target overlay + stationary-bootstrap Sharpe tests

Protocol unchanged: expanding-window walk-forward, initial train 1260 days,
refit every 63 days, filters/recursions updated daily, zero look-ahead.

Usage: python run_full_study_v2.py [--asset SP500|NASDAQ|WTI] [--smoke]
Writes per-asset checkpoints results/v2_<asset>_*.csv then aggregates.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from evaluate import (christoffersen_independence, dm_test,
                      gaussian_logscore, kupiec_pof, sharpe_diff_bootstrap)
from evaluate2 import (berkowitz_lr, es_ratio, fit_garch, garch_recursion_var,
                       gaussian_es, mixture_es_generic, mixture_var_generic,
                       model_confidence_set, pit_gaussian, pit_mixture_gauss,
                       pit_mixture_t, pit_t, t_es, t_logscore, t_var)
from hmm import HMMFilter, baum_welch
from hmm_t import TFilter, baum_welch_t
from regimes import classify_regimes

TRADING_DAYS = 252
INIT_TRAIN = 1260
REFIT_EVERY = 63
COST = 2e-4
TARGET_ANN = 0.10
VAR_LEVELS = [0.01, 0.025, 0.05]
ES_LEVELS = [0.01, 0.025]
MODELS = ["hmm_n", "hmm_t", "rule", "ewma", "roll", "garch_n", "garch_t"]
LABELS = {"hmm_n": "HMM-Normal", "hmm_t": "HMM-t",
          "rule": "Rule-based", "ewma": "EWMA(0.94)",
          "roll": "21d rolling", "garch_n": "GARCH-N", "garch_t": "GARCH-t"}
CRISES = {"GFC_2008H2_09H1": ("2008-07-01", "2009-06-30"),
          "COVID_2020H1": ("2020-02-01", "2020-07-31"),
          "Hike_2022": ("2022-01-01", "2022-12-31")}


def load_assets():
    eq = pd.read_csv("data/equities_1999_2026.csv", parse_dates=["date"],
                     index_col="date")
    wti = pd.read_csv("data/wti_1986_2026.csv", parse_dates=["date"],
                      index_col="date")["WTI"].dropna()
    out = {}
    for name in ["SP500", "NASDAQ"]:
        s = eq[name].dropna()
        out[name] = (s.pct_change().dropna().values,
                     s.pct_change().dropna().index)
    r = wti.pct_change().dropna()
    out["WTI"] = (r.values, r.index)
    return out


def walk_forward(rv, smoke=False):
    T = len(rv)
    refit = REFIT_EVERY if not smoke else 21
    start = INIT_TRAIN if not smoke else T - 200
    P = {m: {"var": np.full(T, np.nan), "ls": np.full(T, np.nan),
             "pit": np.full(T, np.nan),
             **{f"q{int(a*1000)}": np.full(T, np.nan) for a in VAR_LEVELS},
             **{f"es{int(a*1000)}": np.full(T, np.nan) for a in ES_LEVELS}}
         for m in MODELS}

    ewma_v = rv[:start].var()
    garch_vn = garch_vt = None
    t_wall = time.time()
    for t0 in range(start, T, refit):
        train = rv[:t0]
        fn = baum_welch(train, n_states=3, n_iter=80)
        filt_n = HMMFilter(fn["pi"], fn["A"], fn["mu"], fn["sigma"])
        ft = baum_welch_t(train, n_states=3, n_iter=40)
        filt_t = TFilter(ft["pi"], ft["A"], ft["mu"], ft["sigma"], ft["nu"])
        for x in train:
            filt_n.update(x); filt_t.update(x)
        gn = fit_garch(train, "normal"); gt = fit_garch(train, "t")
        if garch_vn is None:
            garch_vn, garch_vt = gn["v_last"], gt["v_last"]
        _, edges = classify_regimes(train)
        trs, _ = classify_regimes(train, edges=edges)
        pool_var = np.array([max(train[trs == k].var(), 1e-10)
                             for k in range(3)])

        for t in range(t0, min(t0 + refit, T)):
            r_t = rv[t]
            # ---- HMM Normal
            q = filt_n.predict_state_probs()
            P["hmm_n"]["var"][t] = filt_n.predict_next_var()
            z = (r_t - filt_n.mu) / filt_n.sigma
            pdf = np.exp(-0.5 * z * z) / (filt_n.sigma * np.sqrt(2 * np.pi))
            P["hmm_n"]["ls"][t] = np.log(max((q * pdf).sum(), 1e-300))
            P["hmm_n"]["pit"][t] = pit_mixture_gauss(r_t, q, filt_n.mu,
                                                     filt_n.sigma)
            for a in VAR_LEVELS:
                P["hmm_n"][f"q{int(a*1000)}"][t] = mixture_var_generic(
                    q, filt_n.mu, filt_n.sigma, a)
            for a in ES_LEVELS:
                P["hmm_n"][f"es{int(a*1000)}"][t] = mixture_es_generic(
                    q, filt_n.mu, filt_n.sigma, a,
                    var_q=P["hmm_n"][f"q{int(a*1000)}"][t])
            # ---- HMM t
            qt = filt_t.predict_state_probs()
            P["hmm_t"]["var"][t] = filt_t.predict_next_var()
            P["hmm_t"]["ls"][t] = filt_t.logscore(r_t)
            P["hmm_t"]["pit"][t] = pit_mixture_t(r_t, qt, filt_t.mu,
                                                 filt_t.sigma, filt_t.nu)
            for a in VAR_LEVELS:
                P["hmm_t"][f"q{int(a*1000)}"][t] = mixture_var_generic(
                    qt, filt_t.mu, filt_t.sigma, a, nu=filt_t.nu)
            for a in ES_LEVELS:
                P["hmm_t"][f"es{int(a*1000)}"][t] = mixture_es_generic(
                    qt, filt_t.mu, filt_t.sigma, a, nu=filt_t.nu,
                    var_q=P["hmm_t"][f"q{int(a*1000)}"][t])
            # ---- Gaussian-density models
            v21 = np.std(rv[t - 21:t], ddof=1)
            k = 0 if v21 <= edges[0] else (1 if v21 <= edges[1] else 2)
            gvals = {"rule": pool_var[k], "ewma": ewma_v, "roll": v21 ** 2,
                     "garch_n": garch_recursion_var(gn, rv[t - 1], garch_vn)}
            for m, v in gvals.items():
                P[m]["var"][t] = v
                P[m]["ls"][t] = gaussian_logscore(r_t, v)
                P[m]["pit"][t] = pit_gaussian(r_t, v)
                for a in VAR_LEVELS:
                    P[m][f"q{int(a*1000)}"][t] = (
                        np.sqrt(v) * NORM_PPF[a])
                for a in ES_LEVELS:
                    P[m][f"es{int(a*1000)}"][t] = gaussian_es(v, a)
            # ---- GARCH-t
            vt = garch_recursion_var(gt, rv[t - 1], garch_vt)
            P["garch_t"]["var"][t] = vt
            P["garch_t"]["ls"][t] = t_logscore(r_t, vt, gt["nu"])
            P["garch_t"]["pit"][t] = pit_t(r_t, vt, gt["nu"])
            for a in VAR_LEVELS:
                P["garch_t"][f"q{int(a*1000)}"][t] = t_var(vt, gt["nu"], a)
            for a in ES_LEVELS:
                P["garch_t"][f"es{int(a*1000)}"][t] = t_es(vt, gt["nu"], a)
            # ---- post-observation updates
            filt_n.update(r_t); filt_t.update(r_t)
            ewma_v = 0.94 * ewma_v + 0.06 * r_t ** 2
            garch_vn = gvals["garch_n"]; garch_vt = vt
        print(f"    refit@{t0}: {time.time()-t_wall:.0f}s elapsed", flush=True)
    return P


NORM_PPF = {}


def _init_norm_ppf():
    from scipy import stats
    for a in VAR_LEVELS:
        NORM_PPF[a] = stats.norm.ppf(a)


def evaluate_asset(name, rv, dates, smoke=False):
    P = walk_forward(rv, smoke=smoke)
    oos = ~np.isnan(P["hmm_n"]["var"])
    ro = rv[oos]; d_oos = dates[oos]
    r2 = ro ** 2 + 1e-12
    print(f"\n########## {name}: {int(oos.sum())} OOS days "
          f"({d_oos[0].date()} .. {d_oos[-1].date()}) ##########", flush=True)
    out = {}

    # QLIKE + DM + MCS
    ql = {m: np.log(np.maximum(P[m]["var"][oos], 1e-12))
          + r2 / np.maximum(P[m]["var"][oos], 1e-12) for m in MODELS}
    mcs_q = model_confidence_set({LABELS[m]: ql[m] for m in MODELS})
    out["qlike"] = pd.DataFrame([
        dict(asset=name, model=LABELS[m], QLIKE=ql[m].mean(),
             DM_t_vs_GARCHt=dm_test(ql[m], ql["garch_t"]) if m != "garch_t" else np.nan,
             in_MCS=mcs_q[LABELS[m]][0], MCS_p=mcs_q[LABELS[m]][1])
        for m in MODELS])
    # log score + DM + MCS
    ls = {m: P[m]["ls"][oos] for m in MODELS}
    mcs_l = model_confidence_set({LABELS[m]: -ls[m] for m in MODELS})
    out["logscore"] = pd.DataFrame([
        dict(asset=name, model=LABELS[m], mean_logscore=ls[m].mean(),
             DM_t_vs_GARCHt=dm_test(-ls[m], -ls["garch_t"]) if m != "garch_t" else np.nan,
             in_MCS=mcs_l[LABELS[m]][0], MCS_p=mcs_l[LABELS[m]][1])
        for m in MODELS])
    # VaR at three levels
    rows = []
    for m in MODELS:
        for a in VAR_LEVELS:
            hits = ro < P[m][f"q{int(a*1000)}"][oos]
            _, pk, rate = kupiec_pof(hits, a)
            _, pc = christoffersen_independence(hits)
            rows.append(dict(asset=name, model=LABELS[m], level=a,
                             hit_rate=rate, kupiec_p=pk, christo_p=pc))
    out["var"] = pd.DataFrame(rows)
    # ES ratios
    rows = []
    for m in MODELS:
        for a in ES_LEVELS:
            ratio, nb = es_ratio(ro, P[m][f"q{int(a*1000)}"][oos],
                                 P[m][f"es{int(a*1000)}"][oos])
            rows.append(dict(asset=name, model=LABELS[m], level=a,
                             es_ratio=ratio, n_breaches=nb))
    out["es"] = pd.DataFrame(rows)
    # PIT / Berkowitz
    out["pit"] = pd.DataFrame([
        dict(asset=name, model=LABELS[m],
             berkowitz_p=berkowitz_lr(P[m]["pit"][oos])[1]) for m in MODELS])
    # crisis windows (1% VaR)
    rows = []
    for cname, (a_, b_) in CRISES.items():
        mask = (d_oos >= a_) & (d_oos <= b_)
        if mask.sum() < 30:
            continue
        for m in MODELS:
            hits = ro[mask] < P[m]["q10"][oos][mask]
            rows.append(dict(asset=name, window=cname, model=LABELS[m],
                             days=int(mask.sum()), breaches=int(hits.sum()),
                             hit_rate=float(hits.mean())))
    out["crisis"] = pd.DataFrame(rows)
    # overlay
    tgt = (TARGET_ANN / np.sqrt(TRADING_DAYS)) ** 2
    rows = []
    for m in MODELS:
        w = np.minimum(1.0, np.sqrt(tgt / np.maximum(P[m]["var"][oos], 1e-10)))
        turn = np.abs(np.diff(np.concatenate([[0.0], w])))
        strat = w * ro - COST * turn
        sh = strat.mean() / strat.std(ddof=1) * np.sqrt(TRADING_DAYS)
        _, pb = sharpe_diff_bootstrap(strat, ro, n_boot=500, seed=1)
        eq = np.cumprod(1 + strat)
        rows.append(dict(asset=name, model=LABELS[m], sharpe=sh,
                         max_dd=(eq / np.maximum.accumulate(eq) - 1).min(),
                         p_vs_BH=pb))
    bh_sh = ro.mean() / ro.std(ddof=1) * np.sqrt(TRADING_DAYS)
    eq = np.cumprod(1 + ro)
    rows.append(dict(asset=name, model="Buy&Hold", sharpe=bh_sh,
                     max_dd=(eq / np.maximum.accumulate(eq) - 1).min(),
                     p_vs_BH=np.nan))
    out["overlay"] = pd.DataFrame(rows)

    for key, df in out.items():
        df.to_csv(f"results/v2_{name}_{key}.csv", index=False)
        with pd.option_context("display.float_format", "{:.4f}".format,
                               "display.width", 150):
            print(f"--- {key} ---\n{df.to_string(index=False)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    _init_norm_ppf()
    os.makedirs("results", exist_ok=True)
    assets = load_assets()
    todo = [args.asset] if args.asset else list(assets)
    for name in todo:
        rv, dates = assets[name]
        evaluate_asset(name, rv, dates, smoke=args.smoke)
    print("\nALL DONE")


if __name__ == "__main__":
    main()
