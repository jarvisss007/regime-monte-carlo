"""
run_hybrid_study.py -- Does the hybrid fix both failure modes?

Fields Markov-switching GARCH (Haas-Mittnik-Paolella, K=2, Normal & Student-t)
against the two champions from the main study -- GARCH-t (best point/density
forecaster) and HMM-t (best-calibrated mixture) -- on the identical
out-of-sample days, scored on the identical battery: QLIKE, log score,
VaR at 1/2.5/5% (Kupiec/Christoffersen), expected-shortfall calibration,
Berkowitz density test, crisis-window coverage, and a model confidence set.

Protocol: expanding-window walk-forward, initial train 1260 days, zero
look-ahead. MS-GARCH refits annually (252d, warm-started -- GARCH-type models
adapt daily via their recursion, so frequent refits are unnecessary);
GARCH-t and HMM-t refit every 63d as in the main study. All produce a
prediction every day.

Usage: python run_hybrid_study.py [--asset SP500|NASDAQ|WTI]
Writes results/hybrid_<asset>_*.csv
"""
from __future__ import annotations

import argparse, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from evaluate import (christoffersen_independence, dm_test, kupiec_pof)
from evaluate2 import (berkowitz_lr, es_ratio, fit_garch, garch_recursion_var,
                       mixture_es_generic, mixture_var_generic,
                       model_confidence_set, pit_t, t_es, t_logscore, t_var)
from hmm_t import TFilter, baum_welch_t
from msgarch import MSGarchFilter, fit as msfit

INIT_TRAIN = 1260
REFIT_G = 63
REFIT_MS = 252
VARLV = [0.01, 0.025, 0.05]
ESLV = [0.01, 0.025]
MODELS = ["msg_t", "msg_n", "garch_t", "hmm_t"]
LAB = {"msg_t": "MS-GARCH-t", "msg_n": "MS-GARCH-N",
       "garch_t": "GARCH-t", "hmm_t": "HMM-t"}
CRISES = {"GFC_08_09": ("2008-07-01", "2009-06-30"),
          "COVID_20": ("2020-02-01", "2020-07-31"),
          "Hike_22": ("2022-01-01", "2022-12-31")}


def load(name):
    if name == "WTI":
        s = pd.read_csv("data/wti_1986_2026.csv", parse_dates=["date"],
                        index_col="date")["WTI"].dropna()
    else:
        s = pd.read_csv("data/equities_1999_2026.csv", parse_dates=["date"],
                        index_col="date")[name].dropna()
    r = s.pct_change().dropna()
    return r.values, r.index


def blank(T):
    return {"var": np.full(T, np.nan), "ls": np.full(T, np.nan),
            "pit": np.full(T, np.nan),
            **{f"q{int(a*1000)}": np.full(T, np.nan) for a in VARLV},
            **{f"e{int(a*1000)}": np.full(T, np.nan) for a in ESLV}}


def walk(rv):
    T = len(rv)
    P = {m: blank(T) for m in MODELS}
    v0 = rv[:INIT_TRAIN].var()

    # --- MS-GARCH (annual refit, warm-started) ---
    for dist, key in [("t", "msg_t"), ("normal", "msg_n")]:
        warm = None; filt = None; t0 = time.time()
        for a0 in range(INIT_TRAIN, T, REFIT_MS):
            f = msfit(rv[:a0], student=(dist == "t"), warm=warm,
                      maxiter=500 if warm is not None else 1500)
            warm = f["theta"]
            filt = MSGarchFilter(f, v0=v0)
            for x in rv[:a0]:
                filt.update(x)
            for t in range(a0, min(a0 + REFIT_MS, T)):
                q, sig, nu = filt._mix()
                P[key]["var"][t] = filt.predict_next_var()
                P[key]["ls"][t] = filt.logscore(rv[t])
                P[key]["pit"][t] = filt.pit(rv[t])
                for a in VARLV:
                    P[key][f"q{int(a*1000)}"][t] = mixture_var_generic(q, np.zeros(2), sig, a, nu=nu)
                for a in ESLV:
                    P[key][f"e{int(a*1000)}"][t] = mixture_es_generic(q, np.zeros(2), sig, a, nu=nu)
                filt.update(rv[t])
        print(f"    {key}: {time.time()-t0:.0f}s", flush=True)

    # --- GARCH-t (63d refit) ---
    gv = None; t0 = time.time()
    for a0 in range(INIT_TRAIN, T, REFIT_G):
        g = fit_garch(rv[:a0], "t")
        if gv is None: gv = g["v_last"]
        for t in range(a0, min(a0 + REFIT_G, T)):
            v = garch_recursion_var(g, rv[t-1], gv)
            P["garch_t"]["var"][t] = v
            P["garch_t"]["ls"][t] = t_logscore(rv[t], v, g["nu"])
            P["garch_t"]["pit"][t] = pit_t(rv[t], v, g["nu"])
            for a in VARLV: P["garch_t"][f"q{int(a*1000)}"][t] = t_var(v, g["nu"], a)
            for a in ESLV: P["garch_t"][f"e{int(a*1000)}"][t] = t_es(v, g["nu"], a)
            gv = v
    print(f"    garch_t: {time.time()-t0:.0f}s", flush=True)

    # --- HMM-t (63d refit) ---
    t0 = time.time()
    for a0 in range(INIT_TRAIN, T, REFIT_G):
        ft = baum_welch_t(rv[:a0], n_states=3, n_iter=40)
        filt = TFilter(ft["pi"], ft["A"], ft["mu"], ft["sigma"], ft["nu"])
        for x in rv[:a0]: filt.update(x)
        for t in range(a0, min(a0 + REFIT_G, T)):
            qt = filt.predict_state_probs()
            P["hmm_t"]["var"][t] = filt.predict_next_var()
            P["hmm_t"]["ls"][t] = filt.logscore(rv[t])
            from evaluate2 import pit_mixture_t
            P["hmm_t"]["pit"][t] = pit_mixture_t(rv[t], qt, filt.mu, filt.sigma, filt.nu)
            for a in VARLV:
                P["hmm_t"][f"q{int(a*1000)}"][t] = mixture_var_generic(qt, filt.mu, filt.sigma, a, nu=filt.nu)
            for a in ESLV:
                P["hmm_t"][f"e{int(a*1000)}"][t] = mixture_es_generic(qt, filt.mu, filt.sigma, a, nu=filt.nu)
            filt.update(rv[t])
    print(f"    hmm_t: {time.time()-t0:.0f}s", flush=True)
    return P


def evaluate(name, rv, dates):
    P = walk(rv)
    oos = ~np.isnan(P["msg_t"]["var"]) & ~np.isnan(P["hmm_t"]["var"])
    ro = rv[oos]; d = dates[oos]; r2 = ro**2 + 1e-12; n = int(oos.sum())
    print(f"\n===== {name}: {n} OOS days ({d[0].date()}..{d[-1].date()}) =====", flush=True)
    out = {}

    ql = {m: np.log(np.maximum(P[m]["var"][oos], 1e-12)) + r2/np.maximum(P[m]["var"][oos], 1e-12) for m in MODELS}
    ls = {m: P[m]["ls"][oos] for m in MODELS}
    mcs_q = model_confidence_set({LAB[m]: ql[m] for m in MODELS})
    mcs_l = model_confidence_set({LAB[m]: -ls[m] for m in MODELS})
    out["fc"] = pd.DataFrame([dict(
        asset=name, model=LAB[m], QLIKE=ql[m].mean(), logscore=ls[m].mean(),
        DM_q_vs_GARCHt=(dm_test(ql[m], ql["garch_t"]) if m != "garch_t" else np.nan),
        DM_l_vs_GARCHt=(dm_test(-ls[m], -ls["garch_t"]) if m != "garch_t" else np.nan),
        in_MCS_q=mcs_q[LAB[m]][0], in_MCS_l=mcs_l[LAB[m]][0],
        berkowitz_p=berkowitz_lr(P[m]["pit"][oos])[1]) for m in MODELS])

    vrows = []
    for m in MODELS:
        for a in VARLV:
            hits = ro < P[m][f"q{int(a*1000)}"][oos]
            _, pk, rate = kupiec_pof(hits, a)
            _, pc = christoffersen_independence(hits)
            vrows.append(dict(asset=name, model=LAB[m], level=a, hit_rate=rate,
                              kupiec_p=pk, christo_p=pc))
        for a in ESLV:
            rr, nb = es_ratio(ro, P[m][f"q{int(a*1000)}"][oos], P[m][f"e{int(a*1000)}"][oos])
            vrows.append(dict(asset=name, model=LAB[m], level=a, es_ratio=rr, n_breaches=nb))
    out["risk"] = pd.DataFrame(vrows)

    crows = []
    for cn, (aa, bb) in CRISES.items():
        mask = (d >= aa) & (d <= bb)
        if mask.sum() < 30: continue
        for m in MODELS:
            hits = ro[mask] < P[m]["q10"][oos][mask]
            crows.append(dict(asset=name, window=cn, model=LAB[m],
                              days=int(mask.sum()), breaches=int(hits.sum()),
                              hit_rate=float(hits.mean())))
    out["crisis"] = pd.DataFrame(crows)

    for k, df in out.items():
        df.to_csv(f"results/hybrid_{name}_{k}.csv", index=False)
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
            print(f"--- {k} ---\n{df.to_string(index=False)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--asset", default=None)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    todo = [a.asset] if a.asset else ["SP500", "NASDAQ", "WTI"]
    for name in todo:
        rv, dates = load(name)
        evaluate(name, rv, dates)
    print("\nHYBRID STUDY DONE", flush=True)


if __name__ == "__main__":
    main()
