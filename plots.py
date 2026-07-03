"""plots.py -- generate the README/report figures. Run after run_simulation."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import arch.data.sp500
from regimes import (STATE_NAMES, classify_regimes, regime_return_pools,
                     transition_matrix)
from risk import max_drawdown_per_path
from simulate import gbm_paths, horizon_returns, regime_mc_paths

os.makedirs("figures", exist_ok=True)

px = arch.data.sp500.load()["Adj Close"].dropna()
rets = px.pct_change().dropna()
states, edges = classify_regimes(rets.values)
P = transition_matrix(states)
pools = regime_return_pools(rets.values, states)
start_state = int(states[states >= 0][-1])

N, H = 10_000, 252
gbm = gbm_paths(rets.mean(), rets.std(), H, N, seed=1)
rmc = regime_mc_paths(P, pools, start_state, H, N, seed=1)

# --- fig 1: regimes painted on the price history -------------------------
fig, ax = plt.subplots(figsize=(9, 3.6))
colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
idx = rets.index
for k, (name, c) in enumerate(zip(STATE_NAMES, colors)):
    m = states == k
    ax.scatter(idx[m], px.loc[idx][m], s=1.2, color=c, label=name)
ax.set_yscale("log")
ax.legend(markerscale=8, loc="upper left", frameon=False)
ax.set_title("S&P 500 with volatility regimes (calm / normal / turbulent)")
fig.tight_layout(); fig.savefig("figures/fig1_regimes.png", dpi=150); plt.close(fig)

# --- fig 2: fan chart of simulated paths ---------------------------------
fig, ax = plt.subplots(figsize=(9, 3.8))
q = np.percentile(rmc, [5, 25, 50, 75, 95], axis=0)
t = np.arange(H + 1)
ax.fill_between(t, q[0], q[4], alpha=0.20, color="#264653", label="5-95%")
ax.fill_between(t, q[1], q[3], alpha=0.35, color="#264653", label="25-75%")
ax.plot(t, q[2], color="#264653", lw=1.5, label="median")
for i in range(12):
    ax.plot(t, rmc[i], lw=0.6, alpha=0.6)
ax.set_xlabel("trading days ahead"); ax.set_ylabel("wealth (start = 1.0)")
ax.set_title(f"Regime-switching Monte Carlo: 1-year fan chart "
             f"({N:,} paths, start regime: {STATE_NAMES[start_state]})")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig2_fanchart.png", dpi=150); plt.close(fig)

# --- fig 3: horizon return distributions, GBM vs regime MC ----------------
fig, ax = plt.subplots(figsize=(9, 3.8))
bins = np.linspace(-0.7, 1.0, 120)
ax.hist(horizon_returns(gbm), bins=bins, density=True, alpha=0.55,
        label="GBM (naive)", color="#8ab17d")
ax.hist(horizon_returns(rmc), bins=bins, density=True, alpha=0.55,
        label="Regime-switching MC", color="#e76f51")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("1-year total return"); ax.set_ylabel("density")
ax.set_title("The left tail is where they disagree")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/fig3_distributions.png", dpi=150); plt.close(fig)

# --- fig 4: max-drawdown distributions ------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.8))
bins = np.linspace(-0.8, 0, 90)
ax.hist(max_drawdown_per_path(gbm), bins=bins, density=True, alpha=0.55,
        label="GBM (naive)", color="#8ab17d")
ax.hist(max_drawdown_per_path(rmc), bins=bins, density=True, alpha=0.55,
        label="Regime-switching MC", color="#e76f51")
ax.set_xlabel("max drawdown within the year"); ax.set_ylabel("density")
ax.set_title("Simulated 1-year maximum drawdowns")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig4_drawdowns.png", dpi=150); plt.close(fig)

print("wrote figures/fig1..fig4")
