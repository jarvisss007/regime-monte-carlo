"""plots_hmm.py -- figures for the HMM study (run after run_hmm_study.py)."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import arch.data.sp500
from hmm import baum_welch, forward_backward
from regimes import STATE_NAMES

os.makedirs("figures", exist_ok=True)

px = arch.data.sp500.load()["Adj Close"].dropna()
r = px.pct_change().dropna()
fit = baum_welch(r.values, n_states=3)
_, gamma, _, _ = forward_backward(r.values, fit["pi"], fit["A"],
                                  fit["mu"], fit["sigma"])
states = gamma.argmax(axis=1)

# fig 5: hidden states painted on the price history
fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1]})
colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
for k, (name, c) in enumerate(zip(STATE_NAMES, colors)):
    m = states == k
    axes[0].scatter(r.index[m], px.loc[r.index][m], s=1.2, color=c,
                    label=f"{name} (mu {fit['mu'][k]*1e4:+.0f}bp, "
                          f"sig {fit['sigma'][k]*np.sqrt(252):.0%})")
axes[0].set_yscale("log")
axes[0].legend(markerscale=8, frameon=False, fontsize=8, loc="upper left")
axes[0].set_title("Hidden Markov states on the S&P 500 "
                  "(Baum-Welch, smoothed most-likely state)")
axes[1].plot(r.index, gamma[:, 2], lw=0.6, color="#e76f51")
axes[1].set_ylabel("P(turbulent)")
axes[1].set_ylim(0, 1)
fig.tight_layout(); fig.savefig("figures/fig5_hmm_states.png", dpi=150)
plt.close(fig)
print("wrote figures/fig5_hmm_states.png")
