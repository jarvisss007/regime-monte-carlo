"""
regimes.py -- identify market regimes and estimate the Markov chain that
governs switching between them.

Design choice (deliberately simple and defensible): regimes are defined by
observable, rule-based criteria rather than a latent-state model (HMM), so
every classification can be audited by hand:

    Calm    : trailing 21-day realised volatility in the bottom tercile
    Normal  : middle tercile
    Turbulent: top tercile

Terciles are estimated on the estimation window only. The Markov transition
matrix is estimated by simple counting (maximum likelihood for a first-order
chain):  P[i, j] = (# transitions i -> j) / (# days in state i).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATE_NAMES = ["calm", "normal", "turbulent"]
VOL_WINDOW = 21  # trading days used for realised volatility


def realised_vol(returns, window=VOL_WINDOW):
    """Trailing realised volatility (std of daily returns), aligned so that
    vol[t] uses returns up to and including day t."""
    return pd.Series(returns).rolling(window).std()


def classify_regimes(returns, window=VOL_WINDOW, edges=None):
    """Label each day 0=calm, 1=normal, 2=turbulent by trailing vol terciles.

    Returns (states, edges). Days before the first full window are NaN -> -1.
    Pass precomputed `edges` to classify out-of-sample data with in-sample
    terciles (no look-ahead).
    """
    vol = realised_vol(returns, window)
    if edges is None:
        edges = np.nanquantile(vol, [1 / 3, 2 / 3])
    states = np.full(len(vol), -1, dtype=int)
    ok = vol.notna().values
    v = vol.values
    states[ok & (v <= edges[0])] = 0
    states[ok & (v > edges[0]) & (v <= edges[1])] = 1
    states[ok & (v > edges[1])] = 2
    return states, edges


def transition_matrix(states, n_states=3):
    """Maximum-likelihood first-order Markov transition matrix by counting.

    Ignores days labelled -1 (warm-up). Rows sum to 1; a state never visited
    gets a uniform row (harmless: the chain can never enter it).
    """
    counts = np.zeros((n_states, n_states))
    s = states[states >= 0]
    for a, b in zip(s[:-1], s[1:]):
        counts[a, b] += 1
    P = np.zeros_like(counts)
    for i in range(n_states):
        row = counts[i].sum()
        P[i] = counts[i] / row if row > 0 else 1.0 / n_states
    return P


def stationary_distribution(P):
    """Long-run fraction of time in each state (left eigenvector for eig=1)."""
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    v = np.abs(v)
    return v / v.sum()


def regime_return_pools(returns, states, n_states=3):
    """Historical daily returns bucketed by same-day regime label.

    These pools are what the Monte Carlo engine resamples from, so each
    simulated day draws from the empirical distribution OF ITS REGIME --
    fat tails in the turbulent pool are preserved, not Gaussianised.
    """
    r = np.asarray(returns, dtype=float)
    return [r[states == k] for k in range(n_states)]
