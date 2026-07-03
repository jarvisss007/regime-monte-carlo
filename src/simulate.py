"""
simulate.py -- Monte Carlo engines.

Two simulators over the same horizon so their risk pictures can be compared:

  gbm_paths            -- textbook geometric Brownian motion calibrated to the
                          full-sample mean/vol (the "naive" benchmark)
  regime_mc_paths      -- regime-switching Monte Carlo: each path walks the
                          Markov chain day by day; each day's return is drawn
                          (bootstrap) from the historical pool of ITS regime.

The regime engine therefore reproduces two stylized facts GBM cannot:
volatility clustering (persistence of the chain) and fat tails (empirical
pools, no normality assumption).
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def gbm_paths(mu_daily, sigma_daily, horizon, n_paths, s0=1.0, seed=0):
    """Geometric Brownian motion via exact log-normal discretisation."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, horizon))
    drift = mu_daily - 0.5 * sigma_daily**2
    log_paths = np.cumsum(drift + sigma_daily * z, axis=1)
    return s0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))


def regime_mc_paths(P, pools, start_state, horizon, n_paths, s0=1.0, seed=0):
    """Regime-switching Monte Carlo.

    P           : (k,k) Markov transition matrix
    pools       : list of k arrays of historical daily returns per regime
    start_state : regime the simulation starts in (today's regime)

    Each path: state_{t+1} ~ Categorical(P[state_t]); return_t is an iid draw
    from pools[state_t]. Returns array (n_paths, horizon+1) of price levels.
    """
    rng = np.random.default_rng(seed)
    k = P.shape[0]
    cum = np.cumsum(P, axis=1)

    states = np.full((n_paths,), start_state, dtype=int)
    log_price = np.zeros((n_paths, horizon + 1))
    for t in range(horizon):
        # draw returns for current states, vectorised per regime
        r = np.empty(n_paths)
        for s in range(k):
            m = states == s
            n = int(m.sum())
            if n:
                r[m] = rng.choice(pools[s], size=n, replace=True)
        log_price[:, t + 1] = log_price[:, t] + np.log1p(r)
        # advance the chain
        u = rng.random(n_paths)
        states = (u[:, None] > cum[states]).sum(axis=1)
    return s0 * np.exp(log_price)


def horizon_returns(paths):
    """Total return of each path over the full horizon."""
    return paths[:, -1] / paths[:, 0] - 1.0
