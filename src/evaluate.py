"""
evaluate.py -- forecast-evaluation toolkit for the deepened study:

  garch_fit / garch_recursion : GARCH(1,1) with zero mean (MLE via scipy),
                                then a fixed-parameter daily variance recursion
                                for out-of-sample use between refits
  gaussian_logscore / mixture_logscore : predictive density evaluation
  gaussian_var / mixture_var  : 1-day Value-at-Risk quantiles
  kupiec_pof / christoffersen : VaR backtests (coverage and independence)
  dm_test                     : Diebold-Mariano with HAC variance
  sharpe_diff_bootstrap       : stationary-bootstrap test for overlay Sharpe
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

SQRT2PI = np.sqrt(2 * np.pi)


# ---------------------------------------------------------------------------
# GARCH(1,1), zero mean
# ---------------------------------------------------------------------------
def garch_fit(r):
    """MLE of (omega, alpha, beta) for r_t = sigma_t z_t, z~N(0,1).

    Parameterised through logits to keep alpha,beta >=0 and alpha+beta<1.
    Returns dict(omega, alpha, beta)."""
    r = np.asarray(r, float)
    v0 = r.var()

    def unpack(th):
        a = 1 / (1 + np.exp(-th[1]))            # alpha in (0,1)
        b = (1 - a) / (1 + np.exp(-th[2]))      # beta in (0, 1-alpha)
        w = np.exp(th[0])
        return w, a, b

    def nll(th):
        w, a, b = unpack(th)
        v = np.empty(len(r))
        v[0] = v0
        for t in range(1, len(r)):
            v[t] = w + a * r[t - 1] ** 2 + b * v[t - 1]
        v = np.maximum(v, 1e-12)
        return 0.5 * np.sum(np.log(v) + r ** 2 / v)

    th0 = np.array([np.log(0.05 * v0), np.log(0.08 / 0.92), np.log(0.90 / 0.02)])
    res = optimize.minimize(nll, th0, method="Nelder-Mead",
                            options=dict(maxiter=2000, xatol=1e-6, fatol=1e-8))
    w, a, b = unpack(res.x)
    return dict(omega=w, alpha=a, beta=b, v0=v0)


def garch_next_var(params, r_prev, v_prev):
    """One-step variance recursion with fixed parameters."""
    return params["omega"] + params["alpha"] * r_prev ** 2 + params["beta"] * v_prev


# ---------------------------------------------------------------------------
# density scores and VaR
# ---------------------------------------------------------------------------
def gaussian_logscore(r, var):
    var = np.maximum(var, 1e-12)
    return -0.5 * np.log(2 * np.pi * var) - r ** 2 / (2 * var)


def mixture_logscore(r, q, mu, sigma):
    """log of sum_k q_k N(r; mu_k, sigma_k). r scalar or array with q (.., K)."""
    r = np.atleast_1d(r)[:, None]
    pdf = np.exp(-0.5 * ((r - mu) / sigma) ** 2) / (sigma * SQRT2PI)
    return np.log(np.maximum((q * pdf).sum(axis=1), 1e-300))


def gaussian_var(var, alpha=0.01):
    """1-day VaR (loss threshold, negative return level) for zero-mean normal."""
    return stats.norm.ppf(alpha) * np.sqrt(np.maximum(var, 1e-12))


def mixture_var(q, mu, sigma, alpha=0.01):
    """VaR of a Gaussian mixture by bisection on the CDF."""
    lo = (mu - 10 * sigma).min()
    hi = (mu + 10 * sigma).max()
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = (q * stats.norm.cdf((mid - mu) / sigma)).sum()
        if cdf < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# VaR backtests
# ---------------------------------------------------------------------------
def kupiec_pof(hits, alpha):
    """Kupiec proportion-of-failures LR test. Returns (LR, p-value, hit_rate)."""
    hits = np.asarray(hits, bool)
    n, x = len(hits), int(hits.sum())
    pi_hat = max(x / n, 1e-10)
    ll0 = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
    ll1 = x * np.log(pi_hat) + (n - x) * np.log(max(1 - pi_hat, 1e-10))
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, df=1), x / n


def christoffersen_independence(hits):
    """LR test that VaR breaches do not cluster (first-order independence)."""
    h = np.asarray(hits, int)
    n00 = np.sum((h[:-1] == 0) & (h[1:] == 0))
    n01 = np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10 = np.sum((h[:-1] == 1) & (h[1:] == 0))
    n11 = np.sum((h[:-1] == 1) & (h[1:] == 1))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def safe_log(x):
        return np.log(np.maximum(x, 1e-300))

    ll0 = (n01 + n11) * safe_log(p) + (n00 + n10) * safe_log(1 - p)
    ll1 = (n01 * safe_log(p01) + n00 * safe_log(1 - p01)
           + n11 * safe_log(p11) + n10 * safe_log(1 - p11))
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, df=1)


# ---------------------------------------------------------------------------
# forecast and Sharpe comparison tests
# ---------------------------------------------------------------------------
def dm_test(loss_a, loss_b, lag=5):
    """Diebold-Mariano; positive t => A has HIGHER loss (B better)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    dbar = d.mean()
    g = ((d - dbar) ** 2).mean()
    for l in range(1, lag + 1):
        cov = ((d[l:] - dbar) * (d[:-l] - dbar)).mean()
        g += 2 * (1 - l / (lag + 1)) * cov
    return dbar / np.sqrt(g / n)


def sharpe_diff_bootstrap(ret_a, ret_b, n_boot=2000, mean_block=21, seed=0):
    """Stationary bootstrap (Politis-Romano) p-value for
    H0: Sharpe(A) <= Sharpe(B), using paired resampling of both series."""
    ra, rb = np.asarray(ret_a), np.asarray(ret_b)
    n = len(ra)
    rng = np.random.default_rng(seed)

    def sharpe(x):
        s = x.std(ddof=1)
        return x.mean() / s if s > 0 else 0.0

    obs = sharpe(ra) - sharpe(rb)
    p_geom = 1.0 / mean_block
    count = 0
    for _ in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(0, n)
        for k in range(n):
            idx[k] = i
            i = rng.integers(0, n) if rng.random() < p_geom else (i + 1) % n
        d = sharpe(ra[idx]) - sharpe(rb[idx])
        if d - obs <= -obs:      # centred null: P(diff <= 0)
            count += 1
    return obs, count / n_boot
