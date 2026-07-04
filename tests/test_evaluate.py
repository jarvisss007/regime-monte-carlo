"""Tests for the evaluation toolkit: GARCH recovery, VaR backtests on data
with KNOWN properties, and score/quantile invariants."""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import (christoffersen_independence, dm_test, garch_fit,
                      garch_next_var, gaussian_logscore, gaussian_var,
                      kupiec_pof, mixture_logscore, mixture_var)


def _garch_sim(T=4000, omega=2e-6, alpha=0.08, beta=0.90, seed=3):
    rng = np.random.default_rng(seed)
    r = np.zeros(T)
    v = omega / (1 - alpha - beta)
    for t in range(1, T):
        v = omega + alpha * r[t - 1] ** 2 + beta * v
        r[t] = np.sqrt(v) * rng.standard_normal()
    return r


def test_garch_recovers_persistence():
    r = _garch_sim()
    fit = garch_fit(r)
    assert 0 < fit["alpha"] < 0.2
    assert 0.75 < fit["beta"] < 0.99
    persistence = fit["alpha"] + fit["beta"]
    assert abs(persistence - 0.98) < 0.03
    # recursion produces positive variance
    assert garch_next_var(fit, 0.01, r.var()) > 0


def test_kupiec_correct_and_incorrect_coverage():
    rng = np.random.default_rng(0)
    n = 4000
    # correct 1% coverage -> should NOT reject
    hits_ok = rng.random(n) < 0.01
    _, p_ok, _ = kupiec_pof(hits_ok, 0.01)
    # badly miscalibrated (3x too many hits) -> should reject
    hits_bad = rng.random(n) < 0.03
    _, p_bad, _ = kupiec_pof(hits_bad, 0.01)
    assert p_ok > 0.05
    assert p_bad < 0.01


def test_christoffersen_detects_clustering():
    rng = np.random.default_rng(1)
    n = 4000
    iid = rng.random(n) < 0.02
    _, p_iid = christoffersen_independence(iid)
    clustered = np.zeros(n, bool)          # breaches in runs of 5
    for start in rng.integers(0, n - 5, 16):
        clustered[start:start + 5] = True
    _, p_cl = christoffersen_independence(clustered)
    assert p_iid > 0.05
    assert p_cl < 0.01


def test_mixture_var_matches_gaussian_when_degenerate():
    """A 1-component 'mixture' must reproduce the closed-form Gaussian VaR."""
    q = np.array([1.0])
    mu = np.array([0.0])
    sigma = np.array([0.02])
    v_mix = mixture_var(q, mu, sigma, 0.01)
    v_gau = gaussian_var(np.array(0.02 ** 2), 0.01)
    assert abs(v_mix - v_gau) < 1e-6


def test_logscore_prefers_true_variance():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.02, 5000)
    true = gaussian_logscore(r, np.full_like(r, 0.02 ** 2)).mean()
    toobig = gaussian_logscore(r, np.full_like(r, 0.05 ** 2)).mean()
    toosmall = gaussian_logscore(r, np.full_like(r, 0.008 ** 2)).mean()
    assert true > toobig and true > toosmall


def test_dm_sign_convention():
    rng = np.random.default_rng(4)
    base = rng.random(1000)
    worse = base + 0.3 + 0.01 * rng.standard_normal(1000)
    t = dm_test(worse, base)
    assert t > 3  # positive => first argument has higher loss
