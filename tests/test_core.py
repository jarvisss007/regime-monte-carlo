"""Smoke tests: mathematical invariants + the demo's key qualitative claims."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import arch.data.sp500

from regimes import (classify_regimes, regime_return_pools,
                     stationary_distribution, transition_matrix)
from risk import facts_from_paths, stylized_facts, var_cvar
from simulate import gbm_paths, horizon_returns, regime_mc_paths


def _setup():
    px = arch.data.sp500.load()["Adj Close"].dropna()
    rets = px.pct_change().dropna().values
    states, edges = classify_regimes(rets)
    return rets, states


def test_transition_matrix_is_stochastic():
    rets, states = _setup()
    P = transition_matrix(states)
    assert P.shape == (3, 3)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P >= 0).all()
    # regimes must be persistent (diagonal dominant) on real data
    assert (np.diag(P) > 0.85).all()


def test_stationary_distribution_sums_to_one():
    rets, states = _setup()
    pi = stationary_distribution(transition_matrix(states))
    assert np.isclose(pi.sum(), 1.0)
    assert (pi > 0).all()


def test_regime_vols_are_ordered():
    """Turbulent pool must be more volatile than calm pool -- by construction
    terciles guarantee it, and the test guards the labelling convention."""
    rets, states = _setup()
    pools = regime_return_pools(rets, states)
    assert pools[0].std() < pools[1].std() < pools[2].std()


def test_paths_shapes_and_positivity():
    rets, states = _setup()
    P = transition_matrix(states)
    pools = regime_return_pools(rets, states)
    rmc = regime_mc_paths(P, pools, 1, horizon=60, n_paths=500, seed=3)
    gbm = gbm_paths(rets.mean(), rets.std(), 60, 500, seed=3)
    for paths in (rmc, gbm):
        assert paths.shape == (500, 61)
        assert (paths > 0).all()
        assert np.allclose(paths[:, 0], 1.0)


def test_regime_mc_reproduces_fat_tails_gbm_does_not():
    """The project's central claim, as a regression test."""
    rets, states = _setup()
    P = transition_matrix(states)
    pools = regime_return_pools(rets, states)
    rmc = regime_mc_paths(P, pools, 1, horizon=252, n_paths=600, seed=5)
    gbm = gbm_paths(rets.mean(), rets.std(), 252, 600, seed=5)
    real_k = stylized_facts(rets)["excess_kurtosis"]
    k_rmc = facts_from_paths(rmc, n_sample=100)["excess_kurtosis"]
    k_gbm = facts_from_paths(gbm, n_sample=100)["excess_kurtosis"]
    assert real_k > 5
    assert k_gbm < 1.0          # GBM is thin-tailed
    assert k_rmc > 3 * k_gbm + 1  # regime MC is decisively fatter-tailed


def test_var_cvar_ordering():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.2, 20_000)
    var5, cvar5 = var_cvar(x, 0.05)
    var1, cvar1 = var_cvar(x, 0.01)
    assert cvar5 > var5 > 0
    assert var1 > var5           # deeper tail -> bigger loss
    assert cvar1 > var1
