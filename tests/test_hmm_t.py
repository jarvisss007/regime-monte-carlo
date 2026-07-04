"""t-HMM correctness: recover known fat-tailed states from synthetic data,
and quantile/ES internal consistency."""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hmm_t import TFilter, baum_welch_t


def _synthetic_t(T=4000, seed=11):
    rng = np.random.default_rng(seed)
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    mu = np.array([0.0004, -0.001])
    sigma = np.array([0.006, 0.02])
    nu = np.array([30.0, 4.0])          # calm ~ Gaussian, turbulent fat-tailed
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        u = rng.random()
        s[t] = (0 if u < A[s[t - 1], 0] else 1)
    r = mu[s] + sigma[s] * rng.standard_t(df=nu[s])
    return r, s


def test_recovers_fat_tailed_state():
    r, s = _synthetic_t()
    fit = baum_welch_t(r, n_states=2)
    # turbulent state must be identified as clearly fatter-tailed
    assert fit["nu"][1] < 8, f"nu turbulent {fit['nu'][1]}"
    assert fit["nu"][0] > fit["nu"][1]
    assert fit["sigma"][1] > 2 * fit["sigma"][0]
    assert fit["A"][0, 0] > 0.9


def test_filter_quantile_consistency():
    """Empirical coverage of the model's own 5% quantile must be ~5% when
    scoring data simulated FROM the fitted model (internal consistency)."""
    r, _ = _synthetic_t(T=2500)
    fit = baum_welch_t(r, n_states=2, n_iter=40)
    f = TFilter(fit["pi"], fit["A"], fit["mu"], fit["sigma"], fit["nu"])
    hits = 0
    n = 0
    for x in r[500:]:
        q05 = f.var_quantile(0.05)
        hits += x < q05
        n += 1
        f.update(x)
    rate = hits / n
    assert 0.03 < rate < 0.075, f"5% quantile coverage {rate:.3f}"


def test_es_below_var():
    r, _ = _synthetic_t(T=1500)
    fit = baum_welch_t(r, n_states=2, n_iter=30)
    f = TFilter(fit["pi"], fit["A"], fit["mu"], fit["sigma"], fit["nu"])
    for x in r[:300]:
        f.update(x)
    v = f.var_quantile(0.01)
    es = f.expected_shortfall(0.01)
    assert es < v < 0, "ES must be deeper in the tail than VaR"
