"""MS-GARCH correctness: recover a KNOWN 2-regime MS-GARCH from simulated
data, and check the online filter's internal consistency."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from msgarch import MSGarchFilter, fit


def _simulate(T=6000, seed=17):
    """Haas-MP MS-GARCH: two regimes, each its own GARCH, calm persistent."""
    rng = np.random.default_rng(seed)
    P = np.array([[0.985, 0.015], [0.05, 0.95]])
    om = np.array([2e-6, 2e-5]); al = np.array([0.04, 0.10]); be = np.array([0.94, 0.86])
    h = np.array([om[0] / (1 - al[0] - be[0]), om[1] / (1 - al[1] - be[1])])
    s = 0
    r = np.zeros(T)
    for t in range(T):
        s = 0 if rng.random() < P[s, 0] else 1
        r[t] = np.sqrt(h[s]) * rng.standard_normal()
        r2 = r[t] ** 2
        h = om + al * r2 + be * h
    return r, P, om, al, be


def test_recovers_known_msgarch():
    r, P, om, al, be = _simulate()
    f = fit(r, student=False, maxiter=2500)
    # calm regime (0) must be less volatile & more persistent than regime 1
    unc = f["omega"] / (1 - f["alpha"] - f["beta"])
    assert unc[0] < unc[1], "regime ordering (calm first) failed"
    # persistence recovered in the right ballpark for both regimes
    assert 0.90 < f["alpha"][0] + f["beta"][0] < 0.999
    assert 0.85 < f["alpha"][1] + f["beta"][1] < 0.999
    # the two regimes are genuinely distinct (>2x unconditional vol ratio)
    assert unc[1] / unc[0] > 1.8
    # transition matrix: both states persistent
    assert f["P"][0, 0] > 0.9 and f["P"][1, 1] > 0.85


def test_filter_positive_and_variance_between_regimes():
    r, *_ = _simulate(T=3000)
    f = fit(r, student=True, maxiter=1500)
    filt = MSGarchFilter(f, v0=r.var())
    vs = []
    for x in r[:800]:
        v = filt.predict_next_var()
        assert v > 0
        vs.append(v)
        filt.update(x)
    unc = f["omega"] / (1 - f["alpha"] - f["beta"])
    # predictive variance stays within a sane band around the regime uncond vars
    assert unc.min() * 0.1 < np.median(vs) < unc.max() * 5


def test_filter_pit_in_unit_interval():
    r, *_ = _simulate(T=1500)
    f = fit(r, student=True, maxiter=1200)
    filt = MSGarchFilter(f, v0=r.var())
    u = []
    for x in r:
        u.append(filt.pit(x)); filt.update(x)
    u = np.array(u)
    assert (u > 0).all() and (u < 1).all()
    assert 0.3 < u.mean() < 0.7   # roughly uniform-ish under a decent fit
