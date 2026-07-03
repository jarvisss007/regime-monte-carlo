"""HMM correctness tests: EM must improve likelihood, and the model must
recover KNOWN hidden states from synthetic data before we trust it on markets."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hmm import HMMFilter, baum_welch, forward_backward, viterbi


def _synthetic(T=4000, seed=7):
    """Generate returns from a KNOWN 2-state HMM (calm vs turbulent)."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    mu = np.array([0.0005, -0.001])
    sigma = np.array([0.006, 0.025])
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = rng.random() > A[s[t - 1], 0] if s[t - 1] == 0 else (
            0 if rng.random() < A[1, 0] else 1)
    r = rng.normal(mu[s], sigma[s])
    return r, s, A, mu, sigma


def test_em_loglik_monotone():
    r, *_ = _synthetic(T=1500)
    lls = []
    fit = None
    for n in (2, 5, 15, 40):
        fit = baum_welch(r, n_states=2, n_iter=n)
        lls.append(fit["loglik"])
    assert all(b >= a - 1e-6 for a, b in zip(lls, lls[1:]))


def test_recovers_known_states():
    r, s_true, A_true, mu_true, sigma_true = _synthetic()
    fit = baum_welch(r, n_states=2)
    # parameter recovery (states ordered calm-first by construction)
    assert abs(fit["sigma"][0] - sigma_true[0]) / sigma_true[0] < 0.15
    assert abs(fit["sigma"][1] - sigma_true[1]) / sigma_true[1] < 0.15
    assert fit["A"][0, 0] > 0.9 and fit["A"][1, 1] > 0.85
    # path recovery
    path = viterbi(r, fit["pi"], fit["A"], fit["mu"], fit["sigma"])
    acc = (path == s_true).mean()
    assert acc > 0.85, f"Viterbi accuracy {acc:.2f}"


def test_gamma_rows_sum_to_one():
    r, *_ = _synthetic(T=800)
    fit = baum_welch(r, n_states=3, n_iter=30)
    assert np.allclose(fit["gamma"].sum(axis=1), 1.0)
    assert np.allclose(fit["A"].sum(axis=1), 1.0)


def test_filter_matches_forward_pass():
    """Online filter must reproduce the batch filtered probabilities."""
    r, *_ = _synthetic(T=600)
    fit = baum_welch(r, n_states=2, n_iter=40)
    _, _, _, alpha_last = forward_backward(
        r, fit["pi"], fit["A"], fit["mu"], fit["sigma"])
    f = HMMFilter(fit["pi"] * 0 + fit["pi"], fit["A"], fit["mu"], fit["sigma"])
    # batch alpha starts from pi*B0; replicate: initialise then update on r[1:]
    b0 = np.exp(-0.5 * ((r[0] - fit["mu"]) / fit["sigma"]) ** 2) / (
        fit["sigma"] * np.sqrt(2 * np.pi))
    f.p = fit["pi"] * b0
    f.p /= f.p.sum()
    for x in r[1:]:
        f.update(x)
    assert np.allclose(f.p, alpha_last, atol=1e-8)


def test_predictive_variance_positive_and_sane():
    r, *_ = _synthetic(T=800)
    fit = baum_welch(r, n_states=2, n_iter=40)
    f = HMMFilter(fit["pi"], fit["A"], fit["mu"], fit["sigma"])
    for x in r[:100]:
        f.update(x)
    v = f.predict_next_var()
    assert v > 0
    assert fit["sigma"][0] ** 2 * 0.5 < v < fit["sigma"][1] ** 2 * 2
