"""
hmm_t.py -- Hidden Markov Model with Student-t emissions.

Same latent structure as the Gaussian HMM in hmm.py, but each state emits
r_t ~ mu_k + sigma_k * t(nu_k): fat tails INSIDE each regime, so the model
can separate "regime got riskier" from "a single outlier landed".

EM uses the classical normal/inverse-gamma representation of the t
distribution: given state responsibilities gamma_{tk} (forward-backward,
unchanged), the extra E-step weight is

    u_{tk} = (nu_k + 1) / (nu_k + z_{tk}^2),   z = (r - mu_k)/sigma_k,

and the M-step becomes a weighted mean/variance; nu_k is updated every few
iterations by profile likelihood (1-D bounded search). See Peel & McLachlan
(2000) for the mixture case; the HMM case only swaps mixing weights for
smoothed state probabilities.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

_SIGMA_FLOOR = 1e-5
_NU_LO, _NU_HI = 2.1, 60.0


def _t_pdf(r, mu, sigma, nu):
    """Location-scale t pdf, vectorised to (T, K)."""
    z = (r[:, None] - mu) / sigma
    c = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
         - 0.5 * np.log(np.pi * nu) - np.log(sigma))
    return np.exp(c - 0.5 * (nu + 1) * np.log1p(z ** 2 / nu))


def _forward_backward_t(r, pi, A, mu, sigma, nu):
    T, K = len(r), len(pi)
    B = _t_pdf(r, mu, sigma, nu) + 1e-300
    alpha = np.zeros((T, K)); c = np.zeros(T)
    alpha[0] = pi * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        c[t] = alpha[t].sum(); alpha[t] /= c[t]
    beta = np.zeros((T, K)); beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        xi_sum += (alpha[t][:, None] * A
                   * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1])
    return np.log(c).sum(), gamma, xi_sum, alpha[-1]


def _profile_nu(r, mu_k, sigma_k, gamma_k):
    """Weighted t log-likelihood in nu, maximised on [_NU_LO, _NU_HI]."""
    z2 = ((r - mu_k) / sigma_k) ** 2

    def nll(nu):
        ll = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
              - 0.5 * np.log(np.pi * nu) - np.log(sigma_k)
              - 0.5 * (nu + 1) * np.log1p(z2 / nu))
        return -(gamma_k * ll).sum()

    res = optimize.minimize_scalar(nll, bounds=(_NU_LO, _NU_HI),
                                   method="bounded",
                                   options={"xatol": 0.05})
    return float(res.x)


def baum_welch_t(r, n_states=3, n_iter=60, tol=1e-5, nu_update_every=5):
    """Fit a Student-t HMM by EM. Returns dict(pi, A, mu, sigma, nu, loglik)."""
    r = np.asarray(r, dtype=float)
    K = n_states
    q = np.quantile(np.abs(r), np.linspace(0, 1, K + 1))
    labels = np.clip(np.searchsorted(q, np.abs(r), side="right") - 1, 0, K - 1)
    mu = np.array([r[labels == k].mean() for k in range(K)])
    sigma = np.array([max(r[labels == k].std(), _SIGMA_FLOOR) for k in range(K)])
    nu = np.full(K, 8.0)
    pi = np.full(K, 1.0 / K)
    A = np.full((K, K), 0.05 / (K - 1)); np.fill_diagonal(A, 0.95)

    prev_ll = -np.inf
    for it in range(n_iter):
        ll, gamma, xi_sum, _ = _forward_backward_t(r, pi, A, mu, sigma, nu)
        pi = gamma[0] / gamma[0].sum()
        A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        z2 = ((r[:, None] - mu) / sigma) ** 2
        u = (nu + 1) / (nu + z2)                    # (T, K) tail weights
        gu = gamma * u
        mu = (gu * r[:, None]).sum(axis=0) / gu.sum(axis=0)
        var = (gu * (r[:, None] - mu) ** 2).sum(axis=0) / gamma.sum(axis=0)
        sigma = np.maximum(np.sqrt(var), _SIGMA_FLOOR)
        if (it + 1) % nu_update_every == 0:
            for k in range(K):
                nu[k] = _profile_nu(r, mu[k], sigma[k], gamma[:, k])
        if ll - prev_ll < tol and it > 10:
            break
        prev_ll = ll

    order = np.argsort(sigma * np.sqrt(np.where(nu > 2, nu / (nu - 2), 10)))
    pi, mu, sigma, nu = pi[order], mu[order], sigma[order], nu[order]
    A = A[np.ix_(order, order)]
    return dict(pi=pi, A=A, mu=mu, sigma=sigma, nu=nu, loglik=ll,
                n_iter_used=it + 1)


class TFilter:
    """Online filtered probabilities for the t-HMM (no look-ahead)."""

    def __init__(self, pi, A, mu, sigma, nu):
        self.A, self.mu, self.sigma, self.nu = A, mu, sigma, nu
        self.p = pi.copy()

    def update(self, r_t):
        b = _t_pdf(np.array([r_t]), self.mu, self.sigma, self.nu)[0] + 1e-300
        p = (self.p @ self.A) * b
        self.p = p / p.sum()
        return self.p

    def predict_state_probs(self):
        return self.p @ self.A

    def predict_next_var(self):
        q = self.predict_state_probs()
        v_k = self.sigma ** 2 * np.where(self.nu > 2,
                                         self.nu / (self.nu - 2), 10.0)
        m1 = (q * self.mu).sum()
        m2 = (q * (v_k + self.mu ** 2)).sum()
        return m2 - m1 ** 2

    def logscore(self, r_t):
        q = self.predict_state_probs()
        pdf = _t_pdf(np.array([r_t]), self.mu, self.sigma, self.nu)[0]
        return np.log(max((q * pdf).sum(), 1e-300))

    def var_quantile(self, alpha):
        """alpha-quantile of the predictive t mixture by bisection."""
        q = self.predict_state_probs()
        scale = self.sigma
        lo = float((self.mu - 30 * scale).min())
        hi = float((self.mu + 30 * scale).max())
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            cdf = (q * stats.t.cdf((mid - self.mu) / scale, df=self.nu)).sum()
            if cdf < alpha:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def expected_shortfall(self, alpha, n_grid=20000):
        """ES of the predictive mixture by numerical tail integration."""
        var_q = self.var_quantile(alpha)
        lo = float((self.mu - 40 * self.sigma).min())
        x = np.linspace(lo, var_q, n_grid)
        q = self.predict_state_probs()
        pdf = (q * stats.t.pdf((x[:, None] - self.mu) / self.sigma,
                               df=self.nu) / self.sigma).sum(axis=1)
        mass = np.trapezoid(pdf, x)
        if mass <= 0:
            return var_q
        return float(np.trapezoid(x * pdf, x) / mass)
