"""
hmm.py -- Gaussian Hidden Markov Model, implemented from scratch.

This is the Renaissance-lineage math: Leonard Baum (Jim Simons' first partner)
co-invented the Baum-Welch algorithm implemented here. The market regime is
treated as a HIDDEN state -- never observed, only inferred from returns.

Components (all standard, no libraries beyond numpy):
  forward_backward : scaled alpha/beta recursions -> log-likelihood,
                     smoothed state probabilities gamma, pair counts xi
  baum_welch       : EM -- iterate E-step (forward-backward) and M-step
                     (reestimate pi, A, mu_k, sigma_k) until the
                     log-likelihood stops improving
  viterbi          : most likely hidden state PATH (log-space DP)
  HMMFilter        : online filtered probabilities P(s_t | r_1..r_t) for
                     out-of-sample prediction without look-ahead

Emissions are univariate Gaussians N(mu_k, sigma_k^2) on daily returns.
States are relabelled by ascending sigma_k after fitting, so state 0 is
always the calmest and state K-1 the most turbulent.
"""
from __future__ import annotations

import numpy as np

_SIGMA_FLOOR = 1e-5  # keeps a state from collapsing onto a single point


def _emission_probs(r, mu, sigma):
    """Gaussian pdf of each observation under each state: shape (T, K)."""
    r = r[:, None]
    return np.exp(-0.5 * ((r - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def forward_backward(r, pi, A, mu, sigma):
    """Scaled forward-backward. Returns (loglik, gamma, xi_sum, alpha_last).

    gamma[t, k] = P(s_t = k | all data)          (smoothed)
    xi_sum[i,j] = sum_t P(s_t=i, s_{t+1}=j | data)  (for the A update)
    alpha_last  = filtered P(s_T = k | data)      (for prediction)
    """
    T, K = len(r), len(pi)
    B = _emission_probs(r, mu, sigma)

    alpha = np.zeros((T, K))
    c = np.zeros(T)  # scaling factors
    alpha[0] = pi * B[0]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]

    beta = np.zeros((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        x = alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1]
        xi_sum += x

    return np.log(c).sum(), gamma, xi_sum, alpha[-1]


def baum_welch(r, n_states=3, n_iter=200, tol=1e-6, seed=0):
    """Fit a Gaussian HMM by EM. Deterministic quantile-based initialisation
    (no random restarts needed for these low-dimensional problems).

    Returns dict with pi, A, mu, sigma, loglik, n_iter_used, gamma.
    """
    r = np.asarray(r, dtype=float)
    T = len(r)
    K = n_states

    # init: split days by |r| quantiles -> moment-match each bucket
    q = np.quantile(np.abs(r), np.linspace(0, 1, K + 1))
    labels = np.clip(np.searchsorted(q, np.abs(r), side="right") - 1, 0, K - 1)
    mu = np.array([r[labels == k].mean() for k in range(K)])
    sigma = np.array([max(r[labels == k].std(), _SIGMA_FLOOR) for k in range(K)])
    pi = np.full(K, 1.0 / K)
    A = np.full((K, K), 0.05 / (K - 1))
    np.fill_diagonal(A, 0.95)  # start persistent -- regimes, not noise

    prev_ll = -np.inf
    for it in range(n_iter):
        ll, gamma, xi_sum, _ = forward_backward(r, pi, A, mu, sigma)
        # M-step
        pi = gamma[0] / gamma[0].sum()
        A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        mu = (gamma * r[:, None]).sum(axis=0) / w
        var = (gamma * (r[:, None] - mu) ** 2).sum(axis=0) / w
        sigma = np.maximum(np.sqrt(var), _SIGMA_FLOOR)
        if ll - prev_ll < tol and it > 5:
            break
        prev_ll = ll

    # canonical ordering: state 0 calmest
    order = np.argsort(sigma)
    pi, mu, sigma = pi[order], mu[order], sigma[order]
    A = A[np.ix_(order, order)]
    _, gamma, _, _ = forward_backward(r, pi, A, mu, sigma)
    return dict(pi=pi, A=A, mu=mu, sigma=sigma, loglik=ll,
                n_iter_used=it + 1, gamma=gamma)


def viterbi(r, pi, A, mu, sigma):
    """Most likely hidden state sequence (log-space dynamic programming)."""
    T, K = len(r), len(pi)
    logB = np.log(_emission_probs(np.asarray(r, float), mu, sigma) + 1e-300)
    logA = np.log(A + 1e-300)
    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = np.log(pi + 1e-300) + logB[0]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + logA
        psi[t] = cand.argmax(axis=0)
        delta[t] = cand.max(axis=0) + logB[t]
    path = np.zeros(T, dtype=int)
    path[-1] = delta[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1]]
    return path


class HMMFilter:
    """Online filter: update P(s_t | r_1..r_t) one observation at a time.

    Used out-of-sample so that every prediction at time t uses only
    information available at t (parameters are frozen between refits).
    """

    def __init__(self, pi, A, mu, sigma):
        self.A, self.mu, self.sigma = A, mu, sigma
        self.p = pi.copy()

    def update(self, r_t):
        b = np.exp(-0.5 * ((r_t - self.mu) / self.sigma) ** 2) / (
            self.sigma * np.sqrt(2 * np.pi))
        p = (self.p @ self.A) * b
        s = p.sum()
        self.p = p / s if s > 0 else np.full_like(p, 1 / len(p))
        return self.p

    def predict_next_var(self):
        """Variance of tomorrow's return under the predictive mixture
        q = p_t A:  Var = sum q_k (sigma_k^2 + mu_k^2) - (sum q_k mu_k)^2."""
        q = self.p @ self.A
        m1 = (q * self.mu).sum()
        m2 = (q * (self.sigma ** 2 + self.mu ** 2)).sum()
        return m2 - m1 ** 2

    def predict_state_probs(self):
        return self.p @ self.A
