"""
evaluate2.py -- second-round evaluation machinery:

  fit_garch            : GARCH(1,1) with Normal or Student-t innovations via
                         the `arch` package (MLE); returns params + nu
  garch_recursion_var  : fixed-parameter daily variance recursion
  t_logscore / t_var / t_es : density, VaR and ES for scaled-t innovations
  gaussian_es          : closed-form Gaussian expected shortfall
  pit_gaussian / pit_t / pit_mixture : probability integral transforms
  berkowitz_lr         : Berkowitz (2001) density-calibration LR test
  es_ratio             : realised/predicted ES ratio on VaR-breach days
  model_confidence_set : Hansen-Lunde-Nason (2011) MCS by block bootstrap
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from arch import arch_model

_SCALE = 100.0  # arch prefers percent returns


def fit_garch(r, dist="normal"):
    """GARCH(1,1), zero mean. dist in {'normal','t'}. Returns dict."""
    am = arch_model(np.asarray(r) * _SCALE, mean="Zero", vol="GARCH",
                    p=1, q=1, dist=dist)
    res = am.fit(disp="off", show_warning=False)
    p = res.params
    out = dict(omega=p["omega"] / _SCALE ** 2, alpha=p["alpha[1]"],
               beta=p["beta[1]"],
               nu=float(p["nu"]) if "nu" in p else np.inf)
    out["v_last"] = float(res.conditional_volatility[-1] ** 2) / _SCALE ** 2
    return out


def garch_recursion_var(params, r_prev, v_prev):
    return params["omega"] + params["alpha"] * r_prev ** 2 + params["beta"] * v_prev


# --- scaled-t helpers: innovations t(nu) standardised to unit variance ----
def _t_scale(v, nu):
    """Scale s such that s*t(nu) has variance v."""
    return np.sqrt(v * (nu - 2) / nu)


def t_logscore(r, v, nu):
    s = _t_scale(v, nu)
    return stats.t.logpdf(r / s, df=nu) - np.log(s)


def t_var(v, nu, alpha):
    return stats.t.ppf(alpha, df=nu) * _t_scale(v, nu)


def t_es(v, nu, alpha):
    """Closed-form ES of a scaled t at level alpha (a negative number)."""
    q = stats.t.ppf(alpha, df=nu)
    es_std = -(stats.t.pdf(q, df=nu) / alpha) * (nu + q ** 2) / (nu - 1)
    return es_std * _t_scale(v, nu)


def gaussian_es(v, alpha):
    z = stats.norm.ppf(alpha)
    return -stats.norm.pdf(z) / alpha * np.sqrt(v)


# --- PIT and Berkowitz ------------------------------------------------------
def pit_gaussian(r, v):
    return stats.norm.cdf(r / np.sqrt(np.maximum(v, 1e-12)))


def pit_t(r, v, nu):
    return stats.t.cdf(r / _t_scale(np.maximum(v, 1e-12), nu), df=nu)


def pit_mixture_gauss(r, q, mu, sigma):
    return (q * stats.norm.cdf((r - mu) / sigma)).sum()


def pit_mixture_t(r, q, mu, sigma, nu):
    return (q * stats.t.cdf((r - mu) / sigma, df=nu)).sum()


def berkowitz_lr(pit):
    """Berkowitz (2001): z = Phi^{-1}(PIT) should be iid N(0,1) under a
    correctly calibrated density. LR test of mu=0, var=1, rho=0 (3 dof)."""
    z = stats.norm.ppf(np.clip(pit, 1e-10, 1 - 1e-10))
    n = len(z)
    z0, z1 = z[:-1], z[1:]
    rho = np.corrcoef(z0, z1)[0, 1]
    resid = z1 - rho * z0
    mu = resid.mean() / (1 - rho) if abs(1 - rho) > 1e-8 else resid.mean()
    s2 = np.var(z1 - mu * (1 - rho) - rho * z0, ddof=0)
    llf = (-0.5 * np.log(2 * np.pi * s2)
           - ((z1 - mu * (1 - rho) - rho * z0) ** 2 / (2 * s2))).sum()
    ll0 = (-0.5 * np.log(2 * np.pi) - z1 ** 2 / 2).sum()
    lr = -2 * (ll0 - llf)
    return lr, 1 - stats.chi2.cdf(lr, df=3)


def es_ratio(r, var_pred, es_pred):
    """Mean realised loss beyond VaR divided by mean predicted ES on breach
    days. 1.0 = perfectly calibrated ES; >1 = ES too optimistic."""
    hits = r < var_pred
    if hits.sum() < 5:
        return np.nan, int(hits.sum())
    return float(r[hits].mean() / es_pred[hits].mean()), int(hits.sum())


# --- Model Confidence Set ---------------------------------------------------
def model_confidence_set(losses, alpha=0.10, n_boot=1000, mean_block=21,
                         seed=0):
    """Hansen-Lunde-Nason (2011) MCS with the T_max statistic and a
    stationary block bootstrap. `losses`: dict name -> loss array (same len).
    Returns dict name -> (in_set: bool, elimination p-value)."""
    names = list(losses)
    L = np.column_stack([losses[m] for m in names])
    T, M0 = L.shape
    rng = np.random.default_rng(seed)
    p_geom = 1.0 / mean_block

    # pre-generate bootstrap index matrix once (shared across iterations)
    B_idx = np.empty((n_boot, T), dtype=int)
    for b in range(n_boot):
        i = rng.integers(0, T)
        for t in range(T):
            B_idx[b, t] = i
            i = rng.integers(0, T) if rng.random() < p_geom else (i + 1) % T

    active = list(range(M0))
    result = {}
    while len(active) > 1:
        La = L[:, active]
        dbar_i = La.mean(axis=0) - La.mean()          # loss vs set average
        # bootstrap distribution of the max t-stat
        boot_stats = np.empty(n_boot)
        centred = La - La.mean(axis=0, keepdims=True)
        for b in range(n_boot):
            Lb = centred[B_idx[b]]
            db = Lb.mean(axis=0) - Lb.mean()
            vb = np.maximum((db ** 2).std(), 1e-12)
            boot_stats[b] = np.abs(db).max()
        stat = np.abs(dbar_i).max()
        p = float((boot_stats >= stat).mean())
        worst = active[int(np.argmax(dbar_i))]
        if p < alpha:
            result[names[worst]] = (False, p)
            active.remove(worst)
        else:
            for m in active:
                result[names[m]] = (True, p)
            break
    if len(active) == 1:
        result[names[active[0]]] = (True, 1.0)
    return result


# --- analytic mixture VaR / ES (fast paths for the walk-forward) -----------
def mixture_var_generic(q, mu, sigma, alpha, nu=None):
    """alpha-quantile of a Gaussian (nu=None) or t mixture by bisection."""
    lo = float((mu - 30 * sigma).min())
    hi = float((mu + 30 * sigma).max())
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        z = (mid - mu) / sigma
        cdf = (q * (stats.norm.cdf(z) if nu is None
                    else stats.t.cdf(z, df=nu))).sum()
        if cdf < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mixture_es_generic(q, mu, sigma, alpha, nu=None, var_q=None):
    """Closed-form ES of a Gaussian or t mixture:
    ES = (1/alpha) * sum_k q_k * E[X 1{X < c}] for component k, with
    E[X 1{X<c}] = mu_k F(z) - sigma_k * psi(z), where for the Gaussian
    psi(z) = phi(z) and for the t  psi(z) = f_nu(z) (nu + z^2)/(nu - 1)."""
    c = mixture_var_generic(q, mu, sigma, alpha, nu) if var_q is None else var_q
    z = (c - mu) / sigma
    if nu is None:
        F = stats.norm.cdf(z)
        psi = stats.norm.pdf(z)
    else:
        F = stats.t.cdf(z, df=nu)
        psi = stats.t.pdf(z, df=nu) * (nu + z ** 2) / (nu - 1)
    partial = mu * F - sigma * psi
    return float((q * partial).sum() / alpha)
