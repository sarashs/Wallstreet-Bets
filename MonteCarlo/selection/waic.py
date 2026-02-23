"""
selection/waic.py
-----------------
WAIC (Widely Applicable Information Criterion) computation and model selection.

WAIC = -2 × (lppd − p_waic)

where:
  lppd   = Σ_i log( (1/S) Σ_s exp(log_lik[s, i]) )   [log pointwise predictive density]
  p_waic = Σ_i Var_s(log_lik[s, i])                    [effective number of parameters]

Lower WAIC = better predictive performance.

Decision rule (per spec §5.3):
  delta_waic = waic_B - waic_A
  if delta_waic < -4  →  select Model B
  if delta_waic >  4  →  select Model A
  else               →  select Model A (prefer simpler)

Public API
----------
compute_waic(log_lik_matrix)          -> dict
compare_models(waic_a, waic_b)        -> dict
compute_log_lik_matrix(model, posterior_samples, returns, n_sub) -> np.ndarray
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

WAIC_THRESHOLD = 4.0   # minimum ΔWAIC to claim strong preference


# ---------------------------------------------------------------------------
# Core WAIC computation
# ---------------------------------------------------------------------------

def compute_waic(log_lik_matrix: np.ndarray) -> dict:
    """Compute WAIC from a matrix of pointwise log-likelihoods.

    Parameters
    ----------
    log_lik_matrix : np.ndarray, shape (S, n_obs)
        log_lik_matrix[s, i] = log p(y_i | θ^(s))
        S = number of posterior samples, n_obs = total observations (N × T).

    Returns
    -------
    dict with keys:
        "waic"     : float — WAIC value
        "lppd"     : float
        "p_waic"   : float
        "waic_i"   : np.ndarray (n_obs,) — pointwise WAIC contributions
        "lppd_i"   : np.ndarray (n_obs,)
        "p_waic_i" : np.ndarray (n_obs,)
        "se"       : float — standard error of WAIC (via pointwise)
    """
    S, n_obs = log_lik_matrix.shape
    logger.debug("Computing WAIC: S=%d samples, n_obs=%d", S, n_obs)

    # lppd_i = log( mean_s exp(log_lik[s,i]) ) — use log-sum-exp for stability
    log_mean_lik = _log_mean_exp(log_lik_matrix, axis=0)   # (n_obs,)

    # p_waic_i = Var_s(log_lik[s,i])
    p_waic_i = log_lik_matrix.var(axis=0, ddof=1)          # (n_obs,)

    waic_i = -2.0 * (log_mean_lik - p_waic_i)              # (n_obs,)

    lppd   = log_mean_lik.sum()
    p_waic = p_waic_i.sum()
    waic   = -2.0 * (lppd - p_waic)

    # Standard error of WAIC (over pointwise contributions)
    se = np.sqrt(n_obs * waic_i.var(ddof=1))

    logger.info("WAIC=%.2f, lppd=%.2f, p_waic=%.2f, SE=%.2f", waic, lppd, p_waic, se)
    return {
        "waic"    : float(waic),
        "lppd"    : float(lppd),
        "p_waic"  : float(p_waic),
        "waic_i"  : waic_i,
        "lppd_i"  : log_mean_lik,
        "p_waic_i": p_waic_i,
        "se"      : float(se),
    }


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(waic_a: dict, waic_b: dict) -> dict:
    """Compare WAIC of Model A vs Model B and select the winner.

    Parameters
    ----------
    waic_a, waic_b : dicts returned by compute_waic()

    Returns
    -------
    dict with keys:
        "selected"     : str  — "A" or "B"
        "delta_waic"   : float — waic_B - waic_A  (positive → A wins)
        "se_delta"     : float — standard error of the difference
        "uncertain"    : bool  — |delta| < 2 × se_delta
        "rationale"    : str  — human-readable explanation
    """
    delta = waic_b["waic"] - waic_a["waic"]

    # SE of pointwise difference (see spec §5.3)
    diff_i  = waic_b["waic_i"] - waic_a["waic_i"]
    n       = len(diff_i)
    se_diff = np.sqrt(n * diff_i.var(ddof=1))

    uncertain = abs(delta) < 2.0 * se_diff

    if delta < -WAIC_THRESHOLD:
        selected = "B"
        rationale = (
            f"Model B (Normal mixture) is strongly preferred: ΔWAIC = {delta:.2f} "
            f"(SE={se_diff:.2f}).  Evidence for regime-switching behavior."
        )
    elif delta > WAIC_THRESHOLD:
        selected = "A"
        rationale = (
            f"Model A (Student-t) is strongly preferred: ΔWAIC = {delta:.2f} "
            f"(SE={se_diff:.2f}).  Single fat-tailed regime sufficient."
        )
    else:
        selected = "A"
        if uncertain:
            rationale = (
                f"Models are comparable (ΔWAIC = {delta:.2f}, SE={se_diff:.2f}); "
                f"uncertainty is large (|Δ| < 2·SE).  Defaulting to simpler Model A."
            )
        else:
            rationale = (
                f"Models are comparable (ΔWAIC = {delta:.2f}, SE={se_diff:.2f}).  "
                f"Defaulting to simpler Model A (Student-t)."
            )

    logger.info("Model selection: %s | %s", selected, rationale)

    return {
        "selected"   : selected,
        "delta_waic" : float(delta),
        "se_delta"   : float(se_diff),
        "uncertain"  : uncertain,
        "rationale"  : rationale,
        "waic_a"     : waic_a["waic"],
        "waic_b"     : waic_b["waic"],
    }


# ---------------------------------------------------------------------------
# Build log-likelihood matrix from posterior samples
# ---------------------------------------------------------------------------

def compute_log_lik_matrix(
    model,
    posterior_samples: np.ndarray,   # (S, D) flattened parameter vectors
    returns: np.ndarray,              # (N, T)
    n_sub: int = 500,                 # subsample posterior to keep memory manageable
) -> np.ndarray:
    """Reconstruct parameter states from samples and compute log-likelihoods.

    Parameters
    ----------
    model : ModelA | ModelB
    posterior_samples : np.ndarray, shape (S, D)
    returns : np.ndarray, shape (N, T)
    n_sub : int
        Number of posterior draws to use for WAIC (subsampled for speed).

    Returns
    -------
    np.ndarray, shape (n_sub, N*T)
    """
    S, D = posterior_samples.shape
    N, T = returns.shape

    # Subsample to reduce cost
    idx     = np.random.choice(S, size=min(n_sub, S), replace=False)
    samples = posterior_samples[idx]

    log_lik = np.zeros((len(samples), N * T))

    logger.info("Computing pointwise log-likelihoods for WAIC (%d draws)...", len(samples))

    for j, vec in enumerate(samples):
        state = _vec_to_state(model, vec, N)
        ll    = model.log_lik_pointwise(state, returns)   # (N, T)
        log_lik[j] = ll.ravel()

        if (j + 1) % 100 == 0:
            logger.debug("  WAIC log-lik: %d / %d", j + 1, len(samples))

    return log_lik


# ---------------------------------------------------------------------------
# Model selection convenience wrapper
# ---------------------------------------------------------------------------

def select_model(
    model_a,
    model_b,
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    returns: np.ndarray,
    n_sub: int = 500,
) -> dict:
    """End-to-end model selection: compute WAIC for both models and compare.

    Returns
    -------
    dict with "comparison" (from compare_models) plus "waic_a" and "waic_b" dicts.
    """
    logger.info("Computing WAIC for Model A...")
    ll_a  = compute_log_lik_matrix(model_a, samples_a, returns, n_sub)
    waic_a = compute_waic(ll_a)

    logger.info("Computing WAIC for Model B...")
    ll_b   = compute_log_lik_matrix(model_b, samples_b, returns, n_sub)
    waic_b = compute_waic(ll_b)

    comparison = compare_models(waic_a, waic_b)

    return {
        "comparison" : comparison,
        "waic_a"     : waic_a,
        "waic_b"     : waic_b,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_mean_exp(x: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log( mean( exp(x) ) ) along axis."""
    c = x.max(axis=axis, keepdims=True)
    return np.log(np.exp(x - c).mean(axis=axis)) + c.squeeze(axis=axis)


def _vec_to_state(model, vec: np.ndarray, N: int) -> dict:
    """Reconstruct a parameter-state dict from a flattened vector.

    The layout must match model.state_to_vector() / model.param_names().
    """
    if model.name == "A":
        mu_i    = vec[:N]
        sigma_i = vec[N:2*N]
        nu, mu_0, tau, nu_s, xi = vec[2*N:]

        return dict(mu_i=mu_i, sigma_i=sigma_i, nu=nu, mu_0=mu_0,
                    tau=tau, nu_s=nu_s, xi=xi, lambda_it=None)

    else:  # B
        p  = 0
        mu_ik    = vec[p:p+2*N].reshape(N, 2);  p += 2*N
        sigma_ik = vec[p:p+2*N].reshape(N, 2);  p += 2*N
        pi_i     = vec[p:p+N];                  p += N
        mu_0k    = vec[p:p+2];                  p += 2
        tau_k    = vec[p:p+2];                  p += 2
        nu_k     = vec[p:p+2];                  p += 2
        xi_k     = vec[p:p+2];                  p += 2
        alpha_pi, beta_pi = vec[p], vec[p+1]

        # Re-order mu_ik from model.state_to_vector layout (which is [k0_all, k1_all])
        mu_ik_r    = mu_ik.T.reshape(2, N).T     # back to (N, 2)
        sigma_ik_r = sigma_ik.T.reshape(2, N).T

        return dict(mu_ik=mu_ik_r, sigma_ik=sigma_ik_r, pi_i=pi_i,
                    mu_0k=mu_0k, tau_k=tau_k, nu_k=nu_k, xi_k=xi_k,
                    alpha_pi=alpha_pi, beta_pi=beta_pi, z_it=None)
