"""
mcmc/diagnostics.py
-------------------
Convergence diagnostics for MCMC chains.

Implements:
  - Split-chain R-hat (Gelman & Rubin 1992, updated Vehtari et al. 2021)
  - Bulk / tail effective sample size (ESS)
  - Summary table generation

Public API
----------
compute_rhat(chain_samples)  -> np.ndarray   shape (D,)
compute_ess(chain_samples)   -> np.ndarray   shape (D,)
make_diagnostics_table(chain_samples, param_names) -> dict
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# R-hat (split-chain version per Vehtari et al. 2021)
# ---------------------------------------------------------------------------

def compute_rhat(chain_samples: list[np.ndarray]) -> np.ndarray:
    """Compute split-chain R-hat for each parameter.

    Parameters
    ----------
    chain_samples : list of np.ndarray, each shape (n_stored, D)
        One array per chain.

    Returns
    -------
    np.ndarray, shape (D,) — R-hat values.
        Values close to 1.0 indicate convergence.  Threshold: < 1.05.
    """
    # Split each chain in half to get 2 × n_chains sub-chains
    sub_chains = []
    for ch in chain_samples:
        n = len(ch)
        half = n // 2
        sub_chains.append(ch[:half])
        sub_chains.append(ch[half: 2 * half])

    M = len(sub_chains)       # number of sub-chains
    N = sub_chains[0].shape[0]   # length of each sub-chain

    # Stack: (M, N, D)
    chains_arr = np.stack([sc[:N] for sc in sub_chains], axis=0)
    D = chains_arr.shape[2]

    # Between-chain variance B
    chain_means = chains_arr.mean(axis=1)    # (M, D)
    grand_mean  = chain_means.mean(axis=0)   # (D,)
    B = N * np.var(chain_means, axis=0, ddof=1)   # (D,)

    # Within-chain variance W
    chain_vars = chains_arr.var(axis=1, ddof=1)   # (M, D)
    W = chain_vars.mean(axis=0)                    # (D,)

    # Marginal posterior variance estimator
    var_hat = (N - 1) / N * W + B / N              # (D,)

    rhat = np.sqrt(var_hat / np.maximum(W, 1e-12))

    logger.debug("R-hat: min=%.4f, max=%.4f, >1.05: %d/%d",
                 rhat.min(), rhat.max(), (rhat > 1.05).sum(), D)
    return rhat


# ---------------------------------------------------------------------------
# Effective sample size (bulk ESS via rank-normalisation)
# ---------------------------------------------------------------------------

def compute_ess(chain_samples: list[np.ndarray]) -> np.ndarray:
    """Compute bulk ESS for each parameter using rank-normalisation.

    A simplified version: ESS ≈ S / (1 + 2 × Σ_k ρ_k) where ρ_k is the
    autocorrelation at lag k (estimated from the combined chain).

    Parameters
    ----------
    chain_samples : list of np.ndarray, each shape (n_stored, D)

    Returns
    -------
    np.ndarray, shape (D,) — effective sample sizes.
    """
    combined = np.concatenate(chain_samples, axis=0)  # (S, D)
    S, D = combined.shape

    ess = np.zeros(D)

    for d in range(D):
        x    = combined[:, d]
        x    = (x - x.mean()) / (x.std() + 1e-12)   # standardise
        rho  = _autocorr_sum(x, max_lag=min(S // 2, 500))
        ess[d] = S / (1.0 + 2.0 * rho)

    ess = np.clip(ess, 1.0, S)

    logger.debug("ESS: min=%.1f, max=%.1f, <400: %d/%d",
                 ess.min(), ess.max(), (ess < 400).sum(), D)
    return ess


def _autocorr_sum(x: np.ndarray, max_lag: int) -> float:
    """Compute sum of positive autocorrelations (pair criterion stops at first negative pair)."""
    n     = len(x)
    acf   = np.correlate(x, x, mode="full")[n - 1:]  / n
    acf   /= acf[0]   # normalise by lag-0

    rho_sum = 0.0
    for k in range(1, max_lag, 2):   # pairs (k, k+1)
        if k + 1 >= len(acf):
            break
        pair = acf[k] + acf[k + 1]
        if pair <= 0.0:
            break
        rho_sum += pair

    return max(0.0, rho_sum)


# ---------------------------------------------------------------------------
# Diagnostics summary table
# ---------------------------------------------------------------------------

def make_diagnostics_table(
    chain_samples: list[np.ndarray],
    param_names: list[str],
) -> dict:
    """Compute R-hat, ESS, posterior mean, and posterior std for all parameters.

    Returns
    -------
    dict with keys:
        "param"  : list[str]
        "mean"   : np.ndarray
        "std"    : np.ndarray
        "rhat"   : np.ndarray
        "ess"    : np.ndarray
        "rhat_ok": np.ndarray (bool) — True if R-hat < 1.05
        "ess_ok" : np.ndarray (bool) — True if ESS > 400
    """
    combined = np.concatenate(chain_samples, axis=0)

    rhat = compute_rhat(chain_samples)
    ess  = compute_ess(chain_samples)

    n_params = combined.shape[1]
    names    = param_names[:n_params]  # guard against length mismatch

    diag = {
        "param" : names,
        "mean"  : combined.mean(axis=0),
        "std"   : combined.std(axis=0),
        "rhat"  : rhat,
        "ess"   : ess,
        "rhat_ok": rhat < 1.05,
        "ess_ok" : ess > 400,
    }

    n_bad_rhat = (~diag["rhat_ok"]).sum()
    n_bad_ess  = (~diag["ess_ok"]).sum()
    logger.info(
        "Diagnostics: %d params with R-hat > 1.05, %d with ESS < 400",
        n_bad_rhat, n_bad_ess,
    )
    return diag
