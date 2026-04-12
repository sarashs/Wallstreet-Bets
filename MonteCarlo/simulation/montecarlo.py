"""
simulation/montecarlo.py
------------------------
Forward Monte Carlo simulation using posterior predictive draws.

Procedure (per spec §6.1):
  For s = 1, ..., S_sim:
    1. Draw θ^(s) from posterior (with replacement)
    2. Simulate T_horizon daily returns per stock using the model predictive
    3. Compute cumulative log-return per stock
    4. Convert to simple return and compute weighted portfolio return

Correlation structure:
  Student-t copula (default) or Gaussian copula based on the Spearman
  rank-correlation matrix.  The t-copula introduces tail dependence so that
  joint extreme moves (crashes) are more probable than under a Gaussian copula.

  Generation (t-copula):
    z ~ MVN(0, I_N)  →  z_corr = L @ z
    w ~ χ²(copula_df) / copula_df   (shared across all N stocks on each day)
    t_corr = z_corr / sqrt(w)       (multivariate-t innovations)

  When copula_df → ∞ this reduces to the Gaussian copula.

Public API
----------
run_forward_simulation(config, posterior_a, posterior_b, returns, spearman_corr,
                       weights, tickers, horizon_days, n_sim, seed,
                       copula_df) -> dict
compute_risk_metrics(portfolio_returns, horizon_days) -> dict
"""

import logging
import numpy as np
from scipy.stats import norm, t as student_t
from scipy.linalg import cholesky

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_forward_simulation(
    selected_model: str,               # "A" or "B"
    posterior_samples: np.ndarray,     # (S, D) — the MCMC draws for selected model
    returns_hist: np.ndarray,          # (N, T) — historical returns (for Spearman copula)
    spearman_corr: np.ndarray,         # (N, N)
    weights: np.ndarray,               # (N,) market-cap weights (will be normalised)
    tickers: list[str],
    horizon_days: int = 252,
    n_sim: int = 10_000,
    seed: int = 0,
    progress_callback=None,
    copula_df: float = 5.0,
) -> dict:
    """Run forward Monte Carlo simulation.

    Returns
    -------
    dict with keys:
        "portfolio_returns"   : np.ndarray (n_sim,) — simulated simple returns
        "stock_returns"       : np.ndarray (n_sim, N) — per-stock simple returns
        "cumulative_paths"    : np.ndarray (n_sim, T+1) — cumulative log-return paths
        "risk_metrics"        : dict — VaR, CVaR, etc.
        "percentile_paths"    : dict — {5, 25, 50, 75, 95} percentile fan-chart paths
        "weights"             : np.ndarray (N,) — normalised weights used
    """
    N = returns_hist.shape[0]
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()   # normalise

    copula_type = "Gaussian" if copula_df is None or copula_df >= 200 else f"t(df={copula_df:.0f})"
    logger.info(
        "Forward simulation: model=%s, N=%d stocks, T=%d days, n_sim=%d, copula=%s",
        selected_model, N, horizon_days, n_sim, copula_type,
    )

    # Cholesky of Spearman correlation for copula
    L = _safe_cholesky(spearman_corr)

    rng = np.random.default_rng(seed)
    S   = posterior_samples.shape[0]

    portfolio_returns = np.zeros(n_sim)
    stock_returns     = np.zeros((n_sim, N))
    # Store daily cumulative portfolio paths for fan chart
    cum_paths = np.zeros((n_sim, horizon_days + 1))

    batch = 250   # process this many simulations at a time (memory-efficient)

    for start in range(0, n_sim, batch):
        end       = min(start + batch, n_sim)
        b_size    = end - start

        # Draw posterior parameter indices (with replacement)
        idx = rng.integers(0, S, size=b_size)

        for j, s_idx in enumerate(idx):
            sim_idx = start + j
            params  = posterior_samples[s_idx]

            if selected_model == "A":
                daily_r = _simulate_model_a(params, N, horizon_days, L, rng, copula_df)
            else:
                daily_r = _simulate_model_b(params, N, horizon_days, L, rng, copula_df)

            # Cumulative log-return path (portfolio level)
            log_r_port  = (daily_r * w[:, None]).sum(axis=0)   # (T,) daily portfolio log-ret
            cum_log     = np.concatenate([[0.0], np.cumsum(log_r_port)])  # (T+1,)
            cum_paths[sim_idx] = cum_log

            # Terminal returns per stock
            cum_log_stock  = daily_r.sum(axis=1)               # (N,) cumulative log-return
            total_r_stock  = np.exp(cum_log_stock) - 1.0       # (N,) simple return

            stock_returns[sim_idx]     = total_r_stock
            portfolio_returns[sim_idx] = (w * total_r_stock).sum()

        if progress_callback is not None:
            progress_callback(end / n_sim)

        logger.debug("Simulation batch %d/%d done", end, n_sim)

    risk = compute_risk_metrics(portfolio_returns, horizon_days)
    percentile_paths = _fan_chart_percentiles(cum_paths)

    logger.info(
        "Simulation complete. E[R]=%.2f%%, VaR(5%%)=%.2f%%, CVaR=%.2f%%",
        risk["expected_return"] * 100,
        risk["var_5pct"] * 100,
        risk["cvar_5pct"] * 100,
    )

    return {
        "portfolio_returns"  : portfolio_returns,
        "stock_returns"      : stock_returns,
        "cumulative_paths"   : cum_paths,
        "risk_metrics"       : risk,
        "percentile_paths"   : percentile_paths,
        "weights"            : w,
        "tickers"            : tickers,
    }


# ---------------------------------------------------------------------------
# Per-model daily return simulators
# ---------------------------------------------------------------------------

def _simulate_model_a(
    params: np.ndarray,
    N: int,
    T: int,
    L: np.ndarray,
    rng: np.random.Generator,
    copula_df: float | None = 5.0,
) -> np.ndarray:
    """Simulate T daily log-returns for N stocks under Model A (Student-t).

    Uses data-augmentation representation:
        λ_it ~ Gamma(ν/2, ν/2)
        r_it = μ_i + σ_i * z_corr_it / sqrt(λ_it)
    where z_corr are copula-correlated innovations (t-copula or Gaussian).

    Returns
    -------
    np.ndarray, shape (N, T)
    """
    mu_i    = params[:N]
    sigma_i = params[N:2*N]
    nu      = params[2*N]

    # Correlated innovations via copula
    z_corr = _copula_innovations(T, N, L, rng, copula_df)   # (T, N)

    # Scale-mixture (Student-t via Gamma augmentation) — per-stock, independent
    lam     = rng.gamma(nu / 2.0, 2.0 / nu, size=(T, N))   # (T, N)

    daily_r = mu_i + sigma_i * z_corr / np.sqrt(lam)        # (T, N)
    return daily_r.T   # (N, T)


def _simulate_model_b(
    params: np.ndarray,
    N: int,
    T: int,
    L: np.ndarray,
    rng: np.random.Generator,
    copula_df: float | None = 5.0,
) -> np.ndarray:
    """Simulate T daily log-returns for N stocks under Model B (Normal mixture).

    Each day:
        k_it ~ Bernoulli(π_i)
        r_it = μ_i,k + σ_i,k * z_corr_it
    where z_corr are copula-correlated innovations (t-copula or Gaussian).

    Returns
    -------
    np.ndarray, shape (N, T)
    """
    # Unpack (mirror layout of ModelB.state_to_vector)
    p  = 0
    mu_ik    = params[p:p+2*N].reshape(N, 2);  p += 2*N
    sigma_ik = params[p:p+2*N].reshape(N, 2);  p += 2*N
    pi_i     = params[p:p+N];                  p += N

    # Correlated innovations via copula
    z_corr = _copula_innovations(T, N, L, rng, copula_df)   # (T, N)

    # Regime selection per stock per day
    u_regime  = rng.random((T, N))         # independent uniform for regime draw
    regime    = (u_regime < pi_i[None, :]).astype(int)   # 0=calm, 1=stress (T, N)

    mu_day    = np.where(regime == 0, mu_ik[:, 0][None, :], mu_ik[:, 1][None, :])    # (T, N)
    sigma_day = np.where(regime == 0, sigma_ik[:, 0][None, :], sigma_ik[:, 1][None, :])

    daily_r = mu_day + sigma_day * z_corr   # (T, N)
    return daily_r.T   # (N, T)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def compute_risk_metrics(portfolio_returns: np.ndarray, horizon_days: int) -> dict:
    """Compute standard portfolio risk metrics from simulated returns.

    Parameters
    ----------
    portfolio_returns : np.ndarray, shape (n_sim,)
        Simulated simple portfolio returns over the full horizon.
    horizon_days : int
        Simulation horizon (used for annualisation).

    Returns
    -------
    dict with descriptive keys (all returns as decimals, e.g. 0.05 = 5 %).
    """
    r = portfolio_returns
    n = len(r)

    expected_return = float(r.mean())
    median_return   = float(np.median(r))
    vol_annual      = float(r.std() * np.sqrt(252 / horizon_days))

    pct_5  = float(np.percentile(r, 5))
    pct_25 = float(np.percentile(r, 25))
    pct_75 = float(np.percentile(r, 75))
    pct_95 = float(np.percentile(r, 95))

    var_5     = pct_5
    cvar_tail = r[r <= pct_5]
    cvar_5    = float(cvar_tail.mean()) if len(cvar_tail) > 0 else var_5

    prob_loss = float((r < 0).mean())

    skewness  = float(_skewness(r))
    kurt      = float(_kurtosis(r))

    metrics = {
        "expected_return" : expected_return,
        "median_return"   : median_return,
        "volatility_ann"  : vol_annual,
        "var_5pct"        : var_5,
        "cvar_5pct"       : cvar_5,
        "pct_25"          : pct_25,
        "pct_75"          : pct_75,
        "pct_95"          : pct_95,
        "prob_loss"       : prob_loss,
        "skewness"        : skewness,
        "excess_kurtosis" : kurt,
        "n_sim"           : n,
        "horizon_days"    : horizon_days,
    }

    return metrics


def compute_cvar_contribution(
    portfolio_returns: np.ndarray,
    stock_returns: np.ndarray,
    weights: np.ndarray,
) -> dict:
    """Compute each stock's marginal contribution to CVaR.

    Uses the marginal CVaR formula: each stock's contribution to the portfolio
    CVaR is its average return in the tail, weighted by its portfolio weight.

    Returns
    -------
    dict with "contributions" (N,) and "pct_contributions" (N,)
    """
    tail_mask = portfolio_returns <= np.percentile(portfolio_returns, 5)
    tail_stock = stock_returns[tail_mask]   # (n_tail, N)

    # Contribution = w_i × E[r_i | portfolio in tail]
    avg_tail_stock = tail_stock.mean(axis=0)   # (N,)
    contributions  = weights * avg_tail_stock

    total = contributions.sum()
    pct   = contributions / (total + 1e-12)

    return {
        "contributions"     : contributions,
        "pct_contributions" : pct,
        "avg_tail_stock"    : avg_tail_stock,
    }


# ---------------------------------------------------------------------------
# Fan chart percentiles
# ---------------------------------------------------------------------------

def _fan_chart_percentiles(cum_paths: np.ndarray) -> dict:
    """Compute percentile bands over simulated cumulative log-return paths.

    Parameters
    ----------
    cum_paths : np.ndarray, shape (n_sim, T+1)

    Returns
    -------
    dict: {5: path_5, 25: path_25, 50: path_50, 75: path_75, 95: path_95}
        Each value is np.ndarray of shape (T+1,), in simple-return space.
    """
    pcts = {}
    for p in [5, 25, 50, 75, 95]:
        pcts[p] = np.exp(np.percentile(cum_paths, p, axis=0)) - 1.0
    return pcts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copula_innovations(
    T: int,
    N: int,
    L: np.ndarray,
    rng: np.random.Generator,
    copula_df: float | None,
) -> np.ndarray:
    """Generate (T, N) correlated innovations using a Gaussian or Student-t copula.

    Parameters
    ----------
    T : int
        Number of time steps.
    N : int
        Number of stocks.
    L : np.ndarray, shape (N, N)
        Lower Cholesky factor of the Spearman correlation matrix.
    rng : np.random.Generator
    copula_df : float or None
        Degrees of freedom for the t-copula.  If None or >= 200 the Gaussian
        copula is used (equivalent to df → ∞).

    Returns
    -------
    np.ndarray, shape (T, N) — correlated innovations with unit marginal variance.

    Notes
    -----
    The t-copula shares a single χ² draw across all N stocks on each day,
    creating tail dependence: when the shared scaling variable is small every
    stock receives a large innovation simultaneously.

    Marginal variance of a t(df) is df/(df-2), so we rescale by
    sqrt((df-2)/df) to keep the marginal variance at 1.  This ensures the
    per-stock μ and σ parameters from the MCMC posterior retain their
    calibrated meaning.
    """
    z      = rng.standard_normal((T, N))   # (T, N)
    z_corr = z @ L.T                       # (T, N) — correlated Gaussian

    use_gaussian = copula_df is None or copula_df >= 200
    if use_gaussian:
        return z_corr

    # Shared chi-squared scaling — one draw per day, broadcast across stocks
    # w ~ chi2(df)/df, so 1/sqrt(w) inflates on bad days
    w = rng.chisquare(copula_df, size=(T, 1)) / copula_df   # (T, 1)
    t_corr = z_corr / np.sqrt(w)                             # (T, N)

    # Rescale so marginal variance = 1 (raw t has var = df/(df-2))
    t_corr *= np.sqrt((copula_df - 2.0) / copula_df)

    return t_corr


def _safe_cholesky(C: np.ndarray) -> np.ndarray:
    """Return lower Cholesky factor of C with a small ridge if needed."""
    try:
        return cholesky(C, lower=True)
    except np.linalg.LinAlgError:
        ridge = 1e-6 * np.eye(C.shape[0])
        return cholesky(C + ridge, lower=True)


def _skewness(x: np.ndarray) -> float:
    n   = len(x)
    mu  = x.mean()
    s   = x.std()
    if s < 1e-12:
        return 0.0
    return float(((x - mu) ** 3).mean() / s ** 3)


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (normal = 0)."""
    mu = x.mean()
    s  = x.std()
    if s < 1e-12:
        return 0.0
    return float(((x - mu) ** 4).mean() / s ** 4) - 3.0
