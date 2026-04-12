"""
Tests for the Student-t copula implementation in simulation/montecarlo.py.

Validates:
1. Shape correctness
2. Marginal variance ≈ 1 (variance-neutral rescaling)
3. Tail dependence: t-copula produces more joint extreme events than Gaussian
4. Gaussian fallback: copula_df=None and copula_df=200 behave like Gaussian
5. Portfolio risk: CVaR is more conservative under t-copula than Gaussian
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import spearmanr
from simulation.montecarlo import (
    _copula_innovations,
    _safe_cholesky,
    run_forward_simulation,
    compute_risk_metrics,
)


def make_corr_matrix(N, rho=0.5):
    """Create a simple equicorrelation matrix."""
    C = np.full((N, N), rho)
    np.fill_diagonal(C, 1.0)
    return C


def test_shape():
    """Output shape should be (T, N) for all copula_df settings."""
    N, T = 5, 1000
    L = _safe_cholesky(make_corr_matrix(N))
    rng = np.random.default_rng(42)

    for df in [None, 3, 5, 10, 200]:
        out = _copula_innovations(T, N, L, rng, df)
        assert out.shape == (T, N), f"Shape mismatch for df={df}: {out.shape}"
    print("PASS: test_shape")


def test_marginal_variance():
    """Marginal variance should be ≈ 1 after rescaling, for all df."""
    N, T = 5, 200_000
    L = _safe_cholesky(make_corr_matrix(N, rho=0.3))
    rng = np.random.default_rng(123)

    for df in [3, 5, 10, 50, None]:
        out = _copula_innovations(T, N, L, rng, df)
        var_per_stock = out.var(axis=0)
        label = f"df={df}" if df else "Gaussian"
        mean_var = var_per_stock.mean()
        print(f"  {label:>12s}: mean marginal var = {mean_var:.4f}  (range {var_per_stock.min():.4f} – {var_per_stock.max():.4f})")
        assert abs(mean_var - 1.0) < 0.05, f"Marginal variance too far from 1 for {label}: {mean_var}"
    print("PASS: test_marginal_variance")


def test_tail_dependence():
    """t-copula should produce more joint extremes than Gaussian.

    We measure: fraction of days where ALL stocks are below their 5th percentile.
    Under independence this is 0.05^N ≈ 0.  Under a Gaussian copula it's small.
    Under a t-copula with low df it should be meaningfully larger.
    """
    N, T = 5, 500_000
    rho = 0.5
    L = _safe_cholesky(make_corr_matrix(N, rho))

    results = {}
    for label, df in [("Gaussian", None), ("t(df=5)", 5), ("t(df=3)", 3)]:
        rng = np.random.default_rng(99)
        out = _copula_innovations(T, N, L, rng, df)

        # Threshold: 5th percentile per stock
        thresholds = np.percentile(out, 5, axis=0)  # (N,)
        below = out < thresholds[None, :]            # (T, N)
        all_below = below.all(axis=1)                # (T,) all stocks simultaneously extreme
        frac = all_below.mean()
        results[label] = frac
        print(f"  {label:>12s}: P(all {N} stocks below 5th pctile) = {frac:.6f}")

    # t(df=3) should have strictly more joint extremes than Gaussian
    assert results["t(df=3)"] > results["Gaussian"] * 1.5, (
        f"t(df=3) should have >1.5x the joint tail probability of Gaussian, "
        f"got {results['t(df=3)']:.6f} vs {results['Gaussian']:.6f}"
    )
    assert results["t(df=5)"] > results["Gaussian"] * 1.2, (
        f"t(df=5) should have >1.2x the joint tail probability of Gaussian"
    )
    print("PASS: test_tail_dependence")


def test_gaussian_fallback():
    """copula_df=None and copula_df=200 should produce similar stats to raw Gaussian."""
    N, T = 5, 200_000
    L = _safe_cholesky(make_corr_matrix(N, rho=0.4))

    rng1 = np.random.default_rng(42)
    out_none = _copula_innovations(T, N, L, rng1, copula_df=None)

    rng2 = np.random.default_rng(42)
    out_200 = _copula_innovations(T, N, L, rng2, copula_df=200)

    # With same seed and both using Gaussian path, should be identical
    assert np.allclose(out_none, out_200), "df=None and df=200 should both use Gaussian path"
    print("PASS: test_gaussian_fallback")


def test_correlation_preserved():
    """Pairwise correlations in copula output should approximately match input."""
    N = 5
    rho = 0.6
    T = 200_000
    C = make_corr_matrix(N, rho)
    L = _safe_cholesky(C)

    for label, df in [("Gaussian", None), ("t(df=5)", 5)]:
        rng = np.random.default_rng(77)
        out = _copula_innovations(T, N, L, rng, df)
        empirical_corr = np.corrcoef(out.T)
        off_diag = empirical_corr[np.triu_indices(N, k=1)]
        mean_corr = off_diag.mean()
        print(f"  {label:>12s}: mean pairwise corr = {mean_corr:.4f}  (target ≈ {rho:.2f})")
        # t-copula correlation differs slightly from input (by design) but should be close
        assert abs(mean_corr - rho) < 0.05, f"Correlation too far from target for {label}"
    print("PASS: test_correlation_preserved")


def test_portfolio_cvar_comparison():
    """Run a mini forward simulation and verify t-copula produces fatter tails.

    Key practical test: with tail dependence, portfolio return distribution
    should have higher kurtosis and more extreme tail events.

    Note: at moderate quantiles (5%), the t-copula's unit-variance rescaling
    can actually make VaR/CVaR *less* extreme (the standardized t(df) has its
    5th percentile closer to 0 than the standard normal for low df).  The tail
    dependence effect dominates at more extreme quantiles (1%, 0.5%).  We test
    both: kurtosis for the overall shape, and 1%-CVaR for the deep tail.
    """
    N = 4
    T_hist = 500
    rng = np.random.default_rng(42)

    # Synthetic historical returns
    returns_hist = rng.standard_normal((N, T_hist)) * 0.01

    # Spearman correlation
    corr, _ = spearmanr(returns_hist.T)
    C = np.asarray(corr)

    # Fake posterior samples for Model A: [mu_1..N, sigma_1..N, nu]
    n_posterior = 200
    mu_vals   = np.full(N, 0.0003)
    sig_vals  = np.full(N, 0.012)
    nu_val    = 5.0
    posterior  = np.tile(np.concatenate([mu_vals, sig_vals, [nu_val]]), (n_posterior, 1))
    # Add small noise to posteriors
    posterior += rng.standard_normal(posterior.shape) * 0.0001

    weights = np.ones(N) / N
    tickers = [f"STK{i}" for i in range(N)]

    sim_results = {}
    for label, df in [("Gaussian", 200), ("t(df=5)", 5), ("t(df=3)", 3)]:
        res = run_forward_simulation(
            selected_model="A",
            posterior_samples=posterior,
            returns_hist=returns_hist,
            spearman_corr=C,
            weights=weights,
            tickers=tickers,
            horizon_days=252,
            n_sim=30_000,
            seed=42,
            copula_df=df,
        )
        rm = res["risk_metrics"]
        pr = res["portfolio_returns"]

        # Compute 1% CVaR (deep tail)
        pct_1 = np.percentile(pr, 1)
        cvar_1pct = float(pr[pr <= pct_1].mean()) if (pr <= pct_1).sum() > 0 else pct_1

        sim_results[label] = {**rm, "cvar_1pct": cvar_1pct}
        print(f"  {label:>12s}: E[R]={rm['expected_return']*100:+.2f}%  "
              f"VaR(5%)={rm['var_5pct']*100:+.2f}%  "
              f"CVaR(5%)={rm['cvar_5pct']*100:+.2f}%  "
              f"CVaR(1%)={cvar_1pct*100:+.2f}%  "
              f"ExKurt={rm['excess_kurtosis']:.2f}")

    # Excess kurtosis should be larger for t-copula (fatter tails overall)
    assert sim_results["t(df=3)"]["excess_kurtosis"] > sim_results["Gaussian"]["excess_kurtosis"], (
        "t(df=3) should produce fatter portfolio tails (higher kurtosis)"
    )
    assert sim_results["t(df=5)"]["excess_kurtosis"] > sim_results["Gaussian"]["excess_kurtosis"], (
        "t(df=5) should produce fatter portfolio tails (higher kurtosis)"
    )

    # 1% CVaR should be more extreme under t-copula (deep tail)
    assert sim_results["t(df=3)"]["cvar_1pct"] < sim_results["Gaussian"]["cvar_1pct"], (
        f"t(df=3) 1%-CVaR should be more negative than Gaussian: "
        f"{sim_results['t(df=3)']['cvar_1pct']:.4f} vs {sim_results['Gaussian']['cvar_1pct']:.4f}"
    )

    print("PASS: test_portfolio_cvar_comparison")


if __name__ == "__main__":
    print("=" * 60)
    print("Student-t copula tests")
    print("=" * 60)

    test_shape()
    print()
    test_marginal_variance()
    print()
    test_gaussian_fallback()
    print()
    test_correlation_preserved()
    print()
    test_tail_dependence()
    print()
    test_portfolio_cvar_comparison()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
