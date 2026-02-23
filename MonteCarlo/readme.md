# Bayesian Portfolio Simulation Platform — Implementation Guide

**Version:** 0.1.0-draft  
**Audience:** Engineers and quantitative researchers implementing the MCMC + Monte Carlo simulation engine  
**Last updated:** February 2026

---

## 1. Purpose

This platform takes a user-defined portfolio of stocks (with market-cap weights), fits a hierarchical Bayesian model to historical daily log-returns via MCMC, then runs forward Monte Carlo simulations to produce a full distribution of portfolio outcomes over a configurable horizon.

The system automatically selects between two likelihood specifications using WAIC (Widely Applicable Information Criterion) and reports the result to the user.

---

## 2. Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│  Data Ingest │ ──► │  Model Fit   │ ──► │  Model       │ ──► │  Forward    │
│  & Cleaning  │     │  (MCMC x2)   │     │  Selection   │     │  Simulation │
│              │     │              │     │  (WAIC)      │     │  (MC)       │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
       │                   │                    │                     │
       ▼                   ▼                    ▼                     ▼
  Prices → returns    Posterior samples    Model A vs B         Outcome
  Market cap weights  Trace diagnostics    with diagnostics     distributions
  Correlation matrix  R-hat, ESS                                VaR, CVaR, fan charts
```

### 2.1 Module Breakdown

| Module | Responsibility |
|--------|---------------|
| `data/` | Historical price fetching, log-return computation, missing data handling, train/test split |
| `models/` | Model A (Student-t) and Model B (Normal mixture) specifications, prior definitions |
| `mcmc/` | Gibbs sampler with Metropolis-within-Gibbs steps, adaptation, convergence diagnostics |
| `selection/` | WAIC computation, comparison logic, reporting |
| `simulation/` | Forward MC using posterior draws, portfolio-level aggregation |
| `output/` | Visualization: fan charts, posterior densities, trace plots, return distributions |

---

## 3. Model Specifications

The platform fits two candidate models and selects the better one via WAIC.

### 3.1 Model A — Hierarchical Student-t (simpler)

This is the simpler model. It assumes a single regime but captures fat tails via the Student-t likelihood.

**Likelihood:**

```
r_it | μ_i, σ_i, ν ~ Student-t(ν, μ_i, σ_i)
```

where `r_it` is the daily log-return of stock `i` on day `t`.

**Stock-level priors:**

```
μ_i   ~ Normal(μ_0, τ²)          # partial pooling of means
σ_i   ~ LogNormal(log(ν_s), ξ²)  # partial pooling of volatilities
```

**Hyperpriors:**

```
μ_0   ~ Normal(0, 0.005²)        # weakly informative, centered at zero
τ     ~ HalfCauchy(0.002)        # cross-stock mean dispersion
ν_s   ~ LogNormal(-4, 1)         # population volatility scale (log-space center ≈ 0.018)
ξ     ~ HalfCauchy(1.0)          # vol dispersion across stocks
ν     ~ Gamma(2, 0.1) + 2        # degrees of freedom, shifted to ensure ν > 2
```

**Parameter count:** 2N + 5 (where N = number of stocks)

**Justification:** Student-t is the most established single-distribution model for equity returns (Praetz 1972, Blattberg & Gonedes 1974). The hierarchical structure follows Geweke & Amisano (2011). HalfCauchy hyperpriors per Gelman (2006).

---

### 3.2 Model B — Hierarchical 2-Component Normal Mixture (richer)

This model captures regime-switching behavior (calm vs stress) with asymmetric, fat-tailed marginal distributions.

**Likelihood:**

```
z_it | π_i    ~ Bernoulli(π_i)               # regime indicator (latent)
r_it | z_it=k ~ Normal(μ_ik, σ_ik²)          # conditional on regime
```

The marginal (integrating out z) is a 2-component Normal mixture:

```
p(r_it) = π_i · Normal(μ_i1, σ_i1²) + (1 - π_i) · Normal(μ_i2, σ_i2²)
```

**Stock-level priors (per component k ∈ {1, 2}):**

```
μ_ik  ~ Normal(μ_0k, τ_k²)
σ_ik  ~ LogNormal(log(ν_k), ξ_k²)
π_i   ~ Beta(α_π, β_π)
```

**Hyperpriors (per component k):**

```
μ_0k  ~ Normal(0, 0.005²)
τ_k   ~ HalfCauchy(0.002)
ν_k   ~ LogNormal(-4, 1)         # note: ν_2 will learn a higher value (stress vol)
ξ_k   ~ HalfCauchy(1.0)
```

**Mixing hyperpriors:**

```
α_π   ~ Gamma(2, 0.1)
β_π   ~ Gamma(2, 0.1)
```

**Identifiability constraint:** Enforce `σ_i1 < σ_i2` for all `i`. Component 1 is "calm" (lower vol), component 2 is "stress" (higher vol). This prevents label switching during MCMC.

**Parameter count:** 4N + 10 (where N = number of stocks). The `z_it` are latent but integrated out or Gibbs-sampled, not counted as free parameters.

**Justification:** Geweke & Amisano (2011, J. Applied Econometrics) show hierarchical Normal mixtures outperform GARCH and stochastic volatility models on predictive likelihood for S&P 500 returns. The 2-component mixture produces arbitrary skewness and kurtosis > 3, matching REIT empirical moments (skew ≈ −0.76, kurtosis ≈ 10.5 per Xiong & Idzorek 2011). Massing & Ramos (2021, Physica A) confirm mixtures outperform single Student-t on KS and AD statistics across international equity indices.

---

## 4. MCMC Implementation

### 4.1 Sampler Design

Both models use a **Gibbs sampler with Metropolis-within-Gibbs** steps for non-conjugate conditionals.

**Model A — Sampling scheme:**

| Parameter | Update method | Notes |
|-----------|--------------|-------|
| `μ_i` | Gibbs (conjugate) | Normal-Normal conjugacy with Student-t likelihood via data augmentation (see §4.2) |
| `σ_i` | Metropolis (log-Normal proposal) | Propose on log-scale for positivity; tune acceptance to 25-45% |
| `ν` (df) | Metropolis (log-Normal proposal) | Propose `log(ν - 2)` to respect the `ν > 2` constraint |
| `μ_0` | Gibbs (conjugate) | Normal prior, Normal likelihood of `{μ_i}` |
| `τ` | Metropolis (HalfCauchy prior) | Not conjugate; use log-scale proposal |
| `ν_s, ξ` | Metropolis | Hyperpriors on the vol distribution |

**Data augmentation for Student-t (Model A):** The Student-t can be represented as a scale mixture of Normals:

```
r_it | μ_i, σ_i, λ_it ~ Normal(μ_i, σ_i² / λ_it)
λ_it ~ Gamma(ν/2, ν/2)
```

This makes the conditional on `μ_i` conjugate (weighted Normal likelihood). Augment with `λ_it` and Gibbs-sample them from their Gamma full conditional. This is the Jacquier, Polson & Rossi (1994) trick.

**Model B — Sampling scheme:**

| Parameter | Update method | Notes |
|-----------|--------------|-------|
| `z_it` | Gibbs (categorical) | Full conditional is Bernoulli with weights proportional to component likelihoods × mixing weight |
| `μ_ik` | Gibbs (conjugate) | Conditional on `z`, it's Normal-Normal with only the observations assigned to component k |
| `σ_ik` | Metropolis (log-scale) | Conditional on `z` and `μ_ik`; enforce `σ_i1 < σ_i2` by rejecting swaps |
| `π_i` | Gibbs (conjugate) | Beta-Bernoulli: `π_i | z ~ Beta(α_π + n_i1, β_π + n_i2)` where `n_ik` = count of days assigned to component k |
| `μ_0k, τ_k` | Same as Model A | Operate on the subset `{μ_ik}` for each k |
| `ν_k, ξ_k` | Metropolis | Operate on `{σ_ik}` for each k |
| `α_π, β_π` | Metropolis | Likelihood is the product of `Beta(π_i; α_π, β_π)` over all i |

### 4.2 Practical MCMC Details

**Chain configuration:**

```
num_chains    = 4           # run in parallel for R-hat diagnostics
warmup        = 5000        # burn-in / adaptation phase
num_samples   = 10000       # post-warmup draws per chain
thinning      = 2           # store every 2nd sample to reduce autocorrelation
total_stored  = 4 × 5000 = 20000 posterior draws
```

**Adaptation during warmup:** Use dual-averaging (Nesterov 2009) on the Metropolis step sizes. Target acceptance rate: 0.35 for single parameters, 0.234 for blocks (Roberts & Rosenthal 2001). Freeze step sizes after warmup.

**Convergence diagnostics (compute for ALL parameters):**

| Diagnostic | Criterion | Action if failed |
|------------|-----------|-----------------|
| R-hat (split) | < 1.05 | Extend warmup or reparameterize |
| Effective sample size (ESS) | > 400 per chain | Increase `num_samples` or thin less |
| Divergent transitions | 0 | Reduce step size or reparameterize |
| Trace plot visual | No trend, good mixing | Manual inspection |

**Label switching prevention (Model B only):** The ordering constraint `σ_i1 < σ_i2` is enforced as a hard rejection during the Metropolis step for `σ_ik`. If a proposal would violate the ordering, reject it. This is sufficient for 2-component mixtures (Jasra, Holmes & Stephens 2005).

### 4.3 Initialization

Bad initialization causes long burn-in or chains stuck in local modes.

**Recommended initialization strategy:**

1. Compute sample mean and variance of each stock's returns.
2. Run K-means (K=2) on the *absolute returns* to get a rough calm/stress partition.
3. Initialize component means and variances from the partition statistics.
4. Initialize `π_i` from the cluster proportions.
5. Add small random perturbation (different per chain) to break symmetry.

For Model A, initialize `ν` at 5 (a reasonable starting point per Venables & Ripley).

---

## 5. Model Selection via WAIC

### 5.1 What is WAIC?

WAIC (Watanabe-Akaike / Widely Applicable Information Criterion) is a fully Bayesian model comparison metric. Unlike DIC, it uses the full posterior (not a point estimate) and is valid for singular models like mixtures.

```
WAIC = -2 × (lppd - p_waic)
```

where:

- **lppd** (log pointwise predictive density) measures fit
- **p_waic** (effective number of parameters) penalizes complexity

Lower WAIC = better model.

### 5.2 Computation

Given `S` posterior draws `θ^(s)` for `s = 1, ..., S` and `n` observations:

```python
# For each observation i:
# 1. Compute log-likelihood under each posterior draw
log_lik[s, i] = log p(y_i | θ^(s))

# 2. lppd
lppd = Σ_i log( (1/S) Σ_s exp(log_lik[s, i]) )

# 3. p_waic (using the variance method — more stable than the bias method)
p_waic = Σ_i Var_s(log_lik[s, i])

# 4. WAIC
waic = -2 * (lppd - p_waic)
```

**Important for Model B:** When computing `log p(y_i | θ^(s))`, you must marginalize over the latent `z`:

```
p(r_it | θ^(s)) = π_i^(s) · Normal(r_it; μ_i1^(s), σ_i1^(s)²)
               + (1 - π_i^(s)) · Normal(r_it; μ_i2^(s), σ_i2^(s)²)
```

Do NOT condition on the sampled `z_it`. The marginal likelihood is what matters.

### 5.3 Decision Logic

```python
delta_waic = waic_B - waic_A       # positive means A is better
se_delta   = SE of the difference   # computed via pointwise differences

if delta_waic < -4:
    select Model B (mixture)        # strong evidence for mixture
elif delta_waic > 4:
    select Model A (Student-t)      # strong evidence for Student-t
else:
    select Model A (Student-t)      # prefer simpler model when comparable
    report "Models are comparable; defaulting to simpler specification"
```

The threshold of 4 WAIC points corresponds roughly to a Bayes factor of ~7.5, which is moderate-to-strong evidence (Burnham & Anderson 2002).

**SE of the difference:** Compute pointwise WAIC contributions for both models, take the difference, and compute the standard error:

```python
waic_i_A = -2 * (log(mean(lik_A[:, i])) - var(log_lik_A[:, i]))
waic_i_B = -2 * (log(mean(lik_B[:, i])) - var(log_lik_B[:, i]))
diff_i   = waic_i_B - waic_i_A
se_delta = sqrt(n * var(diff_i))
```

Report the SE alongside the WAIC difference. If `|delta_waic| < 2 * se_delta`, the user should know the comparison is uncertain.

---

## 6. Forward Monte Carlo Simulation

### 6.1 Procedure

After model selection, use the posterior draws from the selected model to simulate forward.

```
For s = 1, ..., S_sim (e.g., 10000 simulations):
    1. Draw θ^(s) from the posterior (sample with replacement from MCMC output)
    2. For each stock i, for each day t = 1, ..., T_horizon:
         Model A: draw λ_it ~ Gamma(ν/2, ν/2), then r_it ~ Normal(μ_i, σ_i² / λ_it)
         Model B: draw z_it ~ Bernoulli(π_i), then r_it ~ Normal(μ_i,z, σ_i,z²)
    3. Compute cumulative log-return per stock: R_i = Σ_t r_it
    4. Convert to simple return: total_return_i = exp(R_i) - 1
    5. Compute portfolio return: R_port = Σ_i w_i × total_return_i
       where w_i = market_cap_i / Σ_j market_cap_j
```

**Critical:** Step 1 draws a *new* parameter vector for each simulation. This propagates parameter uncertainty (not just return randomness) into the forward distribution. This is the key advantage over plug-in Monte Carlo.

### 6.2 Correlation Structure

The models above treat stocks as conditionally independent given their parameters. This is acceptable because the hierarchical structure induces *marginal* correlation: stocks sharing similar posteriors (due to pooling) will tend to co-move.

However, for more accurate portfolio-level simulations, you may want to model the residual correlation explicitly. Two options:

**Option A (recommended for v1): Empirical copula.** After simulating independent returns per stock, re-order them to match the empirical rank-correlation structure from the historical data. This is a Gaussian copula approximation.

```python
# 1. Compute historical rank correlation matrix C (Spearman)
# 2. Cholesky decompose: L = cholesky(C)
# 3. For each simulation s:
#    a. Draw z ~ MVN(0, I) of dimension N
#    b. Correlated draws: u = Φ(L @ z), where Φ is standard normal CDF
#    c. For each stock i, use u_i as the quantile:
#       r_it = F_i^{-1}(u_i), where F_i is the CDF of stock i's posterior predictive
```

**Option B (future): Full multivariate model.** Replace the stock-level priors with a multivariate normal on the vector `(μ_1, ..., μ_N)` with an estimated covariance matrix. This is significantly more complex to implement and sample.

### 6.3 Portfolio Weighting

For a market-cap weighted ETF:

```python
weights[i] = market_cap[i] / sum(market_cap)
```

Weights are fixed at simulation start (no rebalancing assumed for 1-year horizon). If the user wants periodic rebalancing, re-normalize weights at each rebalancing date within the simulation loop.

### 6.4 Output Metrics

Compute from the `S_sim` simulated portfolio returns:

| Metric | Definition |
|--------|-----------|
| Expected return | Mean of simulated portfolio returns |
| Volatility (annualized) | Std dev × √252 |
| VaR (5%) | 5th percentile of simulated returns |
| CVaR / ES (5%) | Mean of returns below the 5th percentile |
| Probability of loss | Fraction of simulations with negative return |
| Median return | 50th percentile |
| 95th percentile | Upside scenario |
| Skewness | Of simulated distribution |
| Kurtosis | Of simulated distribution |

---

## 7. Data Requirements

### 7.1 Inputs

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `ticker` | string | yes | Exchange-prefixed if needed (e.g., `TSX:SRU.UN`) |
| `market_cap` | float | yes | In any consistent currency; used for relative weights only |
| `history_days` | int | no | Default 504 (≈ 2 trading years). Minimum 252 recommended |
| `horizon_days` | int | no | Default 252 (1 year forward) |

### 7.2 Data Cleaning

1. Fetch adjusted close prices (dividend-adjusted) for each ticker.
2. Compute log-returns: `r_it = log(P_it / P_i,t-1)`.
3. Remove days where any stock has a missing return (align calendars).
4. Winsorize at ±15% daily return (flag and report; these are likely data errors, not real moves).
5. Require at least 200 observations per stock after cleaning.

### 7.3 Train/Test Split (Optional)

For backtesting WAIC reliability, withhold the last 63 days (≈ 1 quarter) and compare the selected model's predictive performance on the holdout. This is informational only — the final model is fit on all data.

---

## 8. Computational Considerations

### 8.1 Performance Targets

| Portfolio size (N) | MCMC time (both models) | MC simulation time | Total |
|-------|---------|---------|-------|
| 5 stocks | < 30 sec | < 5 sec | < 35 sec |
| 15 stocks | < 2 min | < 15 sec | < 2.5 min |
| 50 stocks | < 10 min | < 1 min | < 11 min |

Targets assume a modern single-threaded JS runtime. Parallelizing across 4 chains gives ~3x speedup.

### 8.2 Implementation Notes

**Language choice:** The platform UI is React/JS. For v1, implement the MCMC engine in JavaScript (runs in browser / Web Worker). For v2, consider a Python backend (NumPy/Numba) or Rust/WASM for 10-50x speedup.

**Web Worker architecture:** MCMC is CPU-intensive. Run it in a Web Worker to avoid blocking the UI. Post progress updates (iteration count, current R-hat) back to the main thread for a live progress display.

**Numerical stability:**

- Always work in log-space for likelihoods. Compute `log p(r|θ)` not `p(r|θ)`.
- For the mixture log-likelihood, use the log-sum-exp trick:
  ```
  log(π·f₁ + (1-π)·f₂) = log-sum-exp(log(π) + log(f₁), log(1-π) + log(f₂))
  ```
- For Gamma/Beta function evaluations, use `lgamma()` not `log(gamma())`.
- Represent HalfCauchy log-density directly: `log(2/π) - log(s) - log(1 + (x/s)²)`.

**Storage:** Store posterior samples in a typed Float64Array with shape `[num_chains × num_samples, num_params]`. For 20,000 draws × 50 parameters, this is ~8 MB — fine for browser memory.

---

## 9. Outputs & Visualization

### 9.1 Diagnostics Tab

- **Trace plots** for all hyperparameters (μ₀, τ, ν_s, ξ, and for Model B: α_π, β_π, μ₀ₖ, τₖ, ν_k, ξ_k). Show all 4 chains overlaid.
- **R-hat table** with green/red indicators (threshold: 1.05).
- **ESS table** with minimum highlighted.
- **WAIC comparison card**: delta WAIC, SE, selected model, plain-English interpretation.

### 9.2 Posterior Tab

- **Marginal posterior densities** for each stock's μ_i, σ_i (and π_i for Model B). Overlay the prior as a faded line.
- **Shrinkage plot**: sample mean vs posterior mean for each stock, showing the pull toward the group mean.
- For Model B: **Regime posterior**: histogram of inferred stress-day fractions per stock.

### 9.3 Simulation Tab

- **Fan chart**: time series showing 5th / 25th / 50th / 75th / 95th percentile bands of cumulative portfolio return over the horizon.
- **Return distribution histogram** at the terminal date, with VaR and CVaR marked.
- **Probability of loss** as a large, prominent number.
- **Stock-level decomposition**: which stocks contribute most to tail risk (by CVaR contribution).

---

## 10. Testing Strategy

### 10.1 Unit Tests

| Test | What it verifies |
|------|-----------------|
| Known-posterior recovery | Fit Model A to synthetic data drawn from a Student-t with known params. Check that the posterior 95% CI covers the true value for >90% of parameters. |
| Conjugacy check | For conjugate updates (μ_i given σ_i, z), compare Gibbs samples to the analytical posterior. KS test p > 0.05. |
| Label switching | Run Model B for 50k iterations. Verify that σ_i1 < σ_i2 holds for 100% of post-warmup samples. |
| WAIC correctness | Generate data from Model A (Student-t). Fit both models. Verify WAIC selects Model A. Repeat with data from Model B. Verify WAIC selects Model B. |
| Log-sum-exp | Compare naive vs log-sum-exp mixture likelihood on extreme inputs. Verify no NaN/Inf. |

### 10.2 Integration Tests

| Test | What it verifies |
|------|-----------------|
| End-to-end on synthetic | Generate a 10-stock portfolio from known parameters. Run full pipeline. Check that 90% CIs from simulation cover the true forward return distribution (calibration test). |
| Real data smoke test | Run on 5 Canadian REITs with 2 years of real data. Verify: no crashes, R-hat < 1.05, ESS > 400, WAIC is finite, simulation produces non-degenerate fan chart. |
| Determinism | Set RNG seed. Run twice. Verify identical output. |

### 10.3 Calibration Test (Critical)

The ultimate test of a Bayesian simulation platform is calibration: do the predicted intervals match realized frequencies?

```
For each of 100 rolling windows:
    1. Fit model on days [1, ..., T]
    2. Simulate 1-month forward distribution
    3. Observe actual 1-month return
    4. Record the percentile rank of the actual return in the simulated distribution
```

If the model is well-calibrated, the percentile ranks should be Uniform(0, 1). Test with a KS test. A p-value < 0.05 indicates miscalibration.

---

## 11. References

1. Geweke, J. & Amisano, G. (2011). Hierarchical Markov Normal Mixture Models with Applications to Financial Asset Returns. *J. Applied Econometrics* 26(1): 1-29.
2. Massing, T. & Ramos, A. (2021). Student's t mixture models for stock indices: A comparative study. *Physica A* 580: 126143.
3. Jacquier, E., Polson, N.G. & Rossi, P.E. (1994, 2002). Bayesian Analysis of Stochastic Volatility Models. *J. Business & Economic Statistics*.
4. Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models. *Bayesian Analysis* 1(3): 515-534.
5. Xiong, J.X. & Idzorek, T.M. (2011). The Impact of Skewness and Fat Tails on the Asset Allocation Decision. *J. Portfolio Management*.
6. Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and WAIC. *J. Machine Learning Research* 11: 3571-3594.
7. Vehtari, A., Gelman, A. & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing* 27: 1413-1432.
8. Roberts, G.O. & Rosenthal, J.S. (2001). Optimal scaling for various Metropolis-Hastings algorithms. *Statistical Science* 16(4): 351-367.
9. Burnham, K.P. & Anderson, D.R. (2002). *Model Selection and Multimodel Inference*. Springer.
10. Hassan, M.K. et al. (2020). Volatility jumps and their determinants in REIT returns. *J. Banking & Finance*.

---

## 12. Appendix A — Full Conditional Derivations

### A.1 Model A: Full conditional for μ_i (with data augmentation)

Given the augmented model `r_it ~ Normal(μ_i, σ_i² / λ_it)` and prior `μ_i ~ Normal(μ_0, τ²)`:

```
μ_i | rest ~ Normal(m_i, V_i)

where:
  precision_lik  = Σ_t λ_it / σ_i²
  precision_prior = 1 / τ²
  V_i = 1 / (precision_lik + precision_prior)
  m_i = V_i × (precision_prior × μ_0 + (1/σ_i²) × Σ_t λ_it × r_it)
```

### A.2 Model A: Full conditional for λ_it

```
λ_it | rest ~ Gamma((ν + 1) / 2,  (ν + (r_it - μ_i)² / σ_i²) / 2)
```

This is the standard scale-mixture augmentation for Student-t.

### A.3 Model B: Full conditional for z_it

```
P(z_it = 1 | rest) ∝ π_i × Normal(r_it; μ_i1, σ_i1²)
P(z_it = 2 | rest) ∝ (1 - π_i) × Normal(r_it; μ_i2, σ_i2²)
```

Normalize to get a Bernoulli probability. Use log-space to avoid underflow.

### A.4 Model B: Full conditional for π_i

```
π_i | rest ~ Beta(α_π + n_i1, β_π + n_i2)

where n_ik = number of days t with z_it = k
```

### A.5 Model B: Full conditional for μ_ik

Identical to A.1 but using only the observations `{r_it : z_it = k}` and without the λ augmentation (since the likelihood is Normal, not Student-t):

```
μ_ik | rest ~ Normal(m_ik, V_ik)

where:
  T_k = |{t : z_it = k}|
  precision_lik = T_k / σ_ik²
  precision_prior = 1 / τ_k²
  V_ik = 1 / (precision_lik + precision_prior)
  m_ik = V_ik × (precision_prior × μ_0k + (1/σ_ik²) × Σ_{t: z=k} r_it)
```

### A.6 WAIC pointwise log-likelihood

**Model A:**
```
log p(r_it | θ^(s)) = log Student-t(r_it; ν^(s), μ_i^(s), σ_i^(s))
```

Use the standard Student-t log-PDF:
```
log Γ((ν+1)/2) - log Γ(ν/2) - 0.5×log(νπσ²) - ((ν+1)/2) × log(1 + (r-μ)²/(νσ²))
```

**Model B:**
```
log p(r_it | θ^(s)) = log[ π_i^(s) × φ(r_it; μ_i1^(s), σ_i1^(s))
                          + (1 - π_i^(s)) × φ(r_it; μ_i2^(s), σ_i2^(s)) ]
```

Use log-sum-exp for numerical stability.