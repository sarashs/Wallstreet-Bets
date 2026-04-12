# Bayesian Portfolio Simulation Platform — Implementation Documentation

**Version:** 0.1.0
**Language:** Python 3.10+
**Entry point:** `streamlit run MonteCarlo/app.py`

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Installation](#3-installation)
4. [Module Reference](#4-module-reference)
   - 4.1 [data/fetcher.py](#41-datafetcherpy)
   - 4.2 [models/model_a.py](#42-modelsmodel_apy)
   - 4.3 [models/model_b.py](#43-modelsmodel_bpy)
   - 4.4 [mcmc/sampler.py](#44-mcmcsamplerpy)
   - 4.5 [mcmc/diagnostics.py](#45-mcmcdiagnosticspy)
   - 4.6 [selection/waic.py](#46-selectionwaicpy)
   - 4.7 [simulation/montecarlo.py](#47-simulationmontecarolopy)
   - 4.8 [output/visualizations.py](#48-outputvisualizationspy)
   - 4.9 [app.py](#49-apppy)
5. [Mathematical Details](#5-mathematical-details)
   - 5.1 [Model A — Hierarchical Student-t](#51-model-a)
   - 5.2 [Model B — Hierarchical Normal Mixture](#52-model-b)
   - 5.3 [Gibbs Sampler Update Equations](#53-gibbs-sampler-update-equations)
   - 5.4 [WAIC Computation](#54-waic-computation)
   - 5.5 [Forward Simulation with Gaussian Copula](#55-forward-simulation-with-gaussian-copula)
6. [Logging & Debugging](#6-logging--debugging)
7. [Saved File Formats](#7-saved-file-formats)
8. [Performance Notes](#8-performance-notes)
9. [Known Limitations & Future Work](#9-known-limitations--future-work)

---

## 1. System Overview

The platform fits two competing hierarchical Bayesian models to historical daily
log-returns of a user-defined portfolio, selects the better model via WAIC, then
runs a forward Monte Carlo simulation to produce a full distribution of portfolio
outcomes over a configurable horizon.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Yahoo Finance  │    │  MCMC (2 models) │    │  WAIC Model     │    │  Forward MC  │
│  Price Fetch    │ ─► │  Model A: Stud-t │ ─► │  Selection      │ ─► │  Simulation  │
│  Log-returns    │    │  Model B: NormMix│    │  Lower = Better │    │  Risk metrics│
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────┘
         │                      │                       │                      │
         ▼                      ▼                       ▼                      ▼
  Cleaned returns         Posterior draws           Selected model        Fan chart
  Spearman corr.         Trace / R-hat / ESS       WAIC report          VaR / CVaR
```

---

## 2. Directory Structure

```
MonteCarlo/
├── app.py                    # Streamlit dashboard (entry point)
├── IMPLEMENTATION.md         # This file
├── requirements.txt          # Python dependencies
├── readme.md                 # Design specification
│
├── data/
│   ├── __init__.py
│   └── fetcher.py            # Yahoo Finance ingestion, cleaning, Spearman corr.
│
├── models/
│   ├── __init__.py
│   ├── model_a.py            # Hierarchical Student-t (Model A)
│   └── model_b.py            # Hierarchical Normal Mixture (Model B)
│
├── mcmc/
│   ├── __init__.py
│   ├── sampler.py            # GibbsSampler: 4-chain MCMC with adaptation
│   └── diagnostics.py        # R-hat, ESS, diagnostics table
│
├── selection/
│   ├── __init__.py
│   └── waic.py               # WAIC computation and model comparison
│
├── simulation/
│   ├── __init__.py
│   └── montecarlo.py         # Forward MC, Gaussian copula, risk metrics
│
├── output/
│   ├── __init__.py
│   ├── visualizations.py     # Plotly figures (fan chart, trace, etc.)
│   └── <timestamp>/          # Auto-created on save
│       ├── config.yaml       # Run metadata and diagnostics
│       └── posterior.npz     # Compressed posterior samples
│
└── logs/
    └── app.log               # Rolling debug log
```

---

## 3. Installation

```bash
cd /path/to/Wallstreet-Bets
pip install -r MonteCarlo/requirements.txt
streamlit run MonteCarlo/app.py
```

**Dependencies:**

| Package      | Purpose                                    |
|--------------|--------------------------------------------|
| yfinance     | Adjusted close price download              |
| numpy        | All numerical operations                   |
| scipy        | Statistical distributions, Cholesky, KDE   |
| pandas       | DataFrame handling for price data          |
| streamlit    | Interactive dashboard                      |
| plotly       | Interactive charts                         |
| pyyaml       | Config serialisation                       |
| scikit-learn | K-means initialisation for Model B         |

---

## 4. Module Reference

### 4.1 `data/fetcher.py`

**`fetch_returns(tickers, n_days=504) → dict`**

Downloads adjusted close prices via `yfinance`, computes daily log-returns
`r_it = log(P_it / P_i,t-1)`, aligns calendars (drops any day where any ticker
has a missing return), winsorises values beyond ±15% (likely data errors), and
drops tickers with fewer than 200 clean observations.

| Return key        | Type                | Description                          |
|-------------------|---------------------|--------------------------------------|
| `returns`         | `ndarray (N, T)`    | Cleaned log-returns                  |
| `tickers`         | `list[str]`         | Tickers that survived cleaning       |
| `dates`           | `list[str]`         | ISO date strings for each column     |
| `prices`          | `ndarray (N, T)`    | Adjusted close prices                |
| `n_winsorised`    | `int`               | Total observations winsorised        |
| `dropped_tickers` | `list[str]`         | Tickers removed (insufficient data)  |

**`compute_spearman_correlation(returns_matrix) → ndarray (N, N)`**

Computes the Spearman rank-correlation matrix and ensures positive-definiteness
by adding a small ridge if needed (eigenvalue correction).

---

### 4.2 `models/model_a.py`

Implements the **Hierarchical Student-t** model (Model A).

**Class `ModelA`**

| Method | Description |
|--------|-------------|
| `initialize(returns, rng, chain_idx)` | Create starting state via sample stats + perturbation |
| `update_lambda(state, returns, rng)` | Gibbs: λ_it ~ Gamma |
| `update_mu_i(state, returns, rng)` | Gibbs: μ_i ~ Normal (conjugate) |
| `update_sigma_i(state, returns, step_size, rng)` | Metropolis: σ_i (log-scale) |
| `update_nu(state, step_size, rng)` | Metropolis: ν (log(ν-2) scale) |
| `update_mu_0(state, rng)` | Gibbs: μ_0 ~ Normal (conjugate) |
| `update_tau(state, step_size, rng)` | Metropolis: τ (HalfCauchy prior) |
| `update_nu_s(state, step_size, rng)` | Metropolis: ν_s (population vol scale) |
| `update_xi(state, step_size, rng)` | Metropolis: ξ (vol dispersion) |
| `log_lik_pointwise(state, returns)` | Returns `(N, T)` Student-t log-likelihoods |
| `state_to_vector(state)` | Flatten non-latent params to 1-D array |
| `param_names(N)` | Return parameter names in vector order |

**State dict keys:** `mu_i, sigma_i, nu, mu_0, tau, nu_s, xi, lambda_it`

---

### 4.3 `models/model_b.py`

Implements the **Hierarchical 2-component Normal Mixture** model (Model B).

**Class `ModelB`**

| Method | Description |
|--------|-------------|
| `initialize(returns, rng, chain_idx)` | K-means on |returns| for calm/stress init |
| `update_z(state, returns, rng)` | Gibbs: z_it ~ Bernoulli (regime assignment) |
| `update_mu_ik(state, returns, rng)` | Gibbs: μ_ik ~ Normal (conjugate per component) |
| `update_sigma_ik(state, returns, step_sizes, rng)` | Metropolis: σ_ik with σ_i0 < σ_i1 constraint |
| `update_pi(state, rng)` | Gibbs: π_i ~ Beta (conjugate) |
| `update_mu_0k(state, rng)` | Gibbs: μ_0k ~ Normal |
| `update_tau_k(state, step_sizes, rng)` | Metropolis: τ_k (per component) |
| `update_nu_k(state, step_sizes, rng)` | Metropolis: ν_k (per component) |
| `update_xi_k(state, step_sizes, rng)` | Metropolis: ξ_k (per component) |
| `update_alpha_beta_pi(state, step_sizes, rng)` | Metropolis: α_π, β_π |
| `log_lik_pointwise(state, returns)` | `(N, T)` mixture log-likelihoods (log-sum-exp) |

**State dict keys:** `mu_ik, sigma_ik, pi_i, z_it, mu_0k, tau_k, nu_k, xi_k, alpha_pi, beta_pi`

**Identifiability:** Component 0 = calm (lower σ), component 1 = stress (higher σ).
σ ordering constraint is enforced by rejecting proposals that violate σ_i0 < σ_i1.

---

### 4.4 `mcmc/sampler.py`

**Class `GibbsSampler`**

```python
GibbsSampler(model, returns, n_chains=4, warmup=2000, n_samples=5000,
             thinning=2, seed=42, progress_callback=None)
```

**`.run() → dict`**

Runs all chains sequentially and returns:

| Key | Description |
|-----|-------------|
| `samples` | `(n_chains × n_stored, D)` — all post-warmup draws |
| `param_names` | List of parameter names |
| `acceptance` | Dict of mean acceptance rates per parameter group |
| `chain_samples` | List of per-chain arrays (for R-hat) |

**Adaptation scheme:** Every 100 warmup iterations, step sizes are scaled by
factors of 0.8, 0.9, 1.0, or 1.1 to nudge acceptance rates toward the target
(35% for scalar parameters). Step sizes are frozen after warmup.

**Metropolis log-acceptance ratio (with Jacobian):**

```
log α = [log p(θ'|data) − log p(θ|data)] + [log θ' − log θ]
```

The last term is the Jacobian correction for proposing on the log-scale (makes
the sampler target the correct posterior density in original parameter space).

---

### 4.5 `mcmc/diagnostics.py`

**`compute_rhat(chain_samples) → ndarray (D,)`**

Split-chain R-hat (Vehtari et al. 2021): each chain is split in half, giving
2 × n_chains sub-chains. Between-chain variance B and within-chain variance W
are combined into the potential-scale-reduction factor:

```
R-hat = sqrt(var_hat / W)     where var_hat = (N-1)/N × W + B/N
```

Values < 1.05 indicate convergence.

**`compute_ess(chain_samples) → ndarray (D,)`**

Bulk ESS via rank-normalisation and autocorrelation sum:

```
ESS ≈ S / (1 + 2 × Σ_k ρ_k)
```

where the sum stops at the first non-positive consecutive pair (the "initial
monotone sequence" criterion of Geyer 1992). Values > 400 per chain are good.

**`make_diagnostics_table(chain_samples, param_names) → dict`**

Returns posterior mean, std, R-hat, ESS, and pass/fail indicators for all
parameters.

---

### 4.6 `selection/waic.py`

**`compute_waic(log_lik_matrix) → dict`**

Given `log_lik_matrix[s, i] = log p(y_i | θ^(s))`:

```python
lppd_i   = log( mean_s( exp(log_lik[s, i]) ) )   # log-sum-exp for stability
p_waic_i = Var_s( log_lik[s, i] )
waic_i   = -2 × (lppd_i − p_waic_i)
waic     = sum(waic_i)
se       = sqrt(n × var(waic_i))
```

**`compare_models(waic_a, waic_b) → dict`**

```
delta = waic_B − waic_A
if delta < −4:  select Model B (strong evidence for regime switching)
if delta >  4:  select Model A (fat-tails sufficient)
else:           select Model A (prefer simpler model)
```

**`compute_log_lik_matrix(model, posterior_samples, returns, n_sub=500)`**

Reconstructs parameter state dicts from flattened sample vectors and calls
`model.log_lik_pointwise()` for each draw. Subsamples up to `n_sub` draws
for efficiency.

---

### 4.7 `simulation/montecarlo.py`

**`run_forward_simulation(...) → dict`**

Main simulation loop:

```
For s = 1, ..., n_sim:
    1. Draw a posterior parameter vector (random index into posterior_samples)
    2. Generate T_horizon daily returns using Student-t copula (or Gaussian):
         z      ~ MVN(0, I_N)          (standard normals per stock)
         z_corr = L @ z               (apply Cholesky of Spearman corr.)
         w      ~ χ²(copula_df) / copula_df   (shared across all stocks per day)
         t_corr = z_corr / sqrt(w) × sqrt((df-2)/df)   (t-copula innovations)
         Model A: r_it = μ_i + σ_i × t_corr_it / sqrt(λ_it)
                  λ_it ~ Gamma(ν/2, ν/2)
         Model B: k_it ~ Bernoulli(π_i)
                  r_it = μ_i,k + σ_i,k × t_corr_it
    3. Cumulative log-return per stock: R_i = sum_t r_it
    4. Simple return: exp(R_i) − 1
    5. Portfolio return: Σ_i w_i × (exp(R_i) − 1)
```

The `copula_df` parameter (default 5) controls tail dependence.  The shared χ²
draw means all stocks receive inflated innovations on the same "bad days",
producing realistic crash clustering.  Setting copula_df ≥ 200 recovers the
Gaussian copula.  The `sqrt((df-2)/df)` rescaling preserves unit marginal
variance so that posterior μ and σ parameters keep their calibrated meaning.

Returns per-path cumulative curves for the fan chart plus terminal returns
for all risk metrics.

**`compute_risk_metrics(portfolio_returns, horizon_days) → dict`**

| Metric | Calculation |
|--------|-------------|
| Expected return | `mean(r)` |
| Median | `percentile(r, 50)` |
| Volatility (ann.) | `std(r) × sqrt(252 / horizon_days)` |
| VaR (5%) | `percentile(r, 5)` |
| CVaR (5%) | `mean(r[r ≤ VaR])` |
| Prob. of loss | `mean(r < 0)` |
| Skewness | `E[(r−μ)³] / σ³` |
| Excess kurtosis | `E[(r−μ)⁴] / σ⁴ − 3` |

**`compute_cvar_contribution(portfolio_returns, stock_returns, weights)`**

For each stock:
```
contribution_i = w_i × E[r_i | portfolio in tail]
```
where "tail" = simulations where the portfolio return ≤ VaR(5%).

---

### 4.8 `output/visualizations.py`

All functions return a `plotly.graph_objects.Figure`.

| Function | Description |
|----------|-------------|
| `plot_fan_chart` | 5/25/50/75/95 percentile bands over time with VaR annotation |
| `plot_return_distribution` | Histogram of terminal returns with VaR, CVaR, E[R] markers |
| `plot_trace` | Overlaid trace plots for N chains across the first K parameters |
| `plot_posterior_densities` | KDE posterior marginals with mean marker |
| `plot_rhat_table` | Colour-coded table (green < 1.05, red ≥ 1.05) |
| `plot_ess_table` | Colour-coded table (green > 400, red ≤ 400) |
| `plot_shrinkage` | Scatter: sample mean vs posterior mean for μ_i (shows pooling) |
| `plot_var_breakdown` | Horizontal bar chart of per-stock CVaR contributions |

---

### 4.9 `app.py`

**Tab 1 — MCMC & Portfolio Setup**

1. Add tickers individually (ticker + weight input fields) or in bulk (CSV paste)
2. Inline data editor to modify / remove tickers
3. MCMC configuration: history days, warmup, sampling iterations, chains, thinning
4. "Run MCMC" button → calls `run_mcmc_pipeline()` which:
   - Fetches and cleans data
   - Runs Model A MCMC
   - Runs Model B MCMC
   - Computes WAIC for both models
   - Selects the winning model
5. Shows WAIC summary card, convergence diagnostics tabs (R-hat, ESS, trace, posteriors)
6. "Save to YAML + NPZ" button → writes `output/<timestamp>/config.yaml` and `posterior.npz`

**Tab 2 — Monte Carlo Simulation**

1. Load source: session result (from Tab 1) or saved YAML file
2. Configure: horizon days, number of paths, random seed
3. "Run Monte Carlo Simulation" → calls `run_forward_simulation()`
4. Displays:
   - Six key metrics (E[R], median, vol, VaR, CVaR, P(loss))
   - Fan chart (5/25/50/75/95 percentile paths)
   - Return distribution histogram
   - CVaR breakdown by stock
   - Per-stock summary table
5. "Save simulation figures" → writes PNG images + metrics YAML

**Sidebar:** Live activity log (last 50 messages), clearable.

---

## 5. Mathematical Details

### 5.1 Model A

**Likelihood (via data augmentation):**
```
λ_it | ν    ~ Gamma(ν/2, ν/2)
r_it | μ_i, σ_i, λ_it  ~ Normal(μ_i, σ_i² / λ_it)
```
Integrating out λ gives the Student-t marginal:
```
r_it | μ_i, σ_i, ν  ~ Student-t(ν, μ_i, σ_i)
```

**Priors:**
```
μ_i  ~ Normal(μ_0, τ²)
σ_i  ~ LogNormal(log(ν_s), ξ²)
μ_0  ~ Normal(0, 0.005²)
τ    ~ HalfCauchy(0.002)
ν_s  ~ LogNormal(-4, 1)
ξ    ~ HalfCauchy(1.0)
ν    ~ Gamma(2, 0.1) + 2
```

**Parameter count:** 2N + 5

### 5.2 Model B

**Likelihood:**
```
z_it | π_i   ~ Bernoulli(π_i)            # regime: 0=calm, 1=stress
r_it | z=k   ~ Normal(μ_ik, σ_ik²)
```

**Marginal (log-sum-exp for stability):**
```
log p(r_it) = log-sum-exp(
    log(1−π_i) + log φ(r_it; μ_i0, σ_i0),
    log(π_i)   + log φ(r_it; μ_i1, σ_i1)
)
```

**Priors per component k ∈ {0,1}:**
```
μ_ik ~ Normal(μ_0k, τ_k²)
σ_ik ~ LogNormal(log(ν_k), ξ_k²)
π_i  ~ Beta(α_π, β_π)
```

**Identifiability:** `σ_i0 < σ_i1` enforced by hard rejection in Metropolis.

**Parameter count:** 4N + 10

### 5.3 Gibbs Sampler Update Equations

**Model A — μ_i (conjugate Gibbs with augmentation):**
```
precision_lik  = Σ_t λ_it / σ_i²
precision_prior = 1 / τ²
V_i = 1 / (precision_lik + precision_prior)
m_i = V_i × (precision_prior × μ_0 + (1/σ_i²) × Σ_t λ_it × r_it)
μ_i | rest ~ Normal(m_i, V_i)
```

**Model A — λ_it (conjugate Gibbs):**
```
λ_it | rest ~ Gamma((ν+1)/2,  (ν + (r_it − μ_i)² / σ_i²) / 2)
```

**Model B — z_it (categorical Gibbs):**
```
log P(z_it=0 | rest) ∝ log(1−π_i) + log φ(r_it; μ_i0, σ_i0)
log P(z_it=1 | rest) ∝ log(π_i)   + log φ(r_it; μ_i1, σ_i1)
P(z_it=1) = softmax([log_p0, log_p1])[1]
```

**Model B — π_i (conjugate Beta Gibbs):**
```
n_i1 = |{t : z_it = 1}|,  n_i0 = |{t : z_it = 0}|
π_i | rest ~ Beta(α_π + n_i1, β_π + n_i0)
```

**Metropolis log-acceptance ratio (log-scale proposal):**
```
log θ_prop = log θ_curr + step × ε,   ε ~ Normal(0, 1)
log α = [log p(θ_prop|data) − log p(θ_curr|data)] + [log θ_prop − log θ_curr]
```
The last term is the Jacobian for the log-scale change of variables.

### 5.4 WAIC Computation

```python
# S posterior draws, n_obs = N × T observations
log_lik[s, i] = log p(y_i | θ^(s))

# Pointwise LPPD (numerically stable)
lppd_i = log( (1/S) Σ_s exp(log_lik[s, i]) )

# Effective parameters (variance method — more stable than bias method)
p_waic_i = Var_s(log_lik[s, i])

# WAIC
waic_i = -2 × (lppd_i − p_waic_i)
WAIC   = Σ_i waic_i

# Standard error
SE = sqrt(n_obs × Var_i(waic_i))
```

### 5.5 Forward Simulation with Student-t Copula

**Correlation structure:** Spearman rank-correlation matrix C of historical returns.
Cholesky factorisation: `L = cholesky(C)`.

**Per-simulation, per-day (Model A):**
```python
z      ~ Normal(0, I_N)          # independent standard normals
z_corr = L @ z                   # correlated via Cholesky
w      ~ χ²(df_c) / df_c        # shared copula scaling (1 draw per day)
t_corr = z_corr / sqrt(w) * sqrt((df_c - 2) / df_c)  # t-copula innovations
λ      ~ Gamma(ν/2, 2/ν, N)     # per-stock scale-mixture weights (Model A only)
r_t    = μ + σ × t_corr / sqrt(λ)  # correlated Student-t returns
```

The shared `w` draw creates tail dependence: when `w` is small, every stock
receives an inflated innovation on the same day.  The `sqrt((df_c-2)/df_c)`
factor normalises the marginal variance to 1.

**Cumulative portfolio return:**
```python
R_log = Σ_t r_t_i               # cumulative log-return per stock
R_simple = exp(R_log) - 1        # simple return
R_port = Σ_i w_i × R_simple_i   # portfolio return
```

---

## 6. Logging & Debugging

**Log file:** `MonteCarlo/logs/app.log`

The app uses Python's `logging` module at `DEBUG` level, writing to both:
- The log file (persisted across restarts, `mode="a"`)
- stdout (visible in the terminal where Streamlit runs)

**In-app log panel:** The sidebar shows the last 50 log messages in real time.

**Log levels used:**

| Level | When |
|-------|------|
| DEBUG | Detailed per-iteration or shape information |
| INFO  | Major pipeline steps (model start, completion, metrics) |
| WARNING | Winsorised observations, dropped tickers, low ESS |
| ERROR | Exceptions caught during pipeline execution |

**To increase verbosity:** Change `logging.basicConfig(level=...)` in `app.py`.

**To view the full log:**
```bash
tail -f MonteCarlo/logs/app.log
```

---

## 7. Saved File Formats

### `config.yaml`

Human-readable metadata:

```yaml
version: '0.1.0'
created_at: '20260222_143000'
tickers: [AAPL, MSFT, GOOGL]
weights: [0.45, 0.35, 0.20]
history_days: 504
selected_model: A
waic_a: 18432.1
waic_b: 18440.3
delta_waic: 8.2
se_delta: 3.1
uncertain: false
rationale: "Model A (Student-t) is strongly preferred..."
rhat_max_a: 1.021
rhat_max_b: 1.034
ess_min_a: 612.0
ess_min_b: 489.0
n_posterior_samples: 10000
mcmc_settings:
  warmup: 2000
  n_samples: 5000
  n_chains: 4
  thinning: 2
posterior_file: posterior.npz
output_dir: /path/to/MonteCarlo/output/20260222_143000
```

### `posterior.npz` (compressed NumPy archive)

| Array key | Shape | Description |
|-----------|-------|-------------|
| `samples` | `(n_posterior, D)` | Posterior draws for the selected model |
| `spearman_corr` | `(N, N)` | Spearman rank-correlation matrix |
| `returns_hist` | `(N, T)` | Historical log-returns used for fitting |

Load with: `np.load("posterior.npz")`

---

## 8. Performance Notes

| Portfolio size | MCMC time (both models) | MC simulation (10k paths, 252 days) |
|---------------|-------------------------|--------------------------------------|
| 5 stocks       | ~3–8 min                | ~30 s                                |
| 15 stocks      | ~10–20 min              | ~2 min                               |
| 30 stocks      | ~30–60 min              | ~5 min                               |

*Measured on a single CPU core (Python/NumPy, no parallelism within chain).*

**Bottleneck:** The λ_it update in Model A and z_it update in Model B are both
`O(N × T)` per iteration but are fully vectorised in NumPy (one `rng.gamma()` call).
The Metropolis steps for σ_i are also vectorised across stocks.

**Reduce runtime by:**
- Lowering `warmup` (minimum 500 recommended)
- Lowering `n_samples` (5000 post-warmup is the default; 2000 gives coarser posteriors)
- Using fewer chains (2 still gives R-hat, but with more uncertainty)
- Reducing `history_days` (fewer observations → faster per-iteration)

**WAIC computation:** Uses a subsample of 300 posterior draws (configurable via
`n_sub` in `select_model()`). Full 10k draws would be 100× slower with marginal
accuracy gain for model selection.

---

## 9. Known Limitations & Future Work

### Current limitations

1. **No within-Python parallelism:** The 4 chains run sequentially. True parallelism
   (via `multiprocessing`) would give ~3× speedup.

2. **Copula is a plug-in estimate:** The Spearman correlation matrix used by the
   Student-t copula is a single point estimate — it carries no posterior
   uncertainty, unlike the marginal parameters.  A Wishart posterior or Bayesian
   bootstrap over the correlation would propagate this uncertainty.

3. **WAIC subsampling:** We use 300 posterior draws for WAIC (out of potentially
   10,000). This is a statistical approximation. Using more draws improves precision
   at linear cost.

4. **Single-regime copula:** The correlation structure (Cholesky of Spearman corr.)
   is constant across all simulation paths. Regime-dependent correlations (calm vs.
   stress) would be more realistic for Model B.

5. **No rebalancing:** The simulation assumes buy-and-hold over the full horizon.
   Periodic rebalancing is not implemented.

### Future improvements

- [ ] Parallel chain execution via `concurrent.futures`
- [ ] NUTS (No-U-Turn Sampler) for more efficient exploration of complex posteriors
- [x] Student-t copula for tail dependence (replaces Gaussian copula, configurable df)
- [ ] Regime-dependent correlation matrices for Model B
- [ ] Backtesting module with rolling-window calibration test
- [ ] Export to PDF report
- [ ] API mode (FastAPI backend + React frontend) for production deployment
