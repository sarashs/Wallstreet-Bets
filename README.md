# Wallstreet-Bets
This repository contains my quantitative studies and analysis of the stock market for personal trading.
![repo image](Images/repo_image.png)

---

### ⚠️ Disclaimer

This repository is **not financial advice**. It is intended solely for educational and informational purposes. The analyses, models, and tools provided herein reflect my **personal investment approach** and are made available on an as-is basis.

By using any code, strategies, or insights from this repository, you acknowledge that you do so **at your own risk**. I am **not a licensed financial advisor**, and I assume **no liability** for any financial loss, damage, or legal consequences that may result from actions taken based on this material.

Use your own judgment or consult with a qualified professional before making any financial decisions.

---

## Key Features

### 📊 Quantitative Analysis
- **Statistical Modeling**: Advanced statistical methods for portfolio optimization and risk assessment
- **Machine Learning Integration**: LLM-powered fundamental analysis and competitive intelligence extraction
- **Data-Driven Decisions**: Systematic approaches to stock selection and portfolio construction

### 🎯 Risk Management
- **Bayesian Monte Carlo Simulations**: Full posterior inference over return distributions using hierarchical MCMC models
- **Value at Risk (VaR)**: Quantify potential losses at different confidence levels
- **Tail Risk Analysis**: CVaR calculations and per-stock CVaR contribution decomposition
- **Stress Testing**: Evaluate portfolio resilience under extreme market conditions

### 🔍 Market Intelligence
- **SEC Filings Analysis**: Automated extraction of competitive relationships and business insights
- **Dividend Analysis**: Systematic screening for high-quality dividend-paying stocks; total-return (DRIP) modelling
- **Sector Analysis**: REIT and industry-specific investment strategies
- **Network Analysis**: Visualize competitive landscapes and market structures

### 🛠️ Technical Infrastructure
- **Modular Design**: Reusable components for building custom analysis workflows
- **Scalable Processing**: Efficient handling of large datasets and complex calculations
- **Interactive Visualizations**: Comprehensive Plotly dashboards and Streamlit apps
- **API Integration**: Direct connections to Yahoo Finance and AI services

---

## Notebooks

| Notebook | Description |
|---|---|
| **Dividend_stock_finder** | Finding good dividend stocks with fundamentals assessed by an LLM |
| **Pair_trading** | ADF test and visualization for identifying stocks for pair trading |
| **REIT_analysis** | Analysing and finding independent high-dividend REITs for investment |
| **Magic_formula** | Modified Greenblatt Magic Formula for screening stocks with LLM-powered 10-K analysis via the `wallstreet_quant` package |
| **Companies_competitive_relationship** | Extract and visualize self-reported competitive relationships between companies using SEC 10-K filings with AI-powered analysis and network graphs |
| **Fundamental_analysis** | Standalone SEC 10-K/10-Q report generator — given a list of tickers, runs LLM-powered filing analysis and exports results with sentiment (positive/neutral/negative) to Excel |
| **short_squeeze_detection** | Statistical signals for identifying potential short-squeeze candidates |

---

## MonteCarlo — Bayesian Portfolio Simulation Platform

A self-contained Streamlit application for **Bayesian portfolio risk modelling** using from-scratch MCMC inference and forward Monte Carlo simulation.

### How it works

```
Yahoo Finance prices  →  Log-returns (total return)  →  MCMC (Model A or B)
       ↓                                                        ↓
  Dividend data                                         WAIC model selection
  (display only)                                               ↓
                                                     Forward simulation
                                                     (Gaussian copula)
                                                             ↓
                                           Fan chart · VaR · CVaR · distributions
```

### Models

**Model A — Hierarchical Student-t**

Each stock's daily log-return is Student-t distributed via a Gamma scale-mixture augmentation. Per-stock means and volatilities are drawn from a shared global prior (Bayesian shrinkage toward the group mean).

```
μ₀, τ           → μᵢ  ~  N(μ₀, 1/τ²)          [per stock, Gibbs]
νₛ, ξ           → σᵢ  ~  InvGamma(νₛ/2, νₛξ²/2) [per stock, Metropolis]
ν               → λᵢₜ ~  Gamma(ν/2, ν/2)         [per (stock,day), Gibbs]
μᵢ, σᵢ, λᵢₜ    → rᵢₜ ~  N(μᵢ, σᵢ²/λᵢₜ)
```

**Model B — Hierarchical Normal Mixture (K = 2)**

Each stock has two regimes (calm / volatile). A per-stock Bernoulli latent variable selects the active regime each day. Component parameters share global hyperpriors across stocks.

```
μ₀ₖ, τₖ         → μᵢₖ  ~  N(μ₀ₖ, 1/τₖ²)          [per stock·regime, Gibbs]
νₖ, ξₖ          → σᵢₖ  ~  InvGamma(νₖ/2, νₖξₖ²/2)  [per stock·regime, Metropolis]
α_π, β_π        → πᵢ   ~  Beta(α_π, β_π)            [per stock, Gibbs]
πᵢ              → zᵢₜ  ~  Bernoulli(πᵢ)             [per (stock,day), Gibbs]
μᵢ,zᵢₜ, σᵢ,zᵢₜ → rᵢₜ  ~  N(μᵢ,zᵢₜ, σᵢ,zᵢₜ²)
```

Identifiability constraint: `σᵢ₀ < σᵢ₁` (component 0 is always the low-volatility regime).

### MCMC configuration (defaults)

| Setting | Default | Range |
|---|---|---|
| History (trading days) | 504 (~2 years) | 252 – 1 260 |
| Chains | 4 | — |
| Warmup (burn-in + adapt) | 2 000 | user-configurable |
| Post-warmup samples | 5 000 | user-configurable |
| Thinning | 2 | — |
| Stored draws per chain | 2 500 | — |

Step sizes are adapted every 100 warmup iterations toward target acceptance rates (35% scalar, 25% vectorised). Step sizes are frozen after warmup.

### Model selection — WAIC

Both models are fitted every run. WAIC (Widely Applicable Information Criterion) is computed from pointwise log-likelihoods:

```
WAIC = -2 · (lppd - p_waic)
```

Selection rule: if |ΔWAIC| > 4 the lower-WAIC model wins; otherwise Model A is chosen (simpler prior).

### Forward simulation

- Draw a posterior parameter set for the selected model.
- Generate correlated uniform marginals via Gaussian copula (Spearman rank-correlation Cholesky).
- Transform to per-stock returns using the model's conditional distribution.
- Compound daily returns over the chosen horizon to get terminal portfolio return.
- Repeat for the configured number of simulations.

### Risk metrics

| Metric | Description |
|---|---|
| Expected return | Mean terminal portfolio return |
| Median return | 50th percentile terminal return |
| Volatility (ann.) | Annualised std of daily portfolio returns |
| VaR (5%) | 5th percentile terminal return |
| CVaR (5%) | Mean of returns below VaR — expected shortfall |
| Prob. of loss | Fraction of simulations with negative return |
| Skewness / Kurtosis | Higher moment diagnostics |
| Per-stock CVaR contribution | Each stock's marginal share of portfolio CVaR |

### Dividend handling

`yfinance` returns **dividend-adjusted** close prices (`auto_adjust=True`). The daily log-return:

```
r_t = log(adj_close_t / adj_close_{t-1})
```

is therefore a **total return** — price appreciation plus dividend income on ex-dividend dates — implicitly assuming full dividend reinvestment (DRIP). No separate dividend accounting is needed in the MCMC.

Raw dividend history and trailing-12-month yields are fetched separately and shown in the portfolio summary table for reference.

### Data quality checks

- Tickers with fewer than 200 trading days of history are **dropped before calendar alignment** so that short-history assets do not trim the lookback window for the rest of the portfolio.
- Tickers with 75–100% of the requested window generate a warning.
- Returns are winsorised at ±15% per day to suppress data errors.

### Dashboard tabs

**Tab 1 — MCMC**
1. Enter tickers and portfolio weights (individually or via CSV paste).
2. Configure MCMC settings and history lookback.
3. Run MCMC — both models are fitted in parallel chains.
4. Review:
   - Portfolio summary with dividend yields
   - WAIC model selection card and rationale
   - Graphical model plate diagrams for both Model A and Model B
   - R-hat convergence diagnostics, ESS table
   - MCMC trace plots and posterior density plots
   - Bayesian shrinkage plot (sample mean vs posterior mean)
5. Save results to a timestamped folder (`output/<YYYYMMDD_HHMMSS>/`):
   - `config.yaml` — full run metadata
   - `posterior.npz` — all posterior samples

**Tab 2 — Simulation**
1. Load a previously saved `config.yaml` (or use the session result directly).
2. Set simulation horizon (trading days) and number of simulations.
3. Run forward simulation.
4. Review:
   - Key risk metrics row
   - Portfolio return fan chart (5/25/50/75/95 percentile bands)
   - Terminal return distribution histogram with VaR/CVaR/E[R] markers
   - Per-stock CVaR contribution bar chart
   - Per-stock summary table

### Directory structure

```
MonteCarlo/
├── app.py                   # Streamlit dashboard (2 tabs)
├── requirements.txt
├── IMPLEMENTATION.md        # Full mathematical and code documentation
├── data/
│   └── fetcher.py           # Yahoo Finance download, cleaning, dividend fetch
├── models/
│   ├── model_a.py           # Hierarchical Student-t (Gibbs + Metropolis)
│   └── model_b.py           # Hierarchical Normal Mixture K=2
├── mcmc/
│   ├── sampler.py           # GibbsSampler with step-size adaptation
│   └── diagnostics.py       # Split-chain R-hat, bulk ESS
├── selection/
│   └── waic.py              # WAIC computation and model comparison
├── simulation/
│   └── montecarlo.py        # Forward simulation, risk metrics, CVaR decomposition
├── output/
│   ├── visualizations.py    # All Plotly figures (fan chart, diagnostics, model graphs)
│   └── <YYYYMMDD_HHMMSS>/   # Timestamped run output
│       ├── config.yaml
│       └── posterior.npz
└── logs/
    └── app.log
```

### Running the app

```bash
cd MonteCarlo
pip install -r requirements.txt
streamlit run app.py
```

---

## `wallstreet_quant` Package

A comprehensive Python package for quantitative finance analysis:

| Module | Description |
|---|---|
| **edgar_extractor** | Automated fetching and parsing of 10-K and 10-Q filings from EDGAR |
| **edgar_ai** | GPT-4 integration for extracting competitive intelligence from SEC filings |
| **utils** | Advanced name matching and canonicalization for company deduplication |
| **montecarlo** | Abstract base classes for Monte Carlo simulation strategies |
| **edgar_pipeline** | End-to-end workflows for collecting, processing, and analysing financial data |

### To be implemented
- Revenue growth filter
- Insider share purchase / sale screening
- Share buyback detection
