# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantitative finance research repository with two main components:
1. **`wallstreet_quant/`** — Python package for SEC filing analysis using LLM-powered extraction (OpenAI GPT)
2. **`MonteCarlo/`** — Streamlit app for Bayesian portfolio risk modelling with from-scratch MCMC inference
3. **`Notebooks/`** — Jupyter notebooks for various stock screening strategies (Magic Formula, pair trading, REITs, dividends, short squeeze detection)
4. **`cluster.py`** — Standalone HRP (Hierarchical Risk Parity) portfolio construction with sector caps

## Running the MonteCarlo App

```bash
cd MonteCarlo
pip install -r requirements.txt
streamlit run app.py
```

Requirements: yfinance, numpy, scipy, pandas, streamlit, plotly, pyyaml, scikit-learn

## Architecture

### wallstreet_quant Package

- **`edgar_extractor.py`** — Fetches 10-K/10-Q/20-F/6-K filings from SEC EDGAR using `edgartools`. Requires `edgar_identity` env var set to `"user_name email@server.com"`. Core functions: `fetch_10K_and_10Q_filings()`, `extract_items_from_filing()`, `extract_financials()`, `chunk_text()`.
- **`edgar_ai.py`** — LLM analysis functions using OpenAI's `responses.parse()` with Pydantic structured output. Each function targets a specific SEC filing section (risk factors, MD&A, legal, controls, business, tone shift, strategy, human capital, earnings calls, competitor extraction). Default model: `gpt-5.2-pro`.
- **`edgar_pipeline.py`** — `SecAnalysis` orchestrator class. Calls all `edgar_ai` functions per ticker, consolidates via `o3` model into buy recommendation (positive/neutral/negative). Handles missing sections with whole-filing fallback. Logs to date-stamped file.
- **`utils.py`** — `CompanyDeduper` class for fuzzy company name matching using sentence-transformers (BAAI/bge-small-en-v1.5), FAISS for ANN search, and NetworkX connected components for clustering.
- **`montecarlo.py`** — `AbstractMonteCarlo` ABC and `NaiveMonteCarlo` implementation (simpler than the MonteCarlo/ app). Uses yfinance, supports alpha correction and custom weights.

Import pattern: modules use try/except to support both `from edgar_ai import *` (when run from within the package directory) and `from wallstreet_quant.edgar_ai import *` (when imported as a package).

### MonteCarlo App (Bayesian MCMC)

Two-tab Streamlit dashboard (`app.py`):
- **Tab 1 (MCMC)**: Fits two hierarchical Bayesian models to stock returns, selects via WAIC
- **Tab 2 (Simulation)**: Forward Monte Carlo using Student-t copula (configurable df, default 5) with posterior parameter draws

Key modules:
- `models/model_a.py` — Hierarchical Student-t (Gibbs + Metropolis)
- `models/model_b.py` — Hierarchical Normal Mixture K=2 (regime-switching)
- `mcmc/sampler.py` — `GibbsSampler` with adaptive step sizes
- `mcmc/diagnostics.py` — Split-chain R-hat, bulk ESS
- `selection/waic.py` — WAIC computation, model comparison (|ΔWAIC| > 4 threshold)
- `simulation/montecarlo.py` — Forward simulation, VaR/CVaR, per-stock CVaR decomposition
- `data/fetcher.py` — Yahoo Finance data with winsorisation (±15%), short-history filtering
- `output/visualizations.py` — All Plotly figures

Results saved to `MonteCarlo/output/<YYYYMMDD_HHMMSS>/` as `config.yaml` + `posterior.npz`.

## Key Dependencies

- **SEC data**: `edgartools` (requires `edgar_identity` env var)
- **LLM**: `openai` SDK using `client.responses.parse()` with Pydantic `text_format`
- **Finance data**: `yfinance` (auto-adjusted prices = total returns including dividends)
- **Name dedup**: `sentence-transformers`, `faiss-cpu`, `networkx`
- **Notebooks**: `pandas`, `numpy`, `plotly`, `scipy`, `sklearn`

## Conventions

- OpenAI structured output uses `client.responses.parse(model=..., input=[...], text_format=PydanticModel)` pattern throughout
- All LLM analysis functions in `edgar_ai.py` accept a `model` parameter defaulting to `gpt-5.2-pro`
- Pipeline logging goes to both console and date-stamped log files
- No test suite exists; notebooks serve as integration tests
