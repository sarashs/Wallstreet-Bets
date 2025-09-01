# Wallstreet-Bets
This is repository contains my quantitative studies and analysis of the stock market for personal trading.
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
- **Monte Carlo Simulations**: Probabilistic modeling of portfolio performance under various market scenarios
- **Value at Risk (VaR)**: Quantify potential losses at different confidence levels
- **Tail Risk Analysis**: CVaR calculations to understand worst-case scenario impacts
- **Stress Testing**: Evaluate portfolio resilience under extreme market conditions

### 🔍 Market Intelligence
- **SEC Filings Analysis**: Automated extraction of competitive relationships and business insights
- **Dividend Analysis**: Systematic screening for high-quality dividend-paying stocks
- **Sector Analysis**: REIT and industry-specific investment strategies
- **Network Analysis**: Visualize competitive landscapes and market structures

### 🛠️ Technical Infrastructure
- **Modular Design**: Reusable components for building custom analysis workflows
- **Scalable Processing**: Efficient handling of large datasets and complex calculations
- **Interactive Visualizations**: Comprehensive plotting and dashboard capabilities
- **API Integration**: Direct connections to financial data sources and AI services

---

### Notebooks:
- **Dividend_stock_finder**: Finding good dividend stocks with fundamentals assessed by an LLM  
- **Pair_trading**: Basic code for ADF test and visualization for finding stocks for pair trading (over days)  
- **REIT_analysis**: Analysing and finding independent high dividend REITs for investment  
- **Magic_formula**: Modified Greenblatt Magic Formula for screening stocks plus further analysis of the 10-K filings for domestic US stocks. The latter uses the `wallstreet_quant` package 
- **Companies_competitive_relationship**: Extract and visualize self-reported competitive relationships between companies using SEC 10-K filings with AI-powered analysis and network graphs
- **Monte Carlo Simulations**: Portfolio risk analysis using Monte Carlo methods to simulate future returns, calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR) metrics

### Packages:
- **wallstreet_quant**: A comprehensive Python package for quantitative finance analysis including:
  - **edgar_extractor**: Automated fetching and parsing of 10-K and 10-Q filings from EDGAR database
  - **edgar_ai**: GPT-4 integration for extracting competitive intelligence and business insights from SEC filings
  - **utils**: Utilities such as advanced name matching and canonicalization for handling different company naming conventions (deduplication)
  - **montecarlo**: Abstract base classes for implementing various Monte Carlo simulation strategies
  - **Risk Analytics**: Built-in functions for calculating financial risk metrics and portfolio optimization
  - **edgar_pipeline**: End-to-end workflows for collecting, processing, and analyzing financial data

  ### To be implemented:
  - revenue growth filter
  - Insider share purchase or sale
  - Share byback by the company