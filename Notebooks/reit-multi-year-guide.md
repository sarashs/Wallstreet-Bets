# Multi-Year REIT Financial Metrics Extraction Guide

## Time Series Requirements for REITs

REIT analysis requires **multiple years/quarters** of data to assess:
- Interest rate cycle performance
- Portfolio quality trends
- Dividend sustainability patterns
- Leverage management through cycles
- Asset composition evolution

## 1. REIT Operating Strength Metrics (Multi-Year)

### Net Interest Margin (NIM) Consistency
- **Time Period**: Last 8 quarters minimum (2 years)
- **Extract from**: Income Statement (multiple periods)
- **Calculation**: `NIM = Net Interest Income / Average Earning Assets`
- **REIT Threshold**: NIM should be stable or improving, typically >1.0% for agency REITs
- **Implementation**: Track quarterly NIM and check for volatility <25%

### Operating Cash Flow Stability
- **Time Period**: Last 2-3 fiscal years
- **Calculation**: 
  - Year 1: Check CFO consistency vs interest payments
  - Year 2: Verify CFO covers dividend payments
  - Year 3: Assess CFO growth trend
- **REIT Focus**: CFO should cover 80%+ of dividend payments

## 2. REIT Growth Metrics (Multi-Year Required)

### Book Value Growth (3-Year)
- **Time Period**: Current year + 2 prior years (3 data points)
- **Calculation**: `Book_Value_CAGR = (BV_Year3/BV_Year1)^(1/2) - 1`
- **REIT Threshold**: Book value growth >3% annually
- **Data needed**: Book value per share from 3 consecutive annual reports

### Asset Growth Analysis
- **Time Period**: Last 8 quarters (2 years of quarterly data)
- **Calculation**: `Asset_Growth = (Total_Assets_Q1 - Total_Assets_Q1_PriorYear) / Total_Assets_Q1_PriorYear`
- **REIT Focus**: Controlled growth 5-15% annually, avoiding overleveraging

### Portfolio Composition Evolution
- **Time Period**: Last 3 years
- **Track**:
  - Agency vs Non-Agency securities ratio
  - Average asset duration trends
  - Geographic/sector diversification (for Equity REITs)

## 3. REIT Profitability Trends (Multi-Year)

### Net Interest Margin Trend
- **Time Period**: Last 12 quarters minimum
- **Calculation**: Calculate NIM for each quarter, then assess:
  - `NIM_Stability = StdDev(NIM_quarters) < 0.25%`
  - Interest rate sensitivity analysis
- **REIT Threshold**: NIM volatility <25%, maintaining >1.0% during rate cycles

### Economic Return Analysis
- **Time Period**: Last 3-5 years
- **Metrics needed**:
  - Total economic return (dividends + book value changes)
  - Return on equity trends
  - Risk-adjusted returns vs benchmarks
- **Check**: Economic returns >8% annually through interest rate cycles

### Hedging Effectiveness
- **Time Period**: Last 2-3 years + quarterly
- **Track**:
  - Derivative gains/losses vs interest rate movements
  - Duration gap management
  - Hedge ratio effectiveness

## 4. REIT Balance Sheet Evolution (Multi-Year)

### Leverage Cycle Management
- **Time Period**: Last 3 years + quarterly
- **Metrics**:
  - Debt-to-Equity ratio through rate cycles
  - Repo funding cost trends
  - Liquidity buffer maintenance
- **REIT Thresholds**: D/E ratio 6:1-10:1 range, stable funding costs

### Funding Diversification Trends
- **Time Period**: Last 3 years
- **Track**:
  - Repo vs long-term debt mix
  - Funding duration vs asset duration
  - Counterparty concentration limits

### Capital Adequacy Evolution
- **Time Period**: Last 2-3 years
- **Metrics**:
  - Tangible book value trends
  - Economic capital vs risk assets
  - Regulatory capital compliance (if applicable)

## 5. REIT Dividend Analysis (Multi-Year Critical)

### Dividend Coverage Sustainability
- **Time Period**: Last 12 quarters minimum
- **Method**:
  1. Calculate quarterly dividend coverage: `Coverage = (Net_Income + Non-Cash_Items) / Dividends_Paid`
  2. Track earnings vs dividend consistency
  3. Assess coverage through interest rate cycles
- **REIT Threshold**: Coverage ratio >0.9x consistently, >1.1x preferred

### Dividend Growth Consistency
- **Time Period**: Last 5-10 years
- **Analysis**:
  - Dividend cut history during stress periods
  - Growth rate sustainability (2-5% annually)
  - Special dividend frequency

### Distribution Quality Analysis
- **Time Period**: Last 3-5 years
- **Track**:
  - Return of capital vs income distribution
  - Taxable income vs GAAP earnings
  - Undistributed taxable income accumulation

## 6. Required REIT Data Structure

```python
reit_financial_data = {
    'annual': {
        '2024': {
            'balance_sheet': df1, 'income': df2, 'cashflow': df3,
            'book_value_per_share': 12.50, 'dividend_per_share': 1.44
        },
        '2023': {
            'balance_sheet': df4, 'income': df5, 'cashflow': df6,
            'book_value_per_share': 11.85, 'dividend_per_share': 1.44
        }
    },
    'quarterly': {
        'Q3_2024': {
            'balance_sheet': df10, 'income': df11, 'cashflow': df12,
            'nim': 0.012, 'dividend_coverage': 0.95
        },
        'Q2_2024': {
            'balance_sheet': df13, 'income': df14, 'cashflow': df15,
            'nim': 0.011, 'dividend_coverage': 1.02
        }
    },
    'market_data': {
        'interest_rates': {'10Y_treasury': [2.5, 3.1, 4.2]},
        'spread_environment': {'agency_spreads': [0.25, 0.35, 0.45]}
    }
}
```

## 7. REIT-Specific Multi-Year Calculations

### Net Interest Margin Stability
```python
def calculate_nim_stability(nim_by_quarter):
    nim_values = list(nim_by_quarter.values())
    return {
        'avg_nim': np.mean(nim_values),
        'nim_volatility': np.std(nim_values),
        'stable_nim': np.std(nim_values) < 0.0025,  # <25bps volatility
        'min_nim': min(nim_values),
        'rate_cycle_performance': assess_rate_sensitivity(nim_values)
    }
```

### Dividend Sustainability Analysis
```python
def analyze_dividend_sustainability(coverage_by_quarter, dividend_history):
    coverage_values = list(coverage_by_quarter.values())
    
    return {
        'avg_coverage': np.mean(coverage_values),
        'coverage_consistency': min(coverage_values) > 0.8,
        'dividend_cuts_5yr': count_dividend_cuts(dividend_history),
        'sustainable': np.mean(coverage_values) > 0.9 and min(coverage_values) > 0.7
    }
```

### Leverage Cycle Analysis
```python
def analyze_leverage_through_cycles(debt_equity_by_quarter, rate_environment):
    de_ratios = list(debt_equity_by_quarter.values())
    
    return {
        'avg_leverage': np.mean(de_ratios),
        'leverage_volatility': np.std(de_ratios),
        'max_leverage': max(de_ratios),
        'deleveraging_periods': count_deleveraging_periods(de_ratios),
        'rate_responsive': assess_leverage_rate_sensitivity(de_ratios, rate_environment)
    }
```

## 8. REIT-Specific Implementation Requirements

1. **Interest Rate Alignment**: Analyze performance across different rate environments
2. **Quarterly Focus**: REITs report quarterly; use quarterly data for trend analysis
3. **Sector Considerations**: Agency vs Non-Agency vs Equity REITs have different metrics
4. **Regulatory Changes**: Account for Basel III, CECL, and other regulatory impacts
5. **Market Conditions**: Adjust expectations based on credit spread environments

## 9. REIT Screening Framework Integration

Your REIT screening framework needs these time-series checks:

### **Operating Strength**: 
- 8+ quarters of stable NIM (volatility <25bps)
- CFO consistently covers 80%+ of dividends

### **Growth**: 
- 3-year book value CAGR >3%
- Asset growth 5-15% annually (controlled expansion)

### **Profitability**: 
- NIM maintained >1.0% through rate cycles
- Economic returns >8% annually

### **Dividend Coverage**: 
- Coverage ratio >0.9x for 8+ quarters
- No dividend cuts in last 5 years during stress

### **Leverage Management**: 
- D/E ratio 6:1-10:1 range maintained
- Responsive deleveraging during stress periods

## 10. REIT Risk Factor Analysis

### Duration Risk Management
- **Time Period**: Last 2-3 years
- **Metrics**: Asset-liability duration gap trends
- **Threshold**: Duration gap <6 months consistently

### Credit Risk Evolution  
- **Time Period**: Last 3-5 years
- **Track**: Credit loss history, non-performing assets
- **Focus**: Minimal credit losses for Agency REITs

### Liquidity Risk Assessment
- **Time Period**: Last 8 quarters
- **Monitor**: Repo funding concentration, unencumbered assets
- **Threshold**: >10% unencumbered assets, diversified funding

The REIT screening decisions require these multi-period analyses focused on interest rate sensitivity, dividend sustainability, and leverage management through economic cycles.
