# Multi-Year Hardware/Semiconductor Financial Metrics Extraction Guide

## Time Series Requirements for Hardware/Semiconductor Companies

Hardware/semiconductor analysis requires **multiple years/quarters** of data to assess:
- Product cycle performance and timing
- Technology node transition success
- Supply chain resilience and inventory management
- R&D efficiency and innovation pipeline
- Capital allocation and fab utilization

## 1. Hardware/Semiconductor Operating Strength Metrics (Multi-Year)

### Gross Margin Cycle Management
- **Time Period**: Last 12 quarters minimum (3 years)
- **Extract from**: Income Statement (multiple periods)
- **Calculation**: `Gross_Margin = (Revenue - COGS) / Revenue`
- **Hardware Threshold**: Gross margins >40% for semiconductors, >25% for hardware
- **Cycle Analysis**: Track margins through product launch cycles and competitive responses

### R&D Efficiency Trends
- **Time Period**: Last 3-5 years
- **Calculation**: 
  - `R&D_Intensity = R&D_Expense / Revenue`
  - `R&D_ROI = (New_Product_Revenue) / (Cumulative_R&D_Last_3_Years)`
- **Hardware Focus**: R&D intensity 15-25% for leading-edge semiconductors, 5-15% for hardware

### Fab Utilization and Capital Efficiency
- **Time Period**: Last 8 quarters
- **Metrics**:
  - `Asset_Turnover = Revenue / PP&E`
  - `Capacity_Utilization = Actual_Production / Max_Capacity`
- **Threshold**: Asset turnover >1.5x for semiconductors, utilization >75%

## 2. Hardware/Semiconductor Growth Metrics (Multi-Year Required)

### Product Cycle Revenue Analysis
- **Time Period**: 5+ years to capture full product cycles
- **Calculation**: Track revenue growth through:
  - Product introduction phase (high growth)
  - Market penetration phase (steady growth)
  - Maturity phase (declining growth)
  - Replacement cycle (new product introduction)

### Technology Node Transition Success
- **Time Period**: Last 2-3 major technology transitions
- **Metrics**:
  - Time to market vs competitors
  - Yield ramp speed (manufacturing efficiency)
  - Market share gains in new nodes
- **Analysis**: Faster transitions = competitive advantage

### Design Win Momentum
- **Time Period**: Last 8 quarters
- **Track**:
  - Number of design wins per quarter
  - Time from design win to revenue
  - Customer concentration trends
- **Leading Indicator**: Design wins predict revenue 6-18 months ahead

## 3. Hardware/Semiconductor Profitability Trends (Multi-Year)

### Manufacturing Learning Curve
- **Time Period**: Product lifecycle (2-4 years typical)
- **Calculation**: Track cost reduction over time:
  - `Cost_Reduction_Rate = (Initial_Cost - Current_Cost) / Initial_Cost`
  - `Yield_Improvement = Current_Yield / Initial_Yield`
- **Threshold**: 20-30% cost reduction annually in first 2 years

### Pricing Power Analysis
- **Time Period**: Last 3-5 years through cycles
- **Metrics**:
  - Average selling price (ASP) trends
  - Price elasticity during downturns
  - Premium capture for new technologies
- **Assessment**: Ability to maintain pricing during commodity cycles

### Operating Leverage Effectiveness
- **Time Period**: Last 3-5 years
- **Analysis**:
  - Fixed cost absorption during volume changes
  - Incremental margins on volume increases
  - Operating expense scaling vs revenue
- **Target**: 60-80% incremental margins for semiconductor companies

## 4. Hardware/Semiconductor Balance Sheet Evolution (Multi-Year)

### Inventory Cycle Management
- **Time Period**: Last 12 quarters minimum
- **Metrics**:
  - `Inventory_Turns = COGS / Average_Inventory`
  - `Days_Inventory_Outstanding = 365 / Inventory_Turns`
  - `Inventory_as_Percent_Revenue = Inventory / Quarterly_Revenue`
- **Thresholds**: 4-6x turns for semiconductors, 6-12x for hardware assembly

### Working Capital Efficiency
- **Time Period**: Last 8 quarters
- **Components**:
  - Days Sales Outstanding (DSO): 30-60 days typical
  - Days Payable Outstanding (DPO): 45-90 days typical  
  - Cash conversion cycle optimization
- **Target**: Negative cash conversion cycle preferred

### Capital Investment Cycles
- **Time Period**: Last 5-7 years (full investment cycles)
- **Analysis**:
  - CapEx as % of revenue through cycles
  - Time from investment to revenue generation
  - Return on invested capital (ROIC) trends
- **Pattern**: High CapEx → capacity expansion → revenue growth → margin expansion

## 5. Hardware/Semiconductor Cash Flow Patterns (Multi-Year)

### Free Cash Flow Cyclicality
- **Time Period**: Last 2-3 complete business cycles
- **Method**:
  1. Track FCF through up-cycles and down-cycles
  2. Assess FCF conversion rates at different revenue levels
  3. Analyze CapEx flexibility during downturns
- **Target**: Positive FCF even at cycle troughs

### Capital Allocation Discipline
- **Time Period**: Last 5-10 years
- **Track**:
  - CapEx timing relative to cycles
  - Share buyback patterns (buy low, reduce high)
  - Dividend sustainability through cycles
  - M&A timing and integration success

## 6. Required Hardware/Semiconductor Data Structure

```python
hardware_financial_data = {
    'annual': {
        '2024': {
            'balance_sheet': df1, 'income': df2, 'cashflow': df3,
            'r&d_expense': 2.1e9, 'capex': 1.5e9, 'inventory_turns': 5.2
        },
        '2023': {
            'balance_sheet': df4, 'income': df5, 'cashflow': df6,
            'r&d_expense': 1.8e9, 'capex': 2.1e9, 'inventory_turns': 4.8
        }
    },
    'quarterly': {
        'Q3_2024': {
            'balance_sheet': df10, 'income': df11, 'cashflow': df12,
            'gross_margin': 0.58, 'fab_utilization': 0.82, 'asp_trend': 1.05
        }
    },
    'product_data': {
        'design_wins': {'Q3_2024': 25, 'Q2_2024': 18},
        'new_product_revenue': {'2024': 0.35, '2023': 0.28},  # % of total revenue
        'technology_nodes': ['7nm', '5nm', '3nm']
    },
    'industry_data': {
        'semiconductor_cycle': {'phase': 'recovery', 'year_in_cycle': 2},
        'capex_intensity_peers': 0.18  # Industry average CapEx/Revenue
    }
}
```

## 7. Hardware/Semiconductor-Specific Multi-Year Calculations

### Gross Margin Sustainability Analysis
```python
def analyze_gross_margin_cycles(margin_by_quarter, product_cycles):
    margins = list(margin_by_quarter.values())
    
    return {
        'avg_margin': np.mean(margins),
        'margin_volatility': np.std(margins),
        'trough_margin': min(margins),  # Worst-case scenario
        'peak_margin': max(margins),    # Best-case scenario
        'cycle_resilience': min(margins) > 0.30,  # Maintains 30%+ at trough
        'competitive_position': assess_margin_vs_peers(margins)
    }
```

### R&D Efficiency Tracking
```python
def calculate_rnd_efficiency(rnd_by_year, revenue_by_year, new_product_revenue):
    rnd_intensity = {year: rnd_by_year[year]/revenue_by_year[year] 
                     for year in rnd_by_year.keys()}
    
    # Calculate 3-year R&D ROI
    total_rnd_3yr = sum(list(rnd_by_year.values())[-3:])
    new_product_rev_current = new_product_revenue['current_year']
    
    return {
        'rnd_intensity_trend': analyze_trend(rnd_intensity),
        'rnd_roi': new_product_rev_current / total_rnd_3yr if total_rnd_3yr > 0 else 0,
        'innovation_pipeline': assess_pipeline_strength(new_product_revenue),
        'efficient_rnd': rnd_roi > 0.3 and rnd_intensity_stable(rnd_intensity)
    }
```

### Inventory Management Assessment
```python
def analyze_inventory_management(inventory_turns_by_quarter, revenue_volatility):
    turns = list(inventory_turns_by_quarter.values())
    
    return {
        'avg_turns': np.mean(turns),
        'turns_consistency': np.std(turns) < 1.0,  # Stable inventory management
        'cycle_responsiveness': assess_inventory_flexibility(turns, revenue_volatility),
        'working_capital_efficiency': min(turns) > 3.0,  # Never below 3x turns
        'supply_chain_resilience': analyze_inventory_buffer_adequacy(turns)
    }
```

## 8. Hardware/Semiconductor-Specific Implementation Requirements

1. **Cycle Awareness**: Analyze performance across semiconductor up/down cycles
2. **Product Lifecycle Tracking**: Account for 2-4 year product cycles
3. **Technology Transitions**: Assess success in node migrations (28nm→14nm→7nm→5nm)
4. **Seasonality**: Q4 typically strongest for consumer semiconductors
5. **Geopolitical Factors**: Supply chain resilience and geographic diversification

## 9. Hardware/Semiconductor Screening Framework Integration

Your hardware/semiconductor screening framework needs these time-series checks:

### **Operating Strength**: 
- Gross margins >40% maintained through cycles (semiconductors)
- R&D intensity 15-25% with improving ROI trends
- Positive FCF even at cycle troughs

### **Growth**: 
- Design win momentum (growing quarterly)
- Successful technology node transitions
- Market share gains in new product categories

### **Profitability**: 
- Manufacturing learning curves (20-30% cost reduction)
- Operating leverage >60% incremental margins
- Pricing power during commodity cycles

### **Balance Sheet Management**: 
- Inventory turns >4x consistently
- Negative cash conversion cycle
- ROIC >15% through cycles

### **Innovation Pipeline**: 
- New product revenue >25% of total
- Time-to-market leadership vs competitors
- Patent portfolio strength and licensing income

## 10. Hardware/Semiconductor Risk Factor Analysis

### Technology Obsolescence Risk
- **Time Period**: Product lifecycle analysis (2-4 years)
- **Metrics**: Time to technology refresh, competitor response time
- **Threshold**: Continuous innovation pipeline with overlapping product cycles

### Customer Concentration Risk
- **Time Period**: Last 3-5 years
- **Monitor**: Top 5 customer concentration, design win diversification
- **Threshold**: No single customer >20% of revenue

### Cyclical Sensitivity
- **Time Period**: Last 2-3 semiconductor cycles
- **Assess**: Revenue decline during downturns, recovery speed
- **Benchmark**: Outperform industry during both up and down cycles

### Geopolitical Supply Chain Risk
- **Time Period**: Last 3-5 years + forward looking
- **Track**: Geographic revenue/supply exposure, local content requirements
- **Mitigation**: Diversified supply chain, multiple fab locations

The hardware/semiconductor screening requires deep understanding of technology cycles, manufacturing excellence, and innovation pipeline strength rather than just traditional financial metrics.
