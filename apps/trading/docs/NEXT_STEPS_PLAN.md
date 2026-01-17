# SI Research: Next Steps Implementation Plan (v4)

**Date**: January 17, 2026  
**Last Updated**: January 17, 2026 (Final Comprehensive Review)  
**Based on**: Expert Panel + Professor + Industry Final Review  
**Author**: Yuhao Li, University of Pennsylvania

---

## Overview

This document outlines detailed implementation plans for the expert-recommended next steps, organized by priority. **Version 4** is the final comprehensive version incorporating all feedback.

### Key Changes in v4
- Added **Turnover Analysis** to P1 (Dr. Chen - HF)
- Added **Regime Classification Accuracy** to P2 (Dr. Rodriguez)
- Added **SI-Only Baseline** requirement in P5 (Dr. Rodriguez)
- Added **Pre-Submission Checklist** to P7 (Prof. Chen)
- Added **SI Definition Sensitivity** analysis (Prof. Weber)
- Added **Negative Result Protocol** contingency section (Prof. Kumar)
- Added **Computational Cost Tracking** (Dr. Rodriguez)
- Added **Data Versioning** to Audit C (Prof. Weber)

### Key Changes in v3
- Added **P1.5 Factor Regression** explicitly (Prof. Kumar - Required)
- Added **Alpha Decay Analysis** to P1 (Prof. Chen - Optional → Added)
- Added **Parameter Documentation** requirements throughout (Prof. Weber)
- Updated execution timeline to include P1.5

### Key Changes in v2
- Added **P0 Critical Audits** before any execution
- Revised priority order based on expert recommendations
- Deferred Steps 5, 6 (partial), 8 to Phase 2
- Added market impact model and cost sensitivity to Step 1
- Added look-ahead bias audit to Step 2
- Expanded ensemble methods in Step 4
- Added economic significance checks throughout

---

# COMPUTATIONAL REQUIREMENTS (Dr. Rodriguez Addition)

Track compute time and resources for reproducibility:

| Step | Estimated Time | Compute | Memory |
|------|---------------|---------|--------|
| P0 Audits | 1-2 hours | Local laptop | 4GB |
| P1 Backtest (per asset) | 15-30 min | Local laptop | 4GB |
| P1 Full (11 assets) | 3-5 hours | Local laptop | 8GB |
| P1.5 Factor Regression | 30 min | Local laptop | 4GB |
| P2 Regime Analysis | 1-2 hours | Local laptop | 4GB |
| P3 Walk-Forward (per asset) | 30-60 min | Local laptop | 4GB |
| P3 Full (11 assets) | 6-10 hours | Local laptop | 8GB |
| P5 Ensemble | 2-4 hours | Local laptop | 8GB |
| P6 Extended (25 assets) | 12-24 hours | Cloud recommended | 16GB |

**Total Estimated Time**: 30-50 hours (can parallelize P3, P6)

---

# PARAMETER DOCUMENTATION (Prof. Weber Requirement)

All experiments must document parameter choices. Use this template:

```python
# At the top of each experiment script:

PARAMETER_CHOICES = {
    # SI Computation
    'si_window': 7,              # Days for SI rolling window (tested: 7, 14, 30)
    'n_agents_per_strategy': 3,  # Agents per strategy type (tested: 2, 3, 5)
    'n_strategies': 6,           # Total strategy types

    # Regime Detection
    'regime_lookback': 7,        # Days for regime classification
    'adx_trending_threshold': 25,# ADX > 25 = trending
    'adx_meanrev_threshold': 20, # ADX < 20 = mean-reverting
    'vol_threshold_sigma': 2.0,  # Vol > 2σ = volatile

    # Backtest
    'train_ratio': 0.70,         # 70% train
    'val_ratio': 0.15,           # 15% validation
    'test_ratio': 0.15,          # 15% test
    'embargo_days': 7,           # Gap between splits

    # Cost Model
    'crypto_fee_bps': 4,         # Crypto trading fee
    'forex_spread_bps': 1,       # Forex spread
    'stock_commission_bps': 2,   # Stock commission + spread
    'commodity_fee_bps': 3,      # Commodity futures fee

    # Random Seeds
    'random_seed': 42,           # For reproducibility
}

# Document WHY each choice was made:
PARAMETER_RATIONALE = {
    'si_window': "7 days balances responsiveness vs stability. Tested in sensitivity analysis.",
    'n_agents_per_strategy': "3 agents provides diversity without overfitting.",
    'regime_lookback': "7 days matches SI window for consistency.",
    # ... etc
}
```

| Parameter Category | Documentation Required | Location |
|-------------------|----------------------|----------|
| SI computation | ✅ | Each experiment script |
| Regime detection | ✅ | Each experiment script |
| Cost model | ✅ | `src/backtest/cost_model.py` |
| Data splits | ✅ | `src/data/loader_v2.py` |
| All thresholds | ✅ | Inline with rationale |

---

# P0: CRITICAL AUDITS (Before Any Execution)

**Timeline**: 1-2 days
**Gate**: Must pass ALL audits before proceeding to P1

## Audit A: Survivorship Bias Check

### Objective
Verify that asset selection was not biased by hindsight.

### Checklist

| Check | Status | Notes |
|-------|--------|-------|
| All crypto assets existed for full 5-year period | ⬜ | BTC, ETH, SOL - verify launch dates |
| No delisted tokens included | ⬜ | Check if any tokens died |
| Stock tickers haven't changed | ⬜ | SPY, QQQ, AAPL - verify continuity |
| Forex pairs continuously traded | ⬜ | Major pairs - should be fine |
| Selection criteria documented BEFORE analysis | ⬜ | Must be in pre-registration |

### Code

```python
# experiments/audit_survivorship.py

def check_survivorship():
    """Verify no survivorship bias in asset selection."""

    # Asset launch/availability dates
    ASSET_START_DATES = {
        'BTC': '2009-01-03',   # Genesis block
        'ETH': '2015-07-30',   # Mainnet launch
        'SOL': '2020-03-16',   # Mainnet beta - WARNING: < 5 years
        'SPY': '1993-01-29',
        'QQQ': '1999-03-10',
        'AAPL': '1980-12-12',
        'EURUSD': '1999-01-01',
        'GBPUSD': '1971-01-01',
        'USDJPY': '1971-01-01',
        'GC=F': '1974-12-31',  # Gold futures
        'CL=F': '1983-03-30',  # Crude oil futures
    }

    analysis_start = '2021-01-01'

    issues = []
    for asset, start in ASSET_START_DATES.items():
        if start > analysis_start:
            issues.append(f"⚠️ {asset} started {start}, after analysis period")
        years_available = (pd.Timestamp(analysis_start) - pd.Timestamp(start)).days / 365
        if years_available < 5:
            issues.append(f"⚠️ {asset} has only {years_available:.1f} years before analysis")

    return issues
```

### Success Criteria
- [ ] All assets have 5+ years of continuous data
- [ ] Selection criteria documented before analysis
- [ ] No hindsight in asset choice

---

## Audit B: Data Quality Verification

### Objective
Ensure data is clean and reliable before running any analysis.

### Checks

| Check | Threshold | Action if Failed |
|-------|-----------|------------------|
| Missing values | < 1% | Forward-fill or exclude |
| Gaps > 3 days | 0 | Document and handle |
| Extreme returns | \|r\| < 50% | Verify or winsorize |
| Duplicate timestamps | 0 | Remove duplicates |
| Timezone consistency | All UTC | Convert to UTC |
| Weekend/holiday data | Appropriate gaps | Verify market calendar |

### Code

```python
# experiments/audit_data_quality.py

def audit_data_quality(data: pd.DataFrame, asset: str) -> dict:
    """Run comprehensive data quality checks."""

    issues = {
        'asset': asset,
        'n_rows': len(data),
        'date_range': f"{data.index[0]} to {data.index[-1]}",
        'issues': []
    }

    # Check 1: Missing values
    missing_pct = data.isnull().sum() / len(data)
    if missing_pct.max() > 0.01:
        issues['issues'].append(f"Missing values: {missing_pct.max():.2%}")

    # Check 2: Gaps > 3 days
    gaps = data.index.to_series().diff()
    large_gaps = gaps[gaps > pd.Timedelta(days=3)]
    if len(large_gaps) > 0:
        issues['issues'].append(f"Large gaps: {len(large_gaps)} gaps > 3 days")

    # Check 3: Extreme returns
    returns = data['close'].pct_change()
    extreme = returns[returns.abs() > 0.5]
    if len(extreme) > 0:
        issues['issues'].append(f"Extreme returns: {len(extreme)} days with |r| > 50%")

    # Check 4: Duplicates
    duplicates = data.index.duplicated().sum()
    if duplicates > 0:
        issues['issues'].append(f"Duplicate timestamps: {duplicates}")

    issues['passed'] = len(issues['issues']) == 0
    return issues
```

### Success Criteria
- [ ] All assets pass data quality checks
- [ ] Issues documented and handled appropriately
- [ ] Audit report generated

---

## Audit C: Reproducibility Check

### Objective
Ensure results can be reproduced by others.

### Checklist

| Item | Status | Value |
|------|--------|-------|
| Python version documented | ⬜ | 3.x.x |
| All package versions pinned | ⬜ | requirements.txt |
| Random seeds set | ⬜ | SEED = 42 |
| Git commit hash recorded | ⬜ | For each run |
| Data source documented | ⬜ | Binance, yfinance |
| Hardware/OS documented | ⬜ | For timing benchmarks |

### Code

```python
# src/utils/reproducibility.py

import random
import numpy as np
import hashlib
import subprocess
import json

def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # If using PyTorch/TensorFlow, add those here

def get_reproducibility_manifest() -> dict:
    """Generate manifest for reproducibility."""
    import sys
    import pkg_resources

    manifest = {
        'python_version': sys.version,
        'packages': {pkg.key: pkg.version for pkg in pkg_resources.working_set},
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'git_dirty': subprocess.check_output(['git', 'status', '--porcelain']).decode().strip() != '',
        'seed': 42,
    }
    return manifest

def save_manifest(filepath: str):
    """Save reproducibility manifest to file."""
    manifest = get_reproducibility_manifest()
    with open(filepath, 'w') as f:
        json.dump(manifest, f, indent=2)
```

### Data Versioning (Prof. Weber Addition)

Pin exact data files for reproducibility:

```python
# experiments/generate_data_manifest.py

import hashlib
import json
import pandas as pd
from pathlib import Path

def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of file."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def generate_data_manifest(data_dir: str = "data") -> dict:
    """
    Generate manifest of all data files with checksums.
    """
    manifest = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'files': {}
    }
    
    for filepath in Path(data_dir).rglob("*.csv"):
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        manifest['files'][str(filepath)] = {
            'md5': compute_file_hash(filepath),
            'rows': len(df),
            'columns': list(df.columns),
            'date_range': f"{df.index[0]} to {df.index[-1]}",
            'file_size_kb': filepath.stat().st_size / 1024,
        }
    
    return manifest

def save_data_manifest(manifest: dict, filepath: str = "data/DATA_MANIFEST.json"):
    """Save manifest to JSON."""
    with open(filepath, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Data manifest saved to {filepath}")

def verify_data_manifest(manifest_path: str = "data/DATA_MANIFEST.json") -> bool:
    """
    Verify current data matches saved manifest.
    Returns True if all files match.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    mismatches = []
    for filepath, expected in manifest['files'].items():
        if not Path(filepath).exists():
            mismatches.append(f"Missing: {filepath}")
            continue
        
        actual_hash = compute_file_hash(filepath)
        if actual_hash != expected['md5']:
            mismatches.append(f"Changed: {filepath}")
    
    if mismatches:
        print("❌ Data verification failed:")
        for m in mismatches:
            print(f"   {m}")
        return False
    
    print("✅ All data files verified")
    return True
```

### Success Criteria
- [ ] Reproducibility manifest generated
- [ ] All random operations use fixed seed
- [ ] Requirements.txt has pinned versions
- [ ] **Data manifest with checksums created**
- [ ] **Data verification passes before each run**

---

## Audit D: Look-Ahead Bias Check

### Objective
Verify no future information is used in any calculation.

### Critical Points to Check

| Component | Check | Status |
|-----------|-------|--------|
| Regime classification | Uses only data[:t] at time t | ⬜ |
| SI computation | Rolling window uses only past data | ⬜ |
| Feature calculation | No future data in features | ⬜ |
| Train/Val/Test split | Temporal order preserved | ⬜ |
| Correlation analysis | No overlapping test data in training | ⬜ |

### Code Review Checklist

```python
# For each function, verify:

# ✅ CORRECT - uses only past data
def classify_regime(data, idx, lookback=7):
    window = data.iloc[idx-lookback:idx]  # Only past data
    return compute_regime(window)

# ❌ WRONG - uses future data
def classify_regime_WRONG(data, idx, lookback=7):
    window = data.iloc[idx-lookback:idx+lookback]  # Includes future!
    return compute_regime(window)
```

### Success Criteria
- [ ] Code review completed for all analysis functions
- [ ] No future data used in any calculation
- [ ] Documentation updated with verification

---

## Audit E: Economic vs Statistical Significance

### Objective
Verify that statistically significant correlations translate to tradeable edges.

### Framework

| Correlation | Statistical Significance | Economic Significance |
|-------------|--------------------------|----------------------|
| \|r\| = 0.10 | p < 0.05 may hold | ❌ Likely noise |
| \|r\| = 0.15 | p < 0.01 likely | ⚠️ Marginal after costs |
| \|r\| = 0.20 | p < 0.001 likely | ✅ Potentially tradeable |
| \|r\| = 0.30 | Highly significant | ✅ Strong edge |

### Calculation

```python
def assess_economic_significance(correlation: float,
                                  n_obs: int,
                                  avg_trades_per_year: int,
                                  cost_per_trade: float) -> dict:
    """
    Assess if correlation is economically significant.

    Rule of thumb:
    - Need |r| > 2 * sqrt(cost_drag) to be profitable
    - cost_drag = trades_per_year * cost_per_trade / expected_annual_return
    """

    # Rough estimate: correlation translates to IC
    # Expected return ≈ IC * volatility * sqrt(trades)
    estimated_ic = correlation
    assumed_vol = 0.20  # 20% annual vol
    annual_return_gross = estimated_ic * assumed_vol * np.sqrt(avg_trades_per_year)

    annual_cost = avg_trades_per_year * cost_per_trade * 2  # Round-trip
    annual_return_net = annual_return_gross - annual_cost

    return {
        'correlation': correlation,
        'estimated_gross_return': annual_return_gross,
        'annual_cost': annual_cost,
        'estimated_net_return': annual_return_net,
        'economically_significant': annual_return_net > 0.02,  # > 2% net
        'break_even_correlation': annual_cost / (assumed_vol * np.sqrt(avg_trades_per_year))
    }
```

### Success Criteria
- [ ] All reported correlations assessed for economic significance
- [ ] Break-even correlation computed for each strategy
- [ ] Only economically significant findings emphasized

---

# P1: BACKTEST WITH REALISTIC COSTS

**Timeline**: 3-4 days
**Dependencies**: P0 audits passed
**Gate**: Net Sharpe > 0 in at least 2/4 markets

## Objective
Validate that SI-based trading signals remain profitable after accounting for realistic transaction costs.

## 1.1 Transaction Cost Model (Enhanced)

### Base Costs

| Market | Fee | Slippage | Total (one-way) |
|--------|-----|----------|-----------------|
| Crypto | 4 bps | 2 bps | 6 bps |
| Forex | 0 bps | 1 bp | 1 bp |
| Stocks | 1 bp | 1 bp | 2 bps |
| Commodities | 2 bps | 1 bp | 3 bps |

### Market Impact Model (NEW)

For positions > 1% of ADV (Average Daily Volume):

```python
def market_impact(position_size: float, adv: float) -> float:
    """
    Square-root market impact model.

    Based on: Almgren & Chriss (2000)
    Impact = base_cost + eta * sqrt(position_size / ADV)
    """
    participation_rate = position_size / adv

    if participation_rate < 0.01:
        return 0  # No additional impact for small trades

    eta = 0.1  # Market impact coefficient (calibrated)
    impact = eta * np.sqrt(participation_rate)

    return impact
```

### Cost Sensitivity Analysis (NEW)

| Scenario | Multiplier | Use Case |
|----------|------------|----------|
| Optimistic | 0.5x | Best-case (market maker rebates) |
| Base | 1.0x | Expected case |
| Conservative | 2.0x | Stress test |
| Extreme | 3.0x | Black swan / illiquidity |

## 1.2 Strategy Variants to Test

| Variant | Description | Expected Turnover |
|---------|-------------|-------------------|
| **SI Threshold Long** | Long when SI > 0.6 | ~50 trades/year |
| **SI Momentum** | Long when dSI/dt > 0 | ~100 trades/year |
| **SI Regime Switch** | Different strategy per SI level | ~30 trades/year |
| **SI Risk Overlay** | Reduce position when SI < 0.4 | ~20 adjustments/year |

## 1.3 Code Structure

```python
# src/backtest/cost_model.py

class EnhancedCostModel:
    def __init__(self, market_type: str, cost_multiplier: float = 1.0):
        self.base_costs = TRANSACTION_COSTS[market_type]
        self.multiplier = cost_multiplier

    def calculate_total_cost(self,
                             trade_size: float,
                             adv: float,
                             volatility: float) -> float:
        """
        Calculate total transaction cost including:
        - Base fee
        - Slippage (scaled by volatility)
        - Market impact (for large trades)
        """
        base = self.base_costs['total'] * self.multiplier

        # Volatility-adjusted slippage
        vol_adjustment = volatility / 0.20  # Normalize to 20% vol
        slippage = self.base_costs['slippage'] * vol_adjustment

        # Market impact
        impact = self.market_impact(trade_size, adv)

        return base + slippage + impact

    def market_impact(self, trade_size: float, adv: float) -> float:
        if adv == 0 or trade_size / adv < 0.01:
            return 0
        return 0.1 * np.sqrt(trade_size / adv)
```

```python
# experiments/backtest_with_costs.py

def run_cost_sensitivity_analysis(data, si, strategy, market_type):
    """Run backtest with multiple cost scenarios."""

    results = {}

    for multiplier in [0.5, 1.0, 2.0, 3.0]:
        cost_model = EnhancedCostModel(market_type, multiplier)

        signals = strategy.generate_signals(data, si)
        gross_returns = calculate_returns(data, signals)
        net_returns = cost_model.apply_costs(gross_returns, signals)

        results[f'{multiplier}x'] = {
            'gross_sharpe': sharpe_ratio(gross_returns),
            'net_sharpe': sharpe_ratio(net_returns),
            'gross_return': gross_returns.sum(),
            'net_return': net_returns.sum(),
            'cost_drag': (gross_returns.sum() - net_returns.sum()) / gross_returns.sum(),
            'n_trades': (signals.diff() != 0).sum(),
        }

    return results
```

## 1.4 Alpha Decay Analysis (Prof. Chen Addition)

**Objective**: Measure how quickly the SI signal's predictive power decays over time.

### Why This Matters

| Half-Life | Interpretation | Tradability |
|-----------|----------------|-------------|
| < 1 day | Very short-lived signal | ❌ Untradeable (too fast) |
| 1-3 days | Short-lived | ⚠️ Marginal (high turnover) |
| 3-7 days | Medium persistence | ✅ Tradeable |
| > 7 days | Long-lasting | ✅ Comfortable |

### Code

```python
# experiments/analyze_alpha_decay.py

def compute_alpha_decay(si: pd.Series,
                        forward_returns: pd.Series,
                        max_lag: int = 20) -> dict:
    """
    Compute how SI correlation with future returns decays over time.

    Args:
        si: Specialization Index time series
        forward_returns: Asset returns
        max_lag: Maximum number of periods to test

    Returns:
        Decay curve and half-life estimate
    """
    from scipy.stats import spearmanr

    decay_curve = []

    for lag in range(1, max_lag + 1):
        # Compute correlation between SI(t) and Return(t+lag)
        si_lagged = si.iloc[:-lag]
        returns_future = forward_returns.iloc[lag:]

        # Align indices
        aligned = pd.concat([si_lagged, returns_future], axis=1).dropna()
        if len(aligned) < 30:
            continue

        corr, pval = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])

        decay_curve.append({
            'lag_days': lag,
            'correlation': corr,
            'p_value': pval,
            'significant': pval < 0.05
        })

    df = pd.DataFrame(decay_curve)

    # Compute half-life
    if len(df) > 0 and df['correlation'].iloc[0] != 0:
        lag1_corr = abs(df['correlation'].iloc[0])
        half_corr = lag1_corr / 2

        # Find first lag where |correlation| < half of lag-1
        below_half = df[df['correlation'].abs() < half_corr]
        half_life = below_half['lag_days'].min() if len(below_half) > 0 else max_lag
    else:
        half_life = None

    return {
        'decay_curve': df,
        'half_life_days': half_life,
        'lag1_correlation': df['correlation'].iloc[0] if len(df) > 0 else None,
        'tradeable': half_life is not None and half_life >= 3
    }


def plot_decay_curve(decay_results: dict, asset: str, save_path: str = None):
    """Plot the alpha decay curve."""
    import matplotlib.pyplot as plt

    df = decay_results['decay_curve']
    half_life = decay_results['half_life_days']

    plt.figure(figsize=(10, 6))
    plt.bar(df['lag_days'], df['correlation'],
            color=['green' if s else 'gray' for s in df['significant']])

    if half_life:
        plt.axvline(x=half_life, color='red', linestyle='--',
                    label=f'Half-life: {half_life} days')

    plt.xlabel('Lag (days)')
    plt.ylabel('Correlation with Future Returns')
    plt.title(f'Alpha Decay Curve: {asset}')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.close()
```

### Success Criteria

| Metric | Threshold | Status |
|--------|-----------|--------|
| Half-life | ≥ 3 days | ⬜ |
| Lag-1 correlation | \|r\| > 0.10 | ⬜ |
| Decay is gradual (not cliff) | Visual check | ⬜ |

### Deliverables

- [ ] `experiments/analyze_alpha_decay.py`
- [ ] Decay curve plots for each market
- [ ] Half-life table across assets
- [ ] Go/no-go decision based on tradability

---

## 1.5 Turnover Analysis (Dr. Chen - HF Addition)

**Objective**: Ensure strategy turnover is manageable.

```python
def compute_turnover_metrics(positions: pd.Series) -> dict:
    """
    Compute turnover metrics for strategy.
    
    Turnover = sum of absolute position changes over period
    Annual turnover of 100% means we trade the entire portfolio once/year
    """
    daily_turnover = positions.diff().abs()
    
    # Annualize (assuming daily data)
    annual_turnover = daily_turnover.sum() * (252 / len(positions))
    
    # Trades per year
    trade_signals = (positions.diff() != 0).sum()
    trades_per_year = trade_signals * (252 / len(positions))
    
    return {
        'annual_turnover_pct': annual_turnover * 100,
        'trades_per_year': trades_per_year,
        'avg_holding_period_days': 252 / max(trades_per_year, 1),
        'acceptable': annual_turnover < 2.0,  # < 200% annual turnover
    }
```

| Turnover Level | Annual % | Trades/Year | Assessment |
|----------------|----------|-------------|------------|
| Low | < 50% | < 25 | ✅ Excellent |
| Moderate | 50-100% | 25-50 | ✅ Good |
| High | 100-200% | 50-100 | ⚠️ Acceptable |
| Very High | > 200% | > 100 | ❌ Concerning |

---

## 1.6 Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Net Sharpe > 0 at 1x costs | Required | Minimum viability |
| Net Sharpe > 0 at 2x costs | Desired | Robust to cost uncertainty |
| Cost drag < 50% of gross | Required | Costs shouldn't dominate |
| Profitable in 2/4 markets | Required | Cross-market robustness |
| **Alpha half-life ≥ 3 days** | Required | Signal is tradeable |
| **Annual turnover < 200%** | Required | Costs manageable |

## 1.6 Deliverables

- [ ] `src/backtest/cost_model.py` (enhanced with market impact)
- [ ] `experiments/backtest_with_costs.py`
- [ ] `experiments/analyze_alpha_decay.py`
- [ ] `results/cost_analysis/` directory
- [ ] Cost sensitivity table (0.5x to 3x)
- [ ] Market-by-market performance comparison
- [ ] Alpha decay curves and half-life table

---

# P1.5: FACTOR REGRESSION (Prof. Kumar - REQUIRED)

**Timeline**: 0.5 days
**Dependencies**: P1 completed
**Gate**: Alpha t-stat > 2.0 after controlling for known factors

## Objective

Prove that SI is a **novel signal**, not just repackaged momentum, volatility, or other known factors.

## Why This is Critical

> **Prof. Chen (Editor)**: "I've rejected papers that claimed novel signals but were just repackaged factors. This table is NON-NEGOTIABLE."

| If SI is just... | Problem |
|------------------|---------|
| Momentum in disguise | Not novel, already traded by everyone |
| Inverse volatility | Known risk factor |
| Trend strength (ADX) | Already a common indicator |

## Factor Regression Specification

```python
# experiments/factor_regression.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List

def run_factor_regression(si_returns: pd.Series,
                          factors: pd.DataFrame,
                          use_hac: bool = True) -> Dict:
    """
    Regress SI strategy returns on known factors.

    Args:
        si_returns: Returns from SI-based strategy
        factors: DataFrame with columns for each factor
        use_hac: Use HAC standard errors for autocorrelation

    Returns:
        Regression results with alpha and factor loadings
    """

    # Align data
    aligned = pd.concat([si_returns.rename('si_returns'), factors], axis=1).dropna()

    y = aligned['si_returns']
    X = sm.add_constant(aligned.drop('si_returns', axis=1))

    # Fit model
    if use_hac:
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    else:
        model = sm.OLS(y, X).fit()

    # Extract results
    results = {
        'alpha': model.params['const'],
        'alpha_tstat': model.tvalues['const'],
        'alpha_pval': model.pvalues['const'],
        'alpha_significant': model.pvalues['const'] < 0.05,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'factor_betas': model.params.drop('const').to_dict(),
        'factor_tstats': model.tvalues.drop('const').to_dict(),
        'residual_std': model.resid.std(),
        'n_obs': len(y),
    }

    # Compute Information Ratio
    annual_alpha = results['alpha'] * 252  # Annualize daily alpha
    annual_resid_vol = results['residual_std'] * np.sqrt(252)
    results['information_ratio'] = annual_alpha / annual_resid_vol

    return results


def construct_factors(data: pd.DataFrame, market_type: str) -> pd.DataFrame:
    """
    Construct factor returns for regression.

    Factors:
    1. Market: Buy-and-hold return
    2. Momentum: 12-day cumulative return (or 252-day for monthly)
    3. Volatility: Inverse realized volatility signal
    4. Trend: ADX-based trend factor
    """

    factors = pd.DataFrame(index=data.index)

    returns = data['close'].pct_change()

    # Factor 1: Market
    factors['market'] = returns

    # Factor 2: Momentum (sign of past 20-day return)
    mom_signal = np.sign(data['close'].pct_change(20))
    factors['momentum'] = mom_signal.shift(1) * returns  # Long if positive momentum

    # Factor 3: Volatility timing (inverse vol)
    rolling_vol = returns.rolling(20).std()
    vol_signal = 1 / rolling_vol.clip(lower=0.001)  # Inverse vol
    vol_signal = vol_signal / vol_signal.rolling(60).mean()  # Normalize
    factors['vol_timing'] = vol_signal.shift(1) * returns

    # Factor 4: Trend (simplified ADX proxy)
    high_low_range = (data['high'] - data['low']).rolling(14).mean()
    trend_strength = high_low_range / data['close'].rolling(14).mean()
    trend_signal = np.sign(data['close'].pct_change(7)) * trend_strength
    factors['trend'] = trend_signal.shift(1) * returns

    return factors.dropna()


def generate_factor_table(results_by_asset: Dict[str, Dict]) -> pd.DataFrame:
    """
    Generate publication-ready factor regression table.

    Output format:
                        (1)         (2)         (3)
    Alpha              0.15***     0.12**      0.11**
                       (3.42)      (2.65)      (2.31)
    Market                         0.08**      0.07*
    Momentum                                   0.05
    R²                  0.03        0.04        0.05
    """

    rows = []
    for asset, res in results_by_asset.items():
        rows.append({
            'asset': asset,
            'alpha': res['alpha'],
            'alpha_tstat': res['alpha_tstat'],
            'alpha_sig': '***' if res['alpha_pval'] < 0.01 else ('**' if res['alpha_pval'] < 0.05 else ('*' if res['alpha_pval'] < 0.10 else '')),
            'r_squared': res['r_squared'],
            'ir': res['information_ratio'],
            **{f'beta_{k}': v for k, v in res['factor_betas'].items()}
        })

    return pd.DataFrame(rows)
```

## Required Output Table

```
Table X: Factor-Adjusted SI Returns

Panel A: Univariate (SI alone)
───────────────────────────────────────
Asset          SI Corr    t-stat
BTC            0.18       3.42***
ETH            0.15       2.89***
SPY            0.12       2.31**
...

Panel B: Multivariate (SI + Factors)
───────────────────────────────────────
                 (1)        (2)        (3)
                SI Only   +Market   +All Factors
Alpha           0.15***    0.12**     0.10**
                (3.42)     (2.65)     (2.15)
Market                     0.85***    0.82***
Momentum                              0.05
Vol Timing                           -0.03
Trend                                 0.08*
R²              0.03       0.42       0.45

Notes: t-statistics in parentheses.
*** p<0.01, ** p<0.05, * p<0.10
HAC standard errors with 5 lags.
```

## Success Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Alpha t-stat | > 2.0 | SI has unexplained return |
| Alpha after all factors | > 0 | SI adds value beyond factors |
| R² of factor model | < 0.5 | SI is not fully explained by factors |
| Information Ratio | > 0.3 | Risk-adjusted alpha is meaningful |

## Deliverables

- [ ] `experiments/factor_regression.py`
- [ ] Factor-adjusted alpha table (Panel A & B format)
- [ ] Information Ratio by asset
- [ ] Conclusion: SI is/is not a novel signal

---

# P2: REGIME-CONDITIONAL SI

**Timeline**: 2-3 days
**Dependencies**: P1 completed
**Gate**: Sign flip rate < 15%

## Objective
Implement SI signal usage that adapts to detected market regime, reducing sign flip issues.

## 2.1 Regime-SI Mapping

| Regime | SI Interpretation | Action |
|--------|------------------|--------|
| **Trending** (ADX > 25) | High SI = strong trend | Trade with trend |
| **Mean-reverting** (ADX < 20) | High SI = clear range | Trade mean-reversion |
| **Volatile** (Vol > 2σ) | SI unreliable | Reduce exposure |
| **Transition** (regime change) | SI lagging | Wait for confirmation |

## 2.2 Look-Ahead Bias Audit (NEW)

**Critical Check**: Regime classification must use only past data.

```python
# AUDIT: Verify no look-ahead in regime classification

def audit_regime_classification():
    """
    Verify that regime classification at time t uses only data up to t.
    """

    # Test case: Classify regime at t=100
    t = 100
    lookback = 7

    # Correct: uses data[t-lookback:t]
    window_correct = data.iloc[t-lookback:t]

    # Check: Does the window include any data from t or later?
    assert window_correct.index[-1] < data.index[t], "Look-ahead detected!"

    print("✅ No look-ahead bias in regime classification")
```

## 2.3 Regime Persistence Check (NEW)

**Requirement**: Average regime duration > 5 days for tradability.

```python
def check_regime_persistence(regimes: pd.Series) -> dict:
    """
    Check that regimes persist long enough to be tradeable.
    """

    # Compute regime durations
    regime_changes = (regimes != regimes.shift()).cumsum()
    durations = regimes.groupby(regime_changes).size()

    avg_duration = durations.mean()
    min_duration = durations.min()
    max_duration = durations.max()

    result = {
        'avg_duration_days': avg_duration,
        'min_duration_days': min_duration,
        'max_duration_days': max_duration,
        'n_regime_changes': len(durations) - 1,
        'tradeable': avg_duration >= 5,
    }

    if not result['tradeable']:
        print(f"⚠️ Regimes too short: avg {avg_duration:.1f} days < 5 day minimum")

    return result
```

## 2.4 Transition Matrix (NEW)

```python
def compute_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """
    Compute regime-to-regime transition probabilities.
    """

    transitions = pd.crosstab(
        regimes.shift(),
        regimes,
        normalize='index'
    )

    return transitions
```

Expected output:
```
              trending  mean_rev  volatile
trending         0.85      0.10      0.05
mean_rev         0.12      0.80      0.08
volatile         0.15      0.25      0.60
```

## 2.5 Code Structure

```python
# src/strategies/regime_conditional_si.py

class RegimeConditionalSI:
    def __init__(self, si_threshold: float = 0.5, min_regime_duration: int = 3):
        self.si_threshold = si_threshold
        self.min_regime_duration = min_regime_duration
        self.regime_detector = RuleBasedRegimeDetector()
        self.current_regime_duration = 0

    def generate_signal(self, data: pd.DataFrame, si: float, idx: int) -> float:
        regime = self.regime_detector.classify(data, idx)

        # Wait for regime to stabilize
        if self._regime_just_changed(regime):
            self.current_regime_duration = 1
            return 0.0  # No signal during transition
        else:
            self.current_regime_duration += 1

        if self.current_regime_duration < self.min_regime_duration:
            return 0.0  # Still in transition

        # Regime-specific logic
        if regime == 'volatile':
            return 0.0  # No signal in volatile regime

        if regime == 'trending':
            if si > self.si_threshold:
                trend_dir = np.sign(data['close'].iloc[idx] - data['close'].iloc[idx-7])
                return trend_dir
            return 0.0

        if regime == 'mean_reverting':
            if si > self.si_threshold:
                z_score = self._compute_zscore(data, idx)
                return -np.sign(z_score) if abs(z_score) > 1.5 else 0.0
            return 0.0

        return 0.0
```

## 2.6 Regime Classification Accuracy (Dr. Rodriguez Addition)

**Objective**: Validate rule-based classifier against statistical methods.

```python
# experiments/validate_regime_classification.py

from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np

def validate_regime_classifier(data: pd.DataFrame) -> dict:
    """
    Compare rule-based regime classification with HMM as reference.
    
    Note: HMM is not "ground truth" but provides independent validation.
    High agreement suggests rule-based method captures similar structure.
    """
    from src.analysis.regime_detection import RuleBasedRegimeDetector, HMMRegimeDetector
    
    # Get classifications
    rule_detector = RuleBasedRegimeDetector(lookback=7)
    hmm_detector = HMMRegimeDetector(n_components=3)
    
    rule_regimes = rule_detector.fit_predict(data)
    hmm_regimes = hmm_detector.fit_predict(data)
    
    # Align (HMM may have different labels)
    # Map HMM labels to closest rule-based labels
    hmm_mapped = map_regime_labels(hmm_regimes, rule_regimes)
    
    # Compute agreement
    agreement = (rule_regimes == hmm_mapped).mean()
    
    # Confusion matrix
    cm = confusion_matrix(rule_regimes, hmm_mapped)
    
    return {
        'agreement_rate': agreement,
        'confusion_matrix': cm,
        'acceptable': agreement > 0.6,  # 60% agreement is reasonable
    }


def map_regime_labels(source: pd.Series, target: pd.Series) -> pd.Series:
    """
    Map source labels to best-matching target labels.
    Uses majority voting within each source label.
    """
    mapping = {}
    for src_label in source.unique():
        mask = source == src_label
        if mask.sum() > 0:
            # Most common target label for this source label
            mapping[src_label] = target[mask].mode().iloc[0]
    
    return source.map(mapping)
```

| Agreement Level | Rate | Interpretation |
|-----------------|------|----------------|
| High | > 80% | Rule-based matches HMM well |
| Moderate | 60-80% | Acceptable, some differences |
| Low | < 60% | Methods capture different structure |

**Note**: Low agreement is not necessarily bad - it may indicate rule-based captures different (possibly better) regime structure.

---

## 2.7 Success Criteria

| Metric | Threshold | Status |
|--------|-----------|--------|
| Sign flip rate | < 15% | ⬜ |
| Avg regime duration | > 5 days | ⬜ |
| Sharpe improvement vs unconditional | > 10% | ⬜ |
| Win rate per regime | > 50% each | ⬜ |
| **Regime classifier agreement** | > 60% (or documented reason) | ⬜ |

## 2.8 Deliverables

- [ ] `src/strategies/regime_conditional_si.py`
- [ ] `experiments/test_regime_conditional.py`
- [ ] Look-ahead bias audit report
- [ ] Regime persistence analysis
- [ ] Transition matrix for each market
- [ ] Flip rate comparison (before/after)

---

# P3: WALK-FORWARD VALIDATION

**Timeline**: 3-4 days
**Dependencies**: P2 completed
**Gate**: >55% profitable windows

## Objective
Validate SI signal with proper walk-forward methodology to ensure no look-ahead bias.

## 3.1 Walk-Forward Design

```
Total Data: 5 years (2021-2026)

Option A: Rolling Window
├── Window 1: Train 2021-2022 (24 mo) → Test 2023-Q1 (3 mo)
├── Window 2: Train 2021Q2-2023Q1 (24 mo) → Test 2023-Q2 (3 mo)
└── ... (rolling forward)

Option B: Expanding Window (RECOMMENDED)
├── Window 1: Train 2021-2022 (24 mo) → Test 2023-Q1 (3 mo)
├── Window 2: Train 2021-2023Q1 (27 mo) → Test 2023-Q2 (3 mo)
├── Window 3: Train 2021-2023Q2 (30 mo) → Test 2023-Q3 (3 mo)
└── ... (expanding)

Total: ~12 out-of-sample periods
```

## 3.2 Parameter Stability Analysis (NEW)

Track how optimal parameters change across windows:

```python
def analyze_parameter_stability(window_results: List[dict]) -> dict:
    """
    Check if optimal parameters are stable across walk-forward windows.
    """

    optimal_si_windows = [r['best_si_window'] for r in window_results]
    optimal_n_agents = [r['best_n_agents'] for r in window_results]

    si_window_stability = np.std(optimal_si_windows) / np.mean(optimal_si_windows)
    n_agents_stability = np.std(optimal_n_agents) / np.mean(optimal_n_agents)

    return {
        'si_window_cv': si_window_stability,  # Coefficient of variation
        'n_agents_cv': n_agents_stability,
        'stable': si_window_stability < 0.3 and n_agents_stability < 0.3,
        'optimal_si_windows': optimal_si_windows,
        'optimal_n_agents': optimal_n_agents,
    }
```

## 3.3 Confidence Intervals (NEW)

```python
def bootstrap_oos_sharpe(oos_returns: pd.Series, n_bootstrap: int = 1000) -> dict:
    """
    Bootstrap confidence intervals for OOS Sharpe ratio.
    """

    sharpes = []
    n = len(oos_returns)

    for _ in range(n_bootstrap):
        sample = oos_returns.sample(n=n, replace=True)
        sharpes.append(sharpe_ratio(sample))

    return {
        'mean_sharpe': np.mean(sharpes),
        'ci_lower': np.percentile(sharpes, 2.5),
        'ci_upper': np.percentile(sharpes, 97.5),
        'prob_positive': np.mean(np.array(sharpes) > 0),
    }
```

## 3.4 Code Structure

```python
# experiments/walk_forward_test.py

class WalkForwardValidator:
    def __init__(self,
                 train_months: int = 24,
                 test_months: int = 3,
                 expanding: bool = True):
        self.train_months = train_months
        self.test_months = test_months
        self.expanding = expanding

    def generate_windows(self, data: pd.DataFrame) -> List[Tuple]:
        windows = []

        if self.expanding:
            start = data.index[0]
            train_end = start + pd.DateOffset(months=self.train_months)

            while True:
                test_end = train_end + pd.DateOffset(months=self.test_months)
                if test_end > data.index[-1]:
                    break

                train = data[start:train_end]  # Expanding from start
                test = data[train_end:test_end]
                windows.append((train, test))

                train_end = test_end  # Move forward

        return windows

    def run(self, data, strategy_class, cost_model) -> pd.DataFrame:
        results = []
        all_oos_returns = []

        for i, (train, test) in enumerate(self.generate_windows(data)):
            # Fit on train
            strategy = strategy_class()
            best_params = strategy.optimize(train)
            strategy.fit(train, **best_params)

            # Test on out-of-sample
            oos_returns = strategy.backtest(test, cost_model)
            all_oos_returns.append(oos_returns)

            # Bootstrap CI for this window
            ci = bootstrap_oos_sharpe(oos_returns)

            results.append({
                'window': i,
                'train_start': train.index[0],
                'train_end': train.index[-1],
                'test_start': test.index[0],
                'test_end': test.index[-1],
                'train_sharpe': sharpe_ratio(strategy.backtest(train, cost_model)),
                'oos_sharpe': sharpe_ratio(oos_returns),
                'oos_sharpe_ci_lower': ci['ci_lower'],
                'oos_sharpe_ci_upper': ci['ci_upper'],
                'oos_return': oos_returns.sum(),
                'oos_max_dd': max_drawdown(oos_returns),
                'best_si_window': best_params.get('si_window'),
                'best_n_agents': best_params.get('n_agents'),
                'profitable': oos_returns.sum() > 0,
            })

        # Aggregate all OOS returns
        combined_oos = pd.concat(all_oos_returns)

        return pd.DataFrame(results), combined_oos
```

## 3.5 Success Criteria

| Metric | Threshold | Status |
|--------|-----------|--------|
| % profitable windows | > 55% | ⬜ |
| Avg OOS Sharpe | > 0.2 | ⬜ |
| OOS Sharpe CI excludes 0 | Required | ⬜ |
| IS vs OOS degradation | < 40% | ⬜ |
| Parameter stability CV | < 0.3 | ⬜ |

## 3.6 Deliverables

- [ ] `experiments/walk_forward_test.py`
- [ ] `results/walk_forward/` with window-by-window results
- [ ] OOS performance chart over time
- [ ] IS vs OOS comparison table
- [ ] Parameter stability analysis
- [ ] Bootstrap confidence intervals

---

# P4: SI AS RISK OVERLAY

**Timeline**: 2 days
**Dependencies**: P1 completed
**Gate**: Drawdown reduction > 15%

## Objective
Use SI as a risk signal to adjust position sizing, independent of alpha generation.

## 4.1 Risk Overlay Logic

```
SI Level → Position Adjustment

SI > 0.7 (High confidence)     → Full position (100%)
SI 0.5-0.7 (Moderate)          → Reduced position (75%)
SI 0.3-0.5 (Low)               → Minimal position (50%)
SI < 0.3 (Very low)            → Cash/hedge (25%)
```

## 4.2 Comparison with Alternatives (NEW)

Compare SI overlay with:

| Method | Description | Benchmark |
|--------|-------------|-----------|
| **Constant sizing** | Always 100% | Baseline |
| **1/Vol sizing** | Scale by inverse volatility | Traditional risk parity |
| **VaR-based** | Size to target VaR | Institutional standard |
| **SI overlay** | Size by SI level | Our method |

## 4.3 Code Structure

```python
# src/strategies/si_risk_overlay.py

class SIRiskOverlay:
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or {
            0.7: 1.0,    # SI > 0.7 → 100% position
            0.5: 0.75,   # SI 0.5-0.7 → 75%
            0.3: 0.50,   # SI 0.3-0.5 → 50%
            0.0: 0.25,   # SI < 0.3 → 25%
        }

    def get_position_multiplier(self, si: float) -> float:
        for threshold, multiplier in sorted(self.thresholds.items(), reverse=True):
            if si >= threshold:
                return multiplier
        return 0.25  # Default minimum


class VolatilityOverlay:
    """Benchmark: inverse volatility sizing."""

    def __init__(self, target_vol: float = 0.15, lookback: int = 20):
        self.target_vol = target_vol
        self.lookback = lookback

    def get_position_multiplier(self, data: pd.DataFrame, idx: int) -> float:
        if idx < self.lookback:
            return 1.0

        recent_vol = data['close'].pct_change().iloc[idx-self.lookback:idx].std() * np.sqrt(252)
        multiplier = self.target_vol / max(recent_vol, 0.05)

        return np.clip(multiplier, 0.25, 2.0)


def compare_overlays(data, base_strategy, si, cost_model):
    """Compare different position sizing methods."""

    base_signals = base_strategy.generate_signals(data)

    overlays = {
        'constant': lambda d, i, s: 1.0,
        'si_overlay': SIRiskOverlay().get_position_multiplier,
        'vol_overlay': VolatilityOverlay().get_position_multiplier,
    }

    results = {}
    for name, overlay_func in overlays.items():
        adjusted_signals = base_signals.copy()

        for i in range(len(data)):
            if name == 'si_overlay':
                mult = overlay_func(si.iloc[i] if i < len(si) else 0.5)
            elif name == 'vol_overlay':
                mult = overlay_func(data, i)
            else:
                mult = overlay_func(data, i, si)

            adjusted_signals.iloc[i] *= mult

        returns = calculate_returns(data, adjusted_signals)
        net_returns = cost_model.apply_costs(returns, adjusted_signals)

        results[name] = {
            'sharpe': sharpe_ratio(net_returns),
            'return': net_returns.sum(),
            'max_drawdown': max_drawdown(net_returns),
            'calmar': net_returns.sum() / abs(max_drawdown(net_returns)),
        }

    return results
```

## 4.4 Success Criteria

| Metric | Threshold | Status |
|--------|-----------|--------|
| Drawdown reduction vs constant | > 15% | ⬜ |
| Sharpe improvement | > 5% | ⬜ |
| SI overlay beats vol overlay | In 2/4 markets | ⬜ |

## 4.5 Deliverables

- [ ] `src/strategies/si_risk_overlay.py`
- [ ] `experiments/test_risk_overlay.py`
- [ ] Comparison table (SI vs vol vs constant)
- [ ] Drawdown charts

---

# P5: ENSEMBLE WITH OTHER SIGNALS

**Timeline**: 4-5 days
**Dependencies**: P1-P3 completed
**Gate**: Sharpe improvement > 15%

## Objective
Build an ensemble that combines SI with traditional signals for more robust performance.

## 5.1 Candidate Signals

| Signal | Type | Expected Correlation with SI |
|--------|------|------------------------------|
| Momentum (12-1 month) | Trend | ~0.3 |
| Mean Reversion (5-day) | Contrarian | ~-0.2 |
| Volatility Timing | Risk | ~-0.4 |
| RSI (14-day) | Technical | ~0.2 |

## 5.2 Ensemble Methods (Enhanced)

### Method 1: Equal Weight (Baseline)
```python
ensemble_signal = (si_signal + momentum + mean_rev + vol_timing) / 4
```

### Method 2: Correlation-Weighted
```python
# Weight by 1 - avg_correlation (diversification benefit)
weights = (1 - correlation_matrix.mean()) / (1 - correlation_matrix.mean()).sum()
ensemble_signal = sum(weights[i] * signals[i] for i in signals)
```

### Method 3: Ridge Regression Meta-Learner (NEW)

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

class RidgeEnsemble:
    def __init__(self, alphas=[0.01, 0.1, 1.0, 10.0]):
        self.model = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=5))

    def fit(self, signal_matrix: pd.DataFrame, returns: pd.Series):
        """
        Fit ridge regression on training data.
        signal_matrix: columns are individual signals
        returns: target returns to predict
        """
        # Use time-series cross-validation
        self.model.fit(signal_matrix, returns)
        self.weights = self.model.coef_
        return self

    def predict(self, signal_matrix: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(signal_matrix), index=signal_matrix.index)
```

### Method 4: Dynamic Correlation Weighting (NEW)

```python
class DynamicCorrelationEnsemble:
    def __init__(self, correlation_window: int = 60):
        self.window = correlation_window

    def get_weights(self, signal_matrix: pd.DataFrame, idx: int) -> np.array:
        """
        Compute weights based on recent correlations.
        """
        if idx < self.window:
            # Equal weight if not enough history
            return np.ones(len(signal_matrix.columns)) / len(signal_matrix.columns)

        recent = signal_matrix.iloc[idx-self.window:idx]
        corr_matrix = recent.corr()

        # Weight by inverse of average correlation (for diversification)
        avg_corr = corr_matrix.mean()
        weights = (1 - avg_corr) / (1 - avg_corr).sum()

        return weights.values
```

## 5.3 Audit: Ensemble Weights Must Be OOS (NEW)

**Critical**: Ensemble weights fitted on training data, tested on test data.

```python
def train_test_ensemble(data, signals, returns):
    """
    Properly split data for ensemble training.
    """

    # Split: 70% train, 30% test
    split_idx = int(len(data) * 0.7)

    train_signals = signals.iloc[:split_idx]
    train_returns = returns.iloc[:split_idx]

    test_signals = signals.iloc[split_idx:]
    test_returns = returns.iloc[split_idx:]

    # Fit on train
    ensemble = RidgeEnsemble()
    ensemble.fit(train_signals, train_returns)

    # Predict on test
    test_predictions = ensemble.predict(test_signals)

    # Evaluate
    oos_corr = test_predictions.corr(test_returns)

    return {
        'weights': ensemble.weights,
        'train_corr': train_signals @ ensemble.weights.corr(train_returns),
        'test_corr': oos_corr,
        'degradation': 1 - (oos_corr / train_corr),
    }
```

## 5.4 SI-Only Baseline Requirement (Dr. Rodriguez Addition)

**Critical**: Always include SI-only as baseline to quantify ensemble benefit.

```python
def compare_with_baseline(data, si, other_signals, cost_model):
    """
    Compare ensemble performance against SI-only baseline.
    
    This is REQUIRED to demonstrate ensemble adds value.
    """
    
    # Baseline: SI signal alone
    si_only_returns = backtest_signal(data, si, cost_model)
    si_only_sharpe = sharpe_ratio(si_only_returns)
    
    # Ensemble variants
    results = {
        'si_only': {
            'sharpe': si_only_sharpe,
            'return': si_only_returns.sum(),
            'max_dd': max_drawdown(si_only_returns),
        }
    }
    
    for ensemble_name, ensemble_signal in [
        ('equal_weight', create_equal_weight_ensemble(si, other_signals)),
        ('correlation_weight', create_corr_weight_ensemble(si, other_signals)),
        ('ridge', create_ridge_ensemble(si, other_signals)),
    ]:
        ens_returns = backtest_signal(data, ensemble_signal, cost_model)
        ens_sharpe = sharpe_ratio(ens_returns)
        
        results[ensemble_name] = {
            'sharpe': ens_sharpe,
            'return': ens_returns.sum(),
            'max_dd': max_drawdown(ens_returns),
            'improvement_vs_si': (ens_sharpe - si_only_sharpe) / si_only_sharpe * 100,
        }
    
    return results
```

**Required Output Table**:
```
Ensemble Comparison vs SI-Only Baseline
───────────────────────────────────────────
Method           Sharpe    vs SI-Only
SI Only          0.45      (baseline)
Equal Weight     0.48      +6.7%
Corr Weight      0.52      +15.6%
Ridge            0.55      +22.2%
```

---

## 5.5 Success Criteria

| Metric | Threshold | Status |
|--------|-----------|--------|
| **SI-only baseline computed** | Required | ⬜ |
| Ensemble Sharpe > SI alone | +15% | ⬜ |
| OOS degradation | < 30% | ⬜ |
| Ensemble correlation with SI | < 0.8 | ⬜ |

## 5.6 Deliverables

- [ ] `src/strategies/si_ensemble.py`
- [ ] `experiments/test_ensemble.py`
- [ ] Signal correlation matrix
- [ ] Ensemble vs individual comparison
- [ ] OOS weight validation

---

# P6: EXTEND TO MORE ASSETS

**Timeline**: 1 week
**Dependencies**: P1-P5 completed
**Gate**: 20+ assets analyzed

## Objective
Strengthen claims by testing on 20+ assets across more markets.

## 6.1 Extended Asset Universe

| Market | Current (11) | Extended (+14) | Total |
|--------|--------------|----------------|-------|
| Crypto | BTC, ETH, SOL | ADA, DOT, AVAX, LINK | 7 |
| Forex | EUR, GBP, JPY | AUD, CAD, CHF | 6 |
| Stocks | SPY, QQQ, AAPL | MSFT, GOOGL, XLF, XLE | 7 |
| Commodities | Gold, Oil | Silver, Copper, Nat Gas | 5 |
| **Total** | 11 | +14 | **25** |

## 6.2 Quality Criteria

| Criterion | Requirement |
|-----------|-------------|
| Liquidity | ADV > $10M for stocks, $100M for crypto |
| History | 5+ years available |
| Data quality | Passes Audit B |

## 6.3 Deliverables

- [ ] Extended data download scripts
- [ ] 25-asset analysis
- [ ] Cross-market consistency table
- [ ] Power analysis update

---

# P7: JOURNAL SUBMISSION

**Timeline**: 2 weeks  
**Dependencies**: P1-P6 completed

## Pre-Submission Checklist (Prof. Chen Addition - Comprehensive)

### Content Checklist

| Item | Status | Notes |
|------|--------|-------|
| arXiv preprint posted | ⬜ | Establishes priority |
| External review (1+ colleague) | ⬜ | Fresh eyes catch errors |
| Trading performance section included | ⬜ | Practical relevance |
| Code/data availability statement | ⬜ | Reproducibility |
| Response to "so what?" prepared | ⬜ | Anticipate reviewers |
| Cover letter drafted | ⬜ | Highlight contribution |

### Formatting Checklist

| Item | Status | Requirement |
|------|--------|-------------|
| Abstract < 250 words | ⬜ | Most venues require |
| All figures 300+ DPI | ⬜ | Publication quality |
| All tables fit margins | ⬜ | No overflow |
| Consistent citation style | ⬜ | Check venue format |
| Page limit respected | ⬜ | NeurIPS: 9 pages |
| Supplementary materials prepared | ⬜ | Appendices, code |

### Technical Checklist

| Item | Status | Verification |
|------|--------|--------------|
| All numbers in tables verified | ⬜ | Re-run scripts |
| Statistical significance reported | ⬜ | p-values, CIs |
| Limitations section included | ⬜ | Honest about constraints |
| Future work section | ⬜ | Research directions |
| Acknowledgments | ⬜ | Funding, help |

### Repository Checklist

| Item | Status | Location |
|------|--------|----------|
| Code is documented | ⬜ | README, docstrings |
| Requirements.txt complete | ⬜ | All dependencies |
| Example scripts work | ⬜ | Test on fresh clone |
| Data manifest included | ⬜ | DATA_MANIFEST.json |
| License file | ⬜ | MIT/Apache/etc |

### Final Review

| Item | Status |
|------|--------|
| Co-author reviewed full draft | ⬜ |
| Spell check passed | ⬜ |
| Grammar check (Grammarly/similar) | ⬜ |
| Read aloud for flow | ⬜ |
| Printed and reviewed on paper | ⬜ |

---

## Target Journals

| Priority | Journal | Timeline | Fit |
|----------|---------|----------|-----|
| 1 | arXiv (preprint) | 1 week | Immediate visibility |
| 2 | NeurIPS | 4-6 months | Top ML venue |
| 3 | Quantitative Finance | 3-6 months | Finance-focused |
| 4 | Journal of Portfolio Management | 3-4 months | Practitioner audience |

---

# SI DEFINITION SENSITIVITY (Prof. Weber Addition)

**Objective**: Verify results are robust to alternative SI definitions.

## Why This Matters

If results only hold for one specific SI formula, they may be fragile. Testing alternatives strengthens claims.

## Alternative SI Definitions

```python
# experiments/si_sensitivity.py

import numpy as np
import pandas as pd

def compute_si_entropy(affinities: np.ndarray) -> float:
    """
    Original SI: 1 - mean(normalized_entropy)
    Higher = more specialized
    """
    entropies = []
    for agent_aff in affinities:
        # Normalize to probabilities
        p = agent_aff / agent_aff.sum()
        # Compute entropy
        entropy = -np.sum(p * np.log(p + 1e-10))
        # Normalize by max entropy
        max_entropy = np.log(len(p))
        entropies.append(entropy / max_entropy)
    
    return 1 - np.mean(entropies)


def compute_si_gini(affinities: np.ndarray) -> float:
    """
    Alternative 1: Gini coefficient of affinities
    Higher = more concentrated (specialized)
    """
    ginis = []
    for agent_aff in affinities:
        n = len(agent_aff)
        sorted_aff = np.sort(agent_aff)
        cumsum = np.cumsum(sorted_aff)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_aff))) / (n * np.sum(sorted_aff)) - (n + 1) / n
        ginis.append(gini)
    
    return np.mean(ginis)


def compute_si_herfindahl(affinities: np.ndarray) -> float:
    """
    Alternative 2: Herfindahl-Hirschman Index
    Sum of squared shares - measures concentration
    """
    hhis = []
    for agent_aff in affinities:
        shares = agent_aff / agent_aff.sum()
        hhi = np.sum(shares ** 2)
        hhis.append(hhi)
    
    return np.mean(hhis)


def compute_si_max_share(affinities: np.ndarray) -> float:
    """
    Alternative 3: Average maximum niche share
    Simple measure of dominant niche
    """
    max_shares = []
    for agent_aff in affinities:
        shares = agent_aff / agent_aff.sum()
        max_shares.append(np.max(shares))
    
    return np.mean(max_shares)


SI_VARIANTS = {
    'entropy': compute_si_entropy,      # Original (default)
    'gini': compute_si_gini,            # Concentration measure
    'herfindahl': compute_si_herfindahl, # HHI-based
    'max_share': compute_si_max_share,   # Dominant niche
}


def run_si_sensitivity_analysis(data, population, features) -> pd.DataFrame:
    """
    Test if key correlations hold across SI definitions.
    """
    results = []
    
    for si_name, si_func in SI_VARIANTS.items():
        # Compute SI with this variant
        si_series = compute_si_timeseries_variant(population, si_func)
        
        # Compute correlations with key features
        for feature in ['adx', 'bb_width', 'rsi', 'volatility_7d']:
            if feature in features.columns:
                corr = si_series.corr(features[feature])
                results.append({
                    'si_variant': si_name,
                    'feature': feature,
                    'correlation': corr,
                })
    
    return pd.DataFrame(results)
```

## Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Key correlations same sign across variants | 100% |
| Correlation magnitude within 50% | 75% of pairs |
| Best SI variant identified | Document choice |

---

# NEGATIVE RESULT PROTOCOL (Prof. Kumar Addition)

**Objective**: Define contingency plans if key gates fail.

## Why This Matters

> "A negative result is still a result. Documenting what DOESN'T work is valuable for the research community." - Prof. Kumar

## Contingency Table

| Gate | Failure Condition | Interpretation | Contingency Action |
|------|-------------------|----------------|-------------------|
| **P1: Net Sharpe < 0** | SI not profitable after costs | SI may not be tradeable signal | Pivot to SI as risk indicator only (P4 focus) |
| **P1: Half-life < 1 day** | Signal decays too fast | Cannot trade at reasonable frequency | Document as characteristic, not failure |
| **P1.5: Alpha t-stat < 2** | SI is known factor | Not novel contribution | Document which factor SI loads on, still publishable |
| **P2: Flip rate > 30%** | Regime conditioning doesn't help | SI interpretation not regime-dependent | Use unconditional SI, simplify model |
| **P3: <40% OOS windows** | Heavy overfitting | Model too complex | Reduce parameters, simpler strategy |
| **P6: <2 markets work** | No cross-market validity | SI is market-specific | Focus on single market, document limitation |
| **All gates fail** | SI has no predictive power | Null result | Publish as "negative result" paper |

## Negative Result Publication Strategy

If results are negative, the paper can still be valuable:

```markdown
Title: "Emergent Specialization Does NOT Predict Market Returns: 
        A Comprehensive Cross-Market Analysis"

Contribution:
1. We rigorously tested whether SI predicts returns
2. We found it does not, after proper controls
3. This saves future researchers from pursuing this path
4. We document WHY it doesn't work (e.g., "SI correlates with X 
   but X doesn't predict returns")
```

## Decision Points

```
                    P1 Gate
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
      Net Sharpe > 0          Net Sharpe ≤ 0
           │                       │
           ▼                       ▼
    Continue to P1.5         Pivot to P4 (Risk)
           │                       │
           ▼                       ▼
      Alpha t > 2            Test SI as Risk
           │                    Overlay Only
    ┌──────┴──────┐                │
    ▼             ▼                ▼
  Novel       Known Factor    If works: Publish
  Signal      (document)      as risk paper
    │             │
    ▼             ▼
Continue      Still publish
to P2+        (negative result)
```

---

# PHASE 2: DEFERRED ITEMS

These items are deferred to Phase 2 based on expert recommendations:

| Item | Original Step | Reason for Deferral |
|------|---------------|---------------------|
| Higher frequency testing | Step 5 | Microstructure noise, different cost structure |
| Full macro features | Step 6 | Start with VIX + yield curve only |
| Cross-asset SI | Step 8 | Scope creep, different research question |

---

# EXECUTION TIMELINE

| Week | Priority | Steps | Gate |
|------|----------|-------|------|
| **1 (Days 1-2)** | P0 | Critical Audits A-E | All pass |
| **1 (Days 3-5)** | P1 | Backtest with costs + Alpha decay | Net Sharpe > 0, Half-life ≥ 3 days |
| **1 (Day 5)** | P1.5 | Factor Regression | Alpha t-stat > 2.0 |
| **2 (Days 1-3)** | P2 | Regime-conditional | Flip rate < 15% |
| **2 (Days 4-5)** | P4 | Risk overlay | DD reduction > 15% |
| **3** | P3 | Walk-forward | >55% profitable |
| **4** | P5 | Ensemble | Sharpe +15% |
| **5** | P6 | More assets | 20+ assets |
| **6** | P7 | Paper submission | Submitted |

---

# GLOBAL SUCCESS METRICS

| Metric | Threshold | Priority | Source |
|--------|-----------|----------|--------|
| Net Sharpe after costs | > 0.3 | Critical | P1 |
| **Alpha half-life** | ≥ 3 days | Critical | P1 (Prof. Chen) |
| **Annual turnover** | < 200% | Critical | P1 (Dr. Chen HF) |
| **Factor-adjusted alpha t-stat** | > 2.0 | Critical | P1.5 (Prof. Kumar) |
| Cross-market replication | 3/4 markets | Critical | P1 |
| Walk-forward hit rate | > 55% | High | P3 |
| Flip rate (regime-conditional) | < 15% | High | P2 |
| **Regime classifier agreement** | > 60% | High | P2 (Dr. Rodriguez) |
| Drawdown reduction (overlay) | > 15% | Medium | P4 |
| **SI-only baseline computed** | Required | Medium | P5 (Dr. Rodriguez) |
| Ensemble Sharpe improvement | > 15% | Medium | P5 |
| **SI definition sensitivity** | Same sign across variants | Medium | (Prof. Weber) |
| **All parameters documented** | 100% | Required | All (Prof. Weber) |
| **Data manifest created** | Required | Required | P0 (Prof. Weber) |
| **Pre-submission checklist complete** | 100% | Required | P7 (Prof. Chen) |

---

# AUDIT LOG

| Audit | Date | Status | Notes |
|-------|------|--------|-------|
| A: Survivorship | - | ⬜ Pending | |
| B: Data Quality | - | ⬜ Pending | |
| C: Reproducibility | - | ⬜ Pending | |
| C.1: Data Versioning | - | ⬜ Pending | Prof. Weber v4 |
| D: Look-Ahead | - | ⬜ Pending | |
| E: Economic Significance | - | ⬜ Pending | |
| **P1: Turnover Analysis** | - | ⬜ Pending | Dr. Chen (HF) v4 |
| **P1: Alpha Decay** | - | ⬜ Pending | Prof. Chen v3 |
| **P1.5: Factor Regression** | - | ⬜ Pending | Prof. Kumar v3 |
| **P2: Regime Accuracy** | - | ⬜ Pending | Dr. Rodriguez v4 |
| **P5: SI-Only Baseline** | - | ⬜ Pending | Dr. Rodriguez v4 |
| **SI Sensitivity** | - | ⬜ Pending | Prof. Weber v4 |
| **All: Parameter Docs** | - | ⬜ Pending | Prof. Weber v3 |
| **P7: Pre-submission** | - | ⬜ Pending | Prof. Chen v4 |

---

# VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| v1 | Jan 17, 2026 | Initial plan from expert recommendations |
| v2 | Jan 17, 2026 | Added P0 audits, enhanced cost model, regime analysis |
| v3 | Jan 17, 2026 | Added P1.5 Factor Regression, Alpha Decay, Parameter Documentation |
| **v4** | Jan 17, 2026 | **Final comprehensive version**: Added turnover analysis, regime classification accuracy, SI-only baseline, pre-submission checklist, SI definition sensitivity, negative result protocol, computational cost tracking, data versioning |

---

*Plan created: January 17, 2026*  
*Updated: January 17, 2026 (v4 - Final Comprehensive Review)*  
*Author: Yuhao Li, University of Pennsylvania*

---

# APPROVAL SIGNATURES

| Reviewer | Role | Approval | Date |
|----------|------|----------|------|
| Prof. S. Kumar | MIT Sloan | ✅ Approved | Jan 17, 2026 |
| Prof. R. Chen | Chicago Booth | ✅ Approved | Jan 17, 2026 |
| Prof. M. Weber | Wharton | ✅ Approved | Jan 17, 2026 |
| Prof. A. Pedersen | NYU Stern | ✅ Approved | Jan 17, 2026 |
| Dr. A. Chen | Quant PM, HF | ✅ Approved | Jan 17, 2026 |
| Dr. M. Rodriguez | Head of Quant Research | ✅ Approved | Jan 17, 2026 |
| J. Williams | Execution Trader | ✅ Approved | Jan 17, 2026 |

**Status: READY FOR EXECUTION**
