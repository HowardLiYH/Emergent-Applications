# SI Research: Next Steps Implementation Plan

**Date**: January 17, 2026  
**Based on**: Expert Panel Review  
**Author**: Yuhao Li, University of Pennsylvania

---

## Overview

This document outlines detailed implementation plans for the 10 expert-recommended next steps, organized by priority.

---

# HIGH PRIORITY (Before Production)

## Step 1: Backtest SI-Based Strategy with Realistic Costs

### Objective
Validate that SI-based trading signals remain profitable after accounting for realistic transaction costs.

### Implementation Plan

```
Timeline: 3-5 days
Dependencies: Existing SI strategy code, transaction cost research
Output: Cost-adjusted performance metrics
```

#### 1.1 Transaction Cost Model

| Market | Cost Type | Value | Source |
|--------|-----------|-------|--------|
| Crypto | Taker fee | 0.04% (4 bps) | Binance VIP |
| Crypto | Slippage | 0.02% (2 bps) | Estimated |
| Forex | Spread | 0.01% (1 bp) | Major pairs |
| Stocks | Commission + spread | 0.02% (2 bps) | ETF average |
| Commodities | Futures spread | 0.03% (3 bps) | Estimated |

#### 1.2 Strategy Variants to Test

| Variant | Description | Expected Turnover |
|---------|-------------|-------------------|
| SI Threshold Long | Long when SI > 0.6 | ~50 trades/year |
| SI Momentum | Long when dSI/dt > 0 | ~100 trades/year |
| SI Regime Switch | Different strategy per SI level | ~30 trades/year |
| SI Risk Overlay | Reduce position when SI < 0.4 | ~20 adjustments/year |

#### 1.3 Code Structure

```python
# experiments/backtest_with_costs.py

class CostModel:
    def __init__(self, market_type: str):
        self.costs = TRANSACTION_COSTS[market_type]
    
    def apply_costs(self, returns: pd.Series, trades: pd.Series) -> pd.Series:
        """Apply round-trip costs to returns."""
        cost_per_trade = self.costs['fee'] + self.costs['slippage']
        return returns - (trades.abs() * cost_per_trade * 2)  # Round-trip

def run_backtest_with_costs(data, si, strategy, cost_model):
    signals = strategy.generate_signals(data, si)
    trades = signals.diff().fillna(0)
    gross_returns = calculate_returns(data, signals)
    net_returns = cost_model.apply_costs(gross_returns, trades)
    return {
        'gross_sharpe': sharpe_ratio(gross_returns),
        'net_sharpe': sharpe_ratio(net_returns),
        'turnover': trades.abs().sum(),
        'cost_drag': (gross_returns.sum() - net_returns.sum())
    }
```

#### 1.4 Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Net Sharpe > 0.3 | Must be positive after costs | Minimum viability |
| Cost drag < 50% of gross | Costs shouldn't dominate | Practical tradability |
| Consistent across markets | 3/4 markets profitable | Robustness |

#### 1.5 Deliverables

- [ ] `experiments/backtest_with_costs.py`
- [ ] `results/cost_analysis/` directory with results
- [ ] Performance comparison table (gross vs net)
- [ ] Sensitivity analysis: cost ±50%

---

## Step 2: Regime-Conditional SI Usage

### Objective
Implement SI signal usage that adapts to detected market regime, reducing sign flip issues.

### Implementation Plan

```
Timeline: 2-3 days
Dependencies: Regime detection module, SI computation
Output: Regime-conditional strategy with reduced flip rate
```

#### 2.1 Regime-SI Mapping

Based on our analysis, different regimes require different SI interpretation:

| Regime | SI Interpretation | Action |
|--------|------------------|--------|
| **Trending** (ADX > 25) | High SI = strong trend | Trade with trend |
| **Mean-reverting** (ADX < 20) | High SI = clear range | Trade mean-reversion |
| **Volatile** (Vol > 2σ) | SI unreliable | Reduce exposure |
| **Transition** (regime change) | SI lagging | Wait for confirmation |

#### 2.2 Code Structure

```python
# src/strategies/regime_conditional_si.py

class RegimeConditionalSI:
    def __init__(self, si_threshold: float = 0.5):
        self.si_threshold = si_threshold
        self.regime_detector = RuleBasedRegimeDetector()
    
    def generate_signal(self, data: pd.DataFrame, si: float, idx: int) -> float:
        regime = self.regime_detector.classify(data, idx)
        
        if regime == 'volatile':
            return 0.0  # No signal in volatile regime
        
        if regime == 'trending':
            if si > self.si_threshold:
                trend_dir = np.sign(data['close'].iloc[idx] - data['close'].iloc[idx-7])
                return trend_dir  # Follow trend when SI high
            return 0.0
        
        if regime == 'mean_reverting':
            if si > self.si_threshold:
                z_score = self._compute_zscore(data, idx)
                return -np.sign(z_score)  # Mean revert when SI high
            return 0.0
        
        return 0.0  # Default: no signal
    
    def _compute_zscore(self, data, idx, lookback=20):
        window = data['close'].iloc[idx-lookback:idx]
        return (data['close'].iloc[idx] - window.mean()) / window.std()
```

#### 2.3 Validation Tests

| Test | Method | Success Criterion |
|------|--------|-------------------|
| Flip rate reduction | Compare conditional vs unconditional | < 10% flip rate |
| Sharpe improvement | Compare Sharpe ratios | > 10% improvement |
| Drawdown reduction | Compare max drawdowns | < 20% reduction |
| Win rate by regime | Segment performance | > 50% in each regime |

#### 2.4 Deliverables

- [ ] `src/strategies/regime_conditional_si.py`
- [ ] `experiments/test_regime_conditional.py`
- [ ] Regime-performance breakdown table
- [ ] Flip rate comparison (before/after)

---

## Step 3: Out-of-Sample Walk-Forward Test

### Objective
Validate SI signal with proper walk-forward methodology to ensure no look-ahead bias.

### Implementation Plan

```
Timeline: 3-4 days
Dependencies: All SI computation code, backtest framework
Output: Walk-forward performance report
```

#### 3.1 Walk-Forward Design

```
Total Data: 5 years (2021-2026)

Walk-Forward Windows:
├── Window 1: Train 2021-2022 (24 mo) → Test 2023-Q1 (3 mo)
├── Window 2: Train 2021-2023Q1 (27 mo) → Test 2023-Q2 (3 mo)
├── Window 3: Train 2021-2023Q2 (30 mo) → Test 2023-Q3 (3 mo)
├── ... (continue quarterly)
└── Window N: Train 2021-2025Q3 → Test 2025-Q4 (3 mo)

Total: ~12 out-of-sample periods
```

#### 3.2 Refit Protocol

| Component | Refit Frequency | What Changes |
|-----------|-----------------|--------------|
| SI parameters | Quarterly | Window size, n_agents |
| Regime thresholds | Quarterly | ADX, volatility cutoffs |
| Strategy weights | Quarterly | If ensemble |
| Feature selection | Never | Pre-registered features only |

#### 3.3 Code Structure

```python
# experiments/walk_forward_test.py

class WalkForwardValidator:
    def __init__(self, train_months: int = 24, test_months: int = 3):
        self.train_months = train_months
        self.test_months = test_months
    
    def generate_windows(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        windows = []
        start = data.index[0]
        
        while True:
            train_end = start + pd.DateOffset(months=self.train_months)
            test_end = train_end + pd.DateOffset(months=self.test_months)
            
            if test_end > data.index[-1]:
                break
            
            train = data[start:train_end]
            test = data[train_end:test_end]
            windows.append((train, test))
            
            start = start + pd.DateOffset(months=self.test_months)  # Rolling
        
        return windows
    
    def run(self, data, strategy_class) -> pd.DataFrame:
        results = []
        
        for i, (train, test) in enumerate(self.generate_windows(data)):
            # Fit on train
            strategy = strategy_class()
            strategy.fit(train)
            
            # Test on out-of-sample
            oos_returns = strategy.backtest(test)
            
            results.append({
                'window': i,
                'train_start': train.index[0],
                'test_start': test.index[0],
                'test_end': test.index[-1],
                'oos_return': oos_returns.sum(),
                'oos_sharpe': sharpe_ratio(oos_returns),
                'oos_max_dd': max_drawdown(oos_returns)
            })
        
        return pd.DataFrame(results)
```

#### 3.4 Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| % profitable windows | > 60% | More wins than losses |
| Avg OOS Sharpe | > 0.3 | Consistent positive risk-adjusted |
| OOS vs IS degradation | < 30% | Not heavily overfit |
| Worst window drawdown | < 20% | Survivable losses |

#### 3.5 Deliverables

- [ ] `experiments/walk_forward_test.py`
- [ ] `results/walk_forward/` with window-by-window results
- [ ] OOS performance chart over time
- [ ] IS vs OOS comparison table

---

## Step 4: Combine with Other Signals (Ensemble)

### Objective
Build an ensemble that combines SI with traditional signals for more robust performance.

### Implementation Plan

```
Timeline: 4-5 days
Dependencies: SI signal, traditional signal implementations
Output: Ensemble model with improved Sharpe
```

#### 4.1 Candidate Signals to Combine

| Signal | Type | Correlation with SI | Rationale |
|--------|------|---------------------|-----------|
| Momentum (12-1) | Trend | Expected ~0.3 | Classic factor |
| Mean Reversion (5d) | Contrarian | Expected ~-0.2 | Opposite regime |
| Volatility Timing | Risk | Expected ~-0.4 | SI captures vol |
| RSI Divergence | Technical | Expected ~0.2 | Already correlated |
| Volume Breakout | Momentum | Unknown | Liquidity signal |

#### 4.2 Ensemble Methods

| Method | Description | Complexity |
|--------|-------------|------------|
| **Equal Weight** | Average all signals | Low |
| **Correlation Weight** | Weight by 1 - correlation | Medium |
| **Risk Parity** | Weight by inverse volatility | Medium |
| **ML Stacking** | Train meta-model on signals | High |

#### 4.3 Code Structure

```python
# src/strategies/si_ensemble.py

class SIEnsemble:
    def __init__(self, method: str = 'correlation_weight'):
        self.method = method
        self.signals = {
            'si': SISignal(),
            'momentum': MomentumSignal(lookback=252),
            'mean_rev': MeanReversionSignal(lookback=5),
            'vol_timing': VolatilityTimingSignal(),
        }
        self.weights = None
    
    def fit(self, data: pd.DataFrame):
        # Compute signal returns
        signal_returns = {}
        for name, signal in self.signals.items():
            signal_returns[name] = signal.backtest(data)
        
        # Compute weights based on method
        if self.method == 'equal_weight':
            self.weights = {k: 1/len(self.signals) for k in self.signals}
        
        elif self.method == 'correlation_weight':
            corr_matrix = pd.DataFrame(signal_returns).corr()
            avg_corr = corr_matrix.mean()
            self.weights = (1 - avg_corr) / (1 - avg_corr).sum()
        
        elif self.method == 'risk_parity':
            vols = pd.DataFrame(signal_returns).std()
            self.weights = (1 / vols) / (1 / vols).sum()
    
    def generate_signal(self, data, idx) -> float:
        combined = 0
        for name, signal in self.signals.items():
            combined += self.weights[name] * signal.generate(data, idx)
        return np.clip(combined, -1, 1)
```

#### 4.4 Success Criteria

| Metric | Target | vs SI Alone |
|--------|--------|-------------|
| Sharpe Ratio | > 0.5 | +20% improvement |
| Max Drawdown | < 15% | -20% reduction |
| Calmar Ratio | > 0.3 | Improvement |
| Correlation with SI | < 0.7 | Diversification benefit |

#### 4.5 Deliverables

- [ ] `src/strategies/si_ensemble.py`
- [ ] `experiments/test_ensemble.py`
- [ ] Signal correlation matrix
- [ ] Ensemble vs individual signal comparison

---

# MEDIUM PRIORITY (Research Extensions)

## Step 5: Test at Higher Frequencies

### Objective
Investigate whether SI dynamics differ at intraday frequencies.

### Implementation Plan

```
Timeline: 5-7 days
Dependencies: Intraday data access, compute resources
Output: Frequency comparison study
```

#### 5.1 Data Requirements

| Frequency | Period | Data Points | Source |
|-----------|--------|-------------|--------|
| 1-hour | 1 year | ~8,760/asset | Binance (crypto) |
| 15-min | 3 months | ~8,640/asset | Binance |
| 5-min | 1 month | ~8,640/asset | Binance |
| 1-min | 1 week | ~10,080/asset | Binance |

#### 5.2 Hypotheses

| H# | Hypothesis | Test |
|----|------------|------|
| H5.1 | SI converges faster at high frequency | Compare convergence time |
| H5.2 | SI correlates with different features intraday | Repeat discovery pipeline |
| H5.3 | SI has higher noise at high frequency | Compare signal-to-noise |
| H5.4 | Optimal SI window differs by frequency | Sensitivity analysis |

#### 5.3 Adjusted Parameters

| Parameter | Daily | Hourly | 15-min | 5-min |
|-----------|-------|--------|--------|-------|
| SI Window | 7 bars | 168 bars | 672 bars | 2016 bars |
| Regime lookback | 7 bars | 168 bars | 672 bars | 2016 bars |
| Min data for competition | 30 | 720 | 2880 | 8640 |

#### 5.4 Deliverables

- [ ] `experiments/high_frequency_si.py`
- [ ] Intraday data download scripts
- [ ] Frequency comparison table
- [ ] Optimal frequency recommendation

---

## Step 6: Add Macro Features

### Objective
Improve regime detection by incorporating macroeconomic indicators.

### Implementation Plan

```
Timeline: 4-5 days
Dependencies: Macro data sources, regime detection module
Output: Macro-enhanced regime model
```

#### 6.1 Macro Features to Add

| Feature | Frequency | Source | Rationale |
|---------|-----------|--------|-----------|
| VIX | Daily | CBOE | Volatility expectation |
| 10Y-2Y Spread | Daily | FRED | Yield curve / recession |
| DXY (Dollar Index) | Daily | Yahoo | Currency regime |
| Fed Funds Rate | Monthly | FRED | Monetary policy |
| PMI | Monthly | ISM | Economic activity |
| CPI YoY | Monthly | BLS | Inflation regime |

#### 6.2 Integration Approach

```python
# src/analysis/macro_regime.py

class MacroRegimeDetector:
    def __init__(self):
        self.macro_data = self._load_macro_data()
    
    def _load_macro_data(self):
        return {
            'vix': yf.download('^VIX'),
            'yield_spread': self._get_yield_spread(),
            'dxy': yf.download('DX-Y.NYB'),
        }
    
    def get_macro_regime(self, date) -> str:
        vix = self.macro_data['vix'].loc[:date, 'Close'].iloc[-1]
        spread = self.macro_data['yield_spread'].loc[:date].iloc[-1]
        
        if vix > 30:
            return 'crisis'
        elif spread < 0:
            return 'recession_warning'
        elif vix < 15:
            return 'calm'
        else:
            return 'normal'
    
    def enhance_regime(self, market_regime: str, date) -> str:
        macro = self.get_macro_regime(date)
        
        # Override market regime in crisis
        if macro == 'crisis':
            return 'volatile'
        
        return market_regime
```

#### 6.3 Deliverables

- [ ] `src/analysis/macro_regime.py`
- [ ] Macro data download scripts
- [ ] Comparison: market-only vs macro-enhanced regimes
- [ ] SI correlation analysis with macro regimes

---

## Step 7: SI as Risk Overlay

### Objective
Use SI as a risk signal to adjust position sizing, independent of alpha generation.

### Implementation Plan

```
Timeline: 3-4 days
Dependencies: SI computation, portfolio framework
Output: Risk overlay implementation
```

#### 7.1 Risk Overlay Logic

```
SI Level → Risk Adjustment

SI > 0.7 (High specialization)   → Full position (100%)
SI 0.5-0.7 (Moderate)            → Reduced position (75%)
SI 0.3-0.5 (Low)                 → Minimal position (50%)
SI < 0.3 (Very low)              → Cash/hedge (25%)
```

#### 7.2 Code Structure

```python
# src/strategies/si_risk_overlay.py

class SIRiskOverlay:
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or {
            'high': (0.7, 1.0),    # 100% position
            'moderate': (0.5, 0.75),
            'low': (0.3, 0.5),
            'very_low': (0.0, 0.25)
        }
    
    def get_position_multiplier(self, si: float) -> float:
        if si > 0.7:
            return 1.0
        elif si > 0.5:
            return 0.75
        elif si > 0.3:
            return 0.5
        else:
            return 0.25
    
    def apply_overlay(self, base_signal: float, si: float) -> float:
        multiplier = self.get_position_multiplier(si)
        return base_signal * multiplier
```

#### 7.3 Backtest Scenarios

| Scenario | Base Strategy | With SI Overlay | Comparison |
|----------|--------------|-----------------|------------|
| Buy & Hold | Full exposure | SI-adjusted | Drawdown reduction |
| Momentum | Trend following | SI-adjusted | Risk-adjusted return |
| Mean Reversion | Contrarian | SI-adjusted | Win rate improvement |

#### 7.4 Deliverables

- [ ] `src/strategies/si_risk_overlay.py`
- [ ] `experiments/test_risk_overlay.py`
- [ ] Drawdown comparison charts
- [ ] Risk-adjusted metrics with/without overlay

---

## Step 8: Cross-Asset SI

### Objective
Compute SI across multiple asset classes jointly for portfolio-level insights.

### Implementation Plan

```
Timeline: 5-7 days
Dependencies: Multi-asset data, extended NichePopulation
Output: Cross-asset SI implementation
```

#### 8.1 Design

```
Traditional SI: Agents compete on ONE asset, specialize by regime
Cross-Asset SI: Agents compete on MULTIPLE assets, specialize by asset

Niches = {BTC, ETH, SPY, EURUSD, Gold}
Each agent develops affinity for specific assets
SI measures cross-asset specialization
```

#### 8.2 Extended NichePopulation

```python
# src/competition/cross_asset_population.py

class CrossAssetNichePopulation:
    def __init__(self, assets: List[str], n_agents: int = 20):
        self.assets = assets
        self.n_niches = len(assets)  # Niches = assets
        self.agents = [Agent(n_niches) for _ in range(n_agents)]
    
    def run(self, multi_data: Dict[str, pd.DataFrame]):
        """
        multi_data: {'BTC': df, 'ETH': df, 'SPY': df, ...}
        """
        # Align all data to common dates
        common_dates = self._get_common_dates(multi_data)
        
        for date in common_dates:
            # Each agent generates signals for all assets
            agent_returns = []
            
            for agent in self.agents:
                total_return = 0
                for i, asset in enumerate(self.assets):
                    signal = agent.strategy.generate(multi_data[asset], date)
                    ret = self._get_return(multi_data[asset], date, signal)
                    total_return += agent.asset_weights[i] * ret
                agent_returns.append(total_return)
            
            # Winner specializes in best-performing asset today
            best_asset_idx = self._get_best_asset(multi_data, date)
            winner_idx = np.argmax(agent_returns)
            
            self.agents[winner_idx].update_affinity(best_asset_idx)
    
    def compute_cross_asset_si(self) -> float:
        """SI measuring specialization across assets."""
        return 1 - np.mean([agent.entropy() for agent in self.agents])
```

#### 8.3 Research Questions

| Question | Hypothesis |
|----------|------------|
| Does cross-asset SI predict sector rotation? | High SI → concentrated opportunities |
| Does low cross-asset SI predict correlation spikes? | Low SI → assets behaving similarly |
| Is cross-asset SI regime-dependent? | Different patterns in risk-on vs risk-off |

#### 8.4 Deliverables

- [ ] `src/competition/cross_asset_population.py`
- [ ] `experiments/cross_asset_si.py`
- [ ] Cross-asset SI time series
- [ ] Correlation with asset correlations

---

# PUBLICATION TRACK

## Step 9: Submit to Quantitative Finance Journal

### Objective
Prepare and submit research paper to peer-reviewed journal.

### Implementation Plan

```
Timeline: 2-3 weeks for preparation, 3-6 months review
Dependencies: All high-priority steps complete
Output: Submitted manuscript
```

#### 9.1 Target Journals

| Journal | Impact Factor | Fit | Timeline |
|---------|---------------|-----|----------|
| **Journal of Financial Economics** | 8.2 | High (if alpha proven) | 6-12 months |
| **Quantitative Finance** | 2.1 | High | 3-6 months |
| **Journal of Portfolio Management** | 1.8 | Good | 3-4 months |
| **Journal of Trading** | 0.8 | Good | 2-3 months |
| **arXiv** (preprint) | N/A | Immediate visibility | 1 week |

#### 9.2 Paper Structure

```
1. Abstract (250 words)
2. Introduction
   - Motivation
   - Research question
   - Contributions
3. Related Work
   - Emergent specialization
   - Regime detection
   - Trading signals
4. Methodology
   - SI definition
   - Competition mechanism
   - Statistical framework
5. Data
6. Results
   - Discovery findings
   - Cross-market validation
   - Regime analysis
7. Discussion
8. Conclusion
9. References
```

#### 9.3 Deliverables

- [ ] Extended LaTeX paper (20+ pages)
- [ ] Supplementary materials
- [ ] Cover letter
- [ ] Submission to arXiv (preprint)
- [ ] Submission to target journal

---

## Step 10: Extend to More Assets

### Objective
Strengthen claims by testing on 20+ assets across more markets.

### Implementation Plan

```
Timeline: 1-2 weeks
Dependencies: Data access, compute resources
Output: Extended cross-asset validation
```

#### 10.1 Extended Asset Universe

| Market | Current | Extended |
|--------|---------|----------|
| **Crypto** | BTC, ETH, SOL | +ADA, DOT, AVAX, LINK, MATIC |
| **Forex** | EUR/USD, GBP/USD, USD/JPY | +AUD/USD, USD/CAD, NZD/USD, EUR/GBP |
| **Stocks** | SPY, QQQ, AAPL | +MSFT, GOOGL, AMZN, TSLA, XLF, XLE, XLK |
| **Commodities** | Gold, Oil | +Silver, Copper, Natural Gas, Corn |
| **Bonds** | (none) | TLT, IEF, HYG, LQD |

**Total: 30+ assets**

#### 10.2 Statistical Power Analysis

| # Assets | Power for r=0.15 | Power for r=0.20 |
|----------|------------------|------------------|
| 11 (current) | 0.72 | 0.88 |
| 20 | 0.89 | 0.97 |
| 30 | 0.96 | 0.99 |

#### 10.3 Deliverables

- [ ] Extended data download scripts
- [ ] Full 30-asset analysis
- [ ] Updated cross-market tables
- [ ] Power analysis documentation

---

## Timeline Summary

| Week | Steps | Focus |
|------|-------|-------|
| 1 | #1, #2 | Cost validation, regime-conditional |
| 2 | #3, #4 | Walk-forward, ensemble |
| 3 | #5, #6 | High-frequency, macro features |
| 4 | #7, #8 | Risk overlay, cross-asset |
| 5-6 | #9, #10 | Paper prep, extended assets |

---

## Success Metrics

| Step | Success Metric | Threshold |
|------|---------------|-----------|
| #1 | Net Sharpe | > 0.3 |
| #2 | Flip rate | < 10% |
| #3 | % profitable windows | > 60% |
| #4 | Ensemble Sharpe improvement | > 20% |
| #5 | Frequency insight | Actionable finding |
| #6 | Regime accuracy | > 70% |
| #7 | Drawdown reduction | > 20% |
| #8 | Cross-asset insight | Novel finding |
| #9 | Paper acceptance | Submitted |
| #10 | Asset coverage | 30+ assets |

---

*Plan created: January 17, 2026*  
*Author: Yuhao Li, University of Pennsylvania*
