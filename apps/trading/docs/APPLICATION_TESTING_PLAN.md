# SI Application Testing Plan

**Date:** January 18, 2026  
**Author:** Yuhao Li, University of Pennsylvania  
**Goal:** Test all 10 SI applications and select the best for thesis

---

## 📋 Overview

| # | Application | Category | Complexity | Priority |
|---|-------------|----------|------------|----------|
| 1 | SI Risk Budgeting | Position Sizing | Low | ⭐⭐⭐ |
| 2 | SI-ADX Spread Trading | Mean Reversion | Medium | ⭐⭐⭐ |
| 3 | Factor Timing | Strategy Selection | Medium | ⭐⭐⭐ |
| 4 | Volatility Forecasting | Risk Mgmt | Low | ⭐⭐ |
| 5 | Dynamic Stop-Loss | Trade Mgmt | Low | ⭐⭐ |
| 6 | Regime Rebalancing | Portfolio Mgmt | Medium | ⭐⭐ |
| 7 | Tail Risk Hedge | Risk Mgmt | Medium | ⭐ |
| 8 | Cross-Asset Momentum | Pairs Trading | High | ⭐ |
| 9 | Ensemble Strategy | Multi-Signal | High | ⭐⭐ |
| 10 | Entry Timing | Trade Timing | Low | ⭐⭐ |

---

## 🧪 Testing Framework

### Universal Test Parameters

```python
ASSETS = {
    'crypto': ['BTCUSDT', 'ETHUSDT'],
    'stocks': ['SPY', 'QQQ'],
    'forex': ['EURUSD', 'GBPUSD'],
}

TRANSACTION_COSTS = {
    'crypto': 0.0004,    # 4 bps
    'stocks': 0.0002,    # 2 bps
    'forex': 0.0001,     # 1 bp
}

SPLIT = {
    'train': 0.60,       # 60% train
    'val': 0.20,         # 20% validation
    'test': 0.20,        # 20% holdout test
}

SI_WINDOWS = [7, 14, 21]  # Days
```

### Universal Metrics

| Metric | Formula | Threshold for Success |
|--------|---------|----------------------|
| **Sharpe Ratio** | mean(r) / std(r) × √252 | > 0.5 |
| **Win Rate** | % profitable periods | > 55% |
| **Max Drawdown** | Max peak-to-trough | < 30% |
| **Calmar Ratio** | Ann. Return / Max DD | > 0.5 |
| **Improvement vs Baseline** | (App - Baseline) / Baseline | > 10% |
| **Cross-Asset Consistency** | % assets where App works | > 50% |

---

## 📊 Application 1: SI Risk Budgeting

### Hypothesis
Position size based on SI percentile improves risk-adjusted returns.

### Test Design

```
BASELINE: Constant position (100%)
TEST: Position = f(SI_percentile)

Variants to test:
A. Linear: pos = 0.5 + 0.5 × SI_rank
B. Quantile: pos = {0.5, 0.75, 1.0, 1.25, 1.5} by SI quintile
C. Threshold: pos = 0.5 if SI < p25, 1.5 if SI > p75, else 1.0
```

### Implementation

```python
def test_risk_budgeting(data, si, variant='linear'):
    returns = data['close'].pct_change()
    si_rank = si.rank(pct=True)
    
    if variant == 'linear':
        position = 0.5 + si_rank * 1.0
    elif variant == 'quantile':
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        mults = [0.5, 0.75, 1.0, 1.25, 1.5]
        position = pd.cut(si_rank, bins, labels=mults).astype(float)
    elif variant == 'threshold':
        position = np.where(si_rank < 0.25, 0.5,
                   np.where(si_rank > 0.75, 1.5, 1.0))
    
    strategy_returns = position.shift(1) * returns
    return apply_costs(strategy_returns, position, cost_rate)
```

### Success Criteria
- [ ] Sharpe > baseline in 4/6 assets
- [ ] Max DD reduced by 10%+
- [ ] Walk-forward: 60%+ profitable quarters

---

## 📊 Application 2: SI-ADX Spread Trading

### Hypothesis
SI and ADX are cointegrated; trade mean reversion of spread.

### Test Design

```
ENTRY: z-score of (SI - β×ADX) > 2 or < -2
EXIT: z-score returns to ±0.5
SIZING: Fixed or inversely proportional to z-score

Parameters to test:
- Entry z: [1.5, 2.0, 2.5]
- Exit z: [0.0, 0.5, 1.0]
- Lookback for z-score: [30, 60, 90]
```

### Implementation

```python
def test_si_adx_spread(data, si, adx, entry_z=2.0, exit_z=0.5, lookback=60):
    # Normalize ADX to SI scale
    adx_norm = adx / 100
    
    # Compute spread
    spread = si - adx_norm
    spread_mean = spread.rolling(lookback).mean()
    spread_std = spread.rolling(lookback).std()
    z_score = (spread - spread_mean) / spread_std
    
    # Generate signals
    position = pd.Series(0, index=data.index)
    in_trade = False
    
    for i in range(1, len(z_score)):
        if not in_trade:
            if z_score.iloc[i] > entry_z:
                position.iloc[i] = -1  # Short spread
                in_trade = True
            elif z_score.iloc[i] < -entry_z:
                position.iloc[i] = 1   # Long spread
                in_trade = True
        else:
            if abs(z_score.iloc[i]) < exit_z:
                position.iloc[i] = 0
                in_trade = False
            else:
                position.iloc[i] = position.iloc[i-1]
    
    returns = data['close'].pct_change()
    return position.shift(1) * returns
```

### Success Criteria
- [ ] Sharpe > 0.7 in crypto
- [ ] Win rate > 55%
- [ ] Avg trade duration > 3 days

---

## 📊 Application 3: Factor Timing

### Hypothesis
Use SI to decide when to apply momentum vs mean-reversion.

### Test Design

```
RULE:
- High SI (> p60) → Apply momentum strategy
- Low SI (< p40) → Apply mean-reversion strategy
- Middle → No position (cash)

BASELINE: Always momentum, always mean-rev, 50/50 mix
```

### Implementation

```python
def test_factor_timing(data, si, threshold=0.5):
    returns = data['close'].pct_change()
    si_rank = si.rank(pct=True)
    
    # Momentum signal (5-day return sign)
    momentum_signal = np.sign(data['close'].pct_change(5))
    
    # Mean-reversion signal (RSI extremes)
    rsi = compute_rsi(data['close'], 14)
    meanrev_signal = np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0))
    
    # Combine based on SI
    position = np.where(si_rank > threshold, momentum_signal,
                np.where(si_rank < (1-threshold), meanrev_signal, 0))
    
    strategy_returns = pd.Series(position).shift(1) * returns
    return strategy_returns
```

### Success Criteria
- [ ] Sharpe > baseline in 4/6 assets
- [ ] Factor timing adds > 0.15 Sharpe vs always-momentum
- [ ] Consistent across markets

---

## 📊 Application 4: Volatility Forecasting

### Hypothesis
SI negatively predicts future volatility.

### Test Design

```
MODEL: Vol_forecast = α + β × SI + γ × Realized_vol
BASELINE: EWMA vol, GARCH(1,1)
METRIC: RMSE, MAE of 1-day ahead vol forecast
```

### Implementation

```python
def test_vol_forecasting(data, si):
    returns = data['close'].pct_change()
    realized_vol = returns.rolling(20).std()
    future_vol = realized_vol.shift(-1)  # Next day's vol
    
    # Baseline: EWMA
    ewma_vol = returns.ewm(span=20).std()
    
    # SI-enhanced
    from sklearn.linear_model import LinearRegression
    X = pd.concat([si, ewma_vol], axis=1).dropna()
    y = future_vol.loc[X.index]
    
    # Train/test split
    split = int(len(X) * 0.7)
    model = LinearRegression()
    model.fit(X.iloc[:split], y.iloc[:split])
    
    pred = model.predict(X.iloc[split:])
    rmse_si = np.sqrt(np.mean((pred - y.iloc[split:])**2))
    rmse_ewma = np.sqrt(np.mean((ewma_vol.iloc[split:] - y.iloc[split:])**2))
    
    return {'rmse_si': rmse_si, 'rmse_ewma': rmse_ewma, 
            'improvement': (rmse_ewma - rmse_si) / rmse_ewma}
```

### Success Criteria
- [ ] RMSE reduced by 5%+ vs EWMA
- [ ] SI coefficient significant (p < 0.05)
- [ ] Works in high-vol environments

---

## 📊 Application 5: Dynamic Stop-Loss

### Hypothesis
SI-adjusted stops reduce whipsaws and improve win rate.

### Test Design

```
BASELINE: Fixed 2× ATR stop
TEST: Stop = ATR × f(SI_regime)
  - High SI: 1.5× ATR (tight)
  - Mid SI: 2.0× ATR (normal)
  - Low SI: 3.0× ATR (wide)
```

### Implementation

```python
def test_dynamic_stop(data, si, base_strategy_signals):
    atr = compute_atr(data, 14)
    si_rank = si.rank(pct=True)
    
    # ATR multiplier based on SI
    multiplier = np.where(si_rank > 0.7, 1.5,
                 np.where(si_rank < 0.3, 3.0, 2.0))
    stop_distance = atr * multiplier
    
    # Apply stops to base strategy
    returns = data['close'].pct_change()
    position = base_strategy_signals.copy()
    
    for i in range(1, len(data)):
        if position.iloc[i-1] != 0:
            # Check if stop hit
            price_change = (data['close'].iloc[i] / data['close'].iloc[i-1] - 1)
            if position.iloc[i-1] > 0 and price_change < -stop_distance.iloc[i]:
                position.iloc[i] = 0  # Stopped out
            elif position.iloc[i-1] < 0 and price_change > stop_distance.iloc[i]:
                position.iloc[i] = 0
    
    return position.shift(1) * returns
```

### Success Criteria
- [ ] Win rate improved by 3%+
- [ ] Fewer whipsaws (count exit-reentry)
- [ ] Max DD reduced

---

## 📊 Application 6: Regime Rebalancing

### Hypothesis
Rebalancing on SI regime changes is better than calendar-based.

### Test Design

```
BASELINE: Quarterly rebalancing
TEST: Rebalance when SI regime changes (High/Mid/Low)

ALLOCATION:
- High SI: 70% equity, 30% bonds
- Mid SI: 60% equity, 40% bonds
- Low SI: 50% equity, 50% bonds
```

### Implementation

```python
def test_regime_rebalancing(equity_data, bond_data, si):
    equity_returns = equity_data['close'].pct_change()
    bond_returns = bond_data['close'].pct_change()
    
    si_rank = si.rank(pct=True)
    regime = np.where(si_rank > 0.67, 'high',
             np.where(si_rank < 0.33, 'low', 'mid'))
    
    # SI-based allocation
    equity_weight = np.where(regime == 'high', 0.7,
                    np.where(regime == 'low', 0.5, 0.6))
    
    si_returns = equity_weight * equity_returns + (1 - equity_weight) * bond_returns
    
    # Baseline: quarterly rebalance (60/40)
    baseline_returns = 0.6 * equity_returns + 0.4 * bond_returns
    
    return {
        'si_sharpe': sharpe_ratio(si_returns),
        'baseline_sharpe': sharpe_ratio(baseline_returns),
        'rebalance_count': (pd.Series(regime) != pd.Series(regime).shift()).sum()
    }
```

### Success Criteria
- [ ] Sharpe > 60/40 baseline
- [ ] Fewer rebalances than monthly
- [ ] Lower turnover cost

---

## 📊 Application 7: Tail Risk Hedge

### Hypothesis
Low SI predicts higher tail risk; scale hedges accordingly.

### Test Design

```
BASELINE: Constant 5% hedge allocation
TEST: Hedge = 5% × f(SI_percentile)
  - SI < p10: 2.5× hedge (12.5%)
  - SI < p25: 1.5× hedge (7.5%)
  - SI > p75: 0.5× hedge (2.5%)
```

### Implementation

```python
def test_tail_hedge(data, si, hedge_returns):
    returns = data['close'].pct_change()
    si_rank = si.rank(pct=True)
    
    # Hedge multiplier
    hedge_mult = np.where(si_rank < 0.1, 2.5,
                 np.where(si_rank < 0.25, 1.5,
                 np.where(si_rank > 0.75, 0.5, 1.0)))
    
    base_hedge = 0.05
    hedge_weight = base_hedge * hedge_mult
    equity_weight = 1 - hedge_weight
    
    # Portfolio returns
    portfolio = equity_weight * returns + hedge_weight * hedge_returns
    
    # Metrics
    tail_events = returns < returns.quantile(0.05)
    portfolio_tail_loss = portfolio[tail_events].mean()
    baseline_tail_loss = returns[tail_events].mean()
    
    return {
        'sharpe': sharpe_ratio(portfolio),
        'tail_protection': (baseline_tail_loss - portfolio_tail_loss) / abs(baseline_tail_loss),
        'avg_hedge_cost': hedge_weight.mean() * abs(hedge_returns.mean())
    }
```

### Success Criteria
- [ ] Tail loss reduced by 20%+
- [ ] Sharpe not significantly worse
- [ ] Efficient hedge usage (low cost in high SI)

---

## 📊 Application 8: Cross-Asset Momentum

### Hypothesis
SI divergence between correlated assets predicts relative returns.

### Test Design

```
PAIRS: SPY-QQQ, BTC-ETH, EURUSD-GBPUSD
SIGNAL: z-score of (SI_1 - SI_2)
TRADE: Long laggard, short leader when z > 2
```

### Implementation

```python
def test_cross_asset_momentum(data1, data2, si1, si2, lookback=60):
    returns1 = data1['close'].pct_change()
    returns2 = data2['close'].pct_change()
    
    # SI spread
    si_spread = si1 - si2
    spread_mean = si_spread.rolling(lookback).mean()
    spread_std = si_spread.rolling(lookback).std()
    z_score = (si_spread - spread_mean) / spread_std
    
    # Position: long asset2, short asset1 when z > 2
    position = np.where(z_score > 2, -1,
               np.where(z_score < -2, 1, 0))
    
    # Relative returns
    relative_returns = returns1 - returns2
    strategy_returns = pd.Series(position).shift(1) * relative_returns
    
    return strategy_returns
```

### Success Criteria
- [ ] Sharpe > 0.5 for pairs
- [ ] Market-neutral (low beta)
- [ ] Consistent across pair types

---

## 📊 Application 9: Ensemble Strategy

### Hypothesis
Combining SI signals reduces variance and improves robustness.

### Test Design

```
COMPONENTS:
1. SI Risk Budgeting signal
2. SI-ADX Spread signal
3. SI Factor Timing signal

COMBINATION:
A. Equal weight
B. Correlation-weighted (inverse of pairwise correlation)
C. Performance-weighted (rolling Sharpe)
```

### Implementation

```python
def test_ensemble(data, si, adx):
    # Generate component signals
    sig1 = generate_risk_budgeting_signal(data, si)
    sig2 = generate_spread_signal(data, si, adx)
    sig3 = generate_factor_timing_signal(data, si)
    
    signals = pd.DataFrame({'rb': sig1, 'spread': sig2, 'timing': sig3})
    returns = data['close'].pct_change()
    
    # Equal weight ensemble
    equal_ensemble = signals.mean(axis=1)
    
    # Correlation-weighted
    corr = signals.rolling(60).corr()
    # ... weight by inverse correlation
    
    # Performance-weighted
    rolling_sharpe = signals.rolling(60).apply(sharpe_ratio)
    weights = rolling_sharpe / rolling_sharpe.sum(axis=1)
    perf_ensemble = (signals * weights).sum(axis=1)
    
    return {
        'equal_sharpe': sharpe_ratio(equal_ensemble.shift(1) * returns),
        'perf_sharpe': sharpe_ratio(perf_ensemble.shift(1) * returns),
    }
```

### Success Criteria
- [ ] Ensemble Sharpe > best single component
- [ ] Lower volatility than components
- [ ] Stable weights over time

---

## 📊 Application 10: Entry Timing

### Hypothesis
High SI + pullback = optimal entry timing.

### Test Design

```
SIGNAL:
- STRONG_BUY: SI > p70 AND price < 20-day low
- BUY: SI > p60 AND price < 10-day avg
- AVOID: SI < p30 AND price > 20-day high
- WAIT: SI < p30
```

### Implementation

```python
def test_entry_timing(data, si):
    returns = data['close'].pct_change()
    si_rank = si.rank(pct=True)
    
    # Price position
    high_20 = data['close'].rolling(20).max()
    low_20 = data['close'].rolling(20).min()
    price_pos = (data['close'] - low_20) / (high_20 - low_20)
    
    # Entry signal
    signal = np.where((si_rank > 0.7) & (price_pos < 0.2), 2,    # Strong buy
             np.where((si_rank > 0.6) & (price_pos < 0.5), 1,     # Buy
             np.where((si_rank < 0.3) & (price_pos > 0.8), -1,    # Avoid
             np.where(si_rank < 0.3, 0, 0.5))))                   # Wait / Neutral
    
    # Calculate returns for each signal type
    strategy_returns = pd.Series(signal).shift(1) * returns
    
    # Track 5-day forward returns by signal
    fwd_5d = returns.rolling(5).sum().shift(-5)
    
    return {
        'sharpe': sharpe_ratio(strategy_returns),
        'strong_buy_edge': fwd_5d[signal == 2].mean(),
        'buy_edge': fwd_5d[signal == 1].mean(),
        'avoid_edge': fwd_5d[signal == -1].mean(),
    }
```

### Success Criteria
- [ ] STRONG_BUY has > 1% 5-day edge
- [ ] Timing improves entry prices
- [ ] Consistent across assets

---

## 🏃 Execution Plan

### Phase 1: Quick Screen (1 day)

Run simplified version of all 10 applications on 2 assets (BTC, SPY):

```python
QUICK_TEST = {
    'assets': ['BTCUSDT', 'SPY'],
    'period': 'last 2 years',
    'si_window': 7,
    'metric': 'sharpe_ratio',
}
```

**Output:** Ranked list of applications by Sharpe

### Phase 2: Deep Dive on Top 5 (2 days)

For top 5 applications:
- Test on all 6 assets
- Parameter optimization on train set
- Validation on val set
- Walk-forward analysis

### Phase 3: Final Validation on Top 3 (1 day)

For top 3 applications:
- Test set (holdout) performance
- Statistical significance (bootstrap CI)
- Cross-market consistency
- Transaction cost sensitivity

### Phase 4: Thesis Selection (0.5 day)

Select best application based on:
1. **Sharpe Ratio** (40% weight)
2. **Cross-Asset Consistency** (25% weight)
3. **Theoretical Clarity** (20% weight)
4. **Practical Implementability** (15% weight)

---

## 📊 Comparison Matrix Template

| Application | BTC | ETH | SPY | QQQ | EUR | GBP | Avg | Rank |
|-------------|-----|-----|-----|-----|-----|-----|-----|------|
| 1. Risk Budgeting | ? | ? | ? | ? | ? | ? | ? | ? |
| 2. SI-ADX Spread | ? | ? | ? | ? | ? | ? | ? | ? |
| 3. Factor Timing | ? | ? | ? | ? | ? | ? | ? | ? |
| 4. Vol Forecasting | ? | ? | ? | ? | ? | ? | ? | ? |
| 5. Dynamic Stop | ? | ? | ? | ? | ? | ? | ? | ? |
| 6. Regime Rebalance | ? | ? | ? | ? | ? | ? | ? | ? |
| 7. Tail Hedge | ? | ? | ? | ? | ? | ? | ? | ? |
| 8. Cross-Asset | ? | ? | ? | ? | ? | ? | ? | ? |
| 9. Ensemble | ? | ? | ? | ? | ? | ? | ? | ? |
| 10. Entry Timing | ? | ? | ? | ? | ? | ? | ? | ? |

---

## 📁 Output Files

```
results/application_testing/
├── phase1_quick_screen.json
├── phase2_deep_dive/
│   ├── app1_risk_budgeting.json
│   ├── app2_spread_trading.json
│   └── ...
├── phase3_final_validation/
│   ├── top3_comparison.json
│   └── statistical_tests.json
├── final_selection.json
└── figures/
    ├── sharpe_comparison.png
    ├── equity_curves.png
    └── consistency_heatmap.png
```

---

## ⏱️ Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Quick Screen | 1 day | Top 5 ranked list |
| Phase 2: Deep Dive | 2 days | Validated top 3 |
| Phase 3: Final Validation | 1 day | Best application selected |
| Phase 4: Selection | 0.5 day | Thesis application confirmed |
| **Total** | **4.5 days** | **Production-ready strategy** |

---

*Plan Created: January 18, 2026*  
*Author: Yuhao Li, University of Pennsylvania*
