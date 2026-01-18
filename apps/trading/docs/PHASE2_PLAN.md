# Phase 2: SI Applications Plan

**Date:** January 17, 2026  
**Foundation:** 125 Phase 1 Discoveries  
**Goal:** Translate SI insights into practical applications

---

## 🎯 Phase 2 Objectives

Based on Phase 1 findings, SI is:
- A **lagging indicator** of market clarity
- **Cointegrated** with ADX (trend strength)
- **Mean-reverts** in 4-5 days
- **Sticky at extremes** (80% persistence in High-SI)
- **Crisis-sensitive** (-27% during crashes)
- **Best at extremes** (p10/p90 sweet spots)

---

## 📋 Phase 2 Applications

### P2.1: Regime Detection System
**Use:** High-SI is sticky (80% persistence)
- Build real-time regime classifier
- Compare to HMM baseline
- Measure accuracy on labeled regimes

### P2.2: Volatility Forecasting
**Use:** SI negatively correlates with volatility
- SI → Future volatility prediction
- Compare to GARCH, HAR-RV
- Measure RMSE improvement

### P2.3: Risk-Adjusted Position Sizing
**Use:** SI extremes are sweet spots
- High SI → Larger positions
- Low SI → Smaller positions
- Compare to constant sizing, inverse-vol

### P2.4: Crisis Early Warning
**Use:** SI crashes during crises (-27%)
- Detect SI drops > 2σ
- Lead time before drawdowns
- False positive rate

### P2.5: Factor Timing
**Use:** SI = market clarity
- SI high → trade momentum
- SI low → trade mean-reversion
- Compare to static factor allocation

### P2.6: Pairs Trading with SI
**Use:** SI-ADX cointegration
- Trade SI-ADX spread
- Mean reversion strategy
- Risk-adjusted returns

### P2.7: Cross-Asset SI Sync Trading
**Use:** Same-market SI correlation (0.53)
- When one asset's SI diverges, trade convergence
- SPY-QQQ, BTC-ETH pairs

### P2.8: Drawdown Prediction
**Use:** SI-Drawdown relationship
- Predict future drawdowns from SI level
- Compare to simple volatility model

### P2.9: Optimal Rebalancing Timing
**Use:** SI mean reversion (4-5 days)
- Rebalance when SI at extreme
- Compare to fixed-interval rebalancing

### P2.10: SI-Enhanced Technical Signals
**Use:** SI confirms trend strength
- Combine SI with RSI, MACD, etc.
- Measure improvement in signal quality

---

## 📊 Success Metrics

| Application | Primary Metric | Target |
|-------------|----------------|--------|
| Regime Detection | Accuracy | > 70% |
| Vol Forecasting | RMSE reduction | > 10% |
| Position Sizing | Sharpe improvement | > 0.2 |
| Crisis Warning | Lead time | > 5 days |
| Factor Timing | Alpha | > 0 (significant) |
| Pairs Trading | Sharpe | > 1.0 |
| Cross-Asset | Profit factor | > 1.5 |
| Drawdown Pred | AUC-ROC | > 0.65 |
| Rebalancing | Cost reduction | > 20% |
| Enhanced Signals | Win rate | > 55% |

---

## 🚀 Execution Order

**Tier 1 (High Confidence):**
1. Regime Detection - uses proven sticky extremes
2. Position Sizing - uses proven p10/p90 insight
3. Crisis Warning - uses proven crash behavior

**Tier 2 (Medium Confidence):**
4. Vol Forecasting - uses established correlation
5. Factor Timing - uses market clarity concept
6. Pairs Trading - uses cointegration

**Tier 3 (Exploratory):**
7. Cross-Asset Sync
8. Drawdown Prediction
9. Rebalancing Timing
10. Enhanced Signals

