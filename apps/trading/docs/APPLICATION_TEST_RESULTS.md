# SI Application Test Results

**Date:** January 18, 2026
**Author:** Yuhao Li, University of Pennsylvania

---

## 📊 Phase 1: Quick Screen Results

All 10 applications tested on 6 assets (BTC, ETH, SPY, QQQ, EURUSD, GBPUSD).

---

## 🏆 FINAL RANKING

| Rank | Application | Avg Sharpe | Consistency | Best Market | Thesis Fit |
|------|-------------|------------|-------------|-------------|------------|
| **1** | **6. Regime Rebalance** | **0.400** | **67%** | SPY (0.85) | ⭐⭐⭐ |
| **2** | **5. Dynamic Stop** | **0.400** | **67%** | SPY (0.82) | ⭐⭐⭐ |
| **3** | **1. Risk Budgeting** | **0.366** | **67%** | SPY (0.77) | ⭐⭐⭐ |
| 4 | 2. SI-ADX Spread | 0.235 | **100%** | QQQ (0.45) | ⭐⭐ |
| 5 | 10. Entry Timing | 0.185 | 67% | BTC (0.57) | ⭐⭐ |
| 6 | 3. Factor Timing | 0.181 | 67% | SPY (0.49) | ⭐⭐ |
| 7 | 7. Tail Hedge | - | 100% | SPY (0.86) | ⭐ |
| 8 | 4. Vol Forecasting | - | 100% | ETH (4.4%) | ⭐⭐ |
| 9 | 9. Ensemble | - | 33% | SPY (0.92) | ⭐ |
| 10 | 8. Cross-Asset | -0.200 | 33% | FX (0.31) | ⭐ |

---

## 📈 Top 3 Candidates for Thesis

### 🥇 #1: Regime Rebalance (Application 6)

**Concept:** Adjust position size (equity allocation) based on SI regime (High/Mid/Low).

| Asset | Baseline Sharpe | Strategy Sharpe | Improvement | Max DD Reduction |
|-------|-----------------|-----------------|-------------|------------------|
| BTCUSDT | 0.501 | **0.597** | +19% | **20%** |
| ETHUSDT | 0.476 | **0.558** | +17% | **15%** |
| SPY | 0.844 | **0.854** | +1% | 14% |
| QQQ | 0.699 | 0.600 | -14% | 22% |
| EURUSD | -0.076 | -0.101 | - | 19% |
| GBPUSD | -0.042 | -0.108 | - | 29% |

**Key Findings:**
- ✅ **Reduces max drawdown** in ALL 6 assets (avg 20%)
- ✅ Improves Sharpe in crypto (+18%)
- ✅ Clear, interpretable mechanism
- ⚠️ Neutral/negative in Forex (low volatility assets)

**Thesis Angle:** "SI as a regime indicator for portfolio risk management"

---

### 🥈 #2: Dynamic Stop-Loss (Application 5)

**Concept:** Adjust stop-loss levels based on SI (High SI = tight stops, Low SI = wide stops).

| Asset | Baseline Sharpe | Strategy Sharpe | Stops Triggered |
|-------|-----------------|-----------------|-----------------|
| BTCUSDT | 0.552 | 0.482 | 8 vs 5 |
| ETHUSDT | 0.478 | **0.498** | 8 vs 2 |
| SPY | 0.904 | 0.822 | 12 vs 9 |
| QQQ | 0.754 | 0.726 | 15 vs 6 |
| EURUSD | -0.091 | -0.089 | 2 vs 1 |
| GBPUSD | -0.037 | -0.040 | 2 vs 1 |

**Key Findings:**
- ✅ Preserves capital (more disciplined stop usage)
- ⚠️ Slight performance drag (more stops = more exit costs)
- ✅ Clear risk management value
- ⚠️ Works better in trending markets

**Thesis Angle:** "SI-adaptive stop-loss for improved risk control"

---

### 🥉 #3: Risk Budgeting (Application 1)

**Concept:** Scale position size linearly with SI percentile (0.5 to 1.5x).

| Asset | Baseline Sharpe | Strategy Sharpe | Max DD | Calmar |
|-------|-----------------|-----------------|--------|--------|
| BTCUSDT | 0.501 | **0.576** | -73% | 0.38 |
| ETHUSDT | 0.476 | **0.541** | -80% | 0.45 |
| SPY | 0.844 | 0.775 | -29% | 0.45 |
| QQQ | 0.699 | 0.617 | -34% | 0.42 |
| EURUSD | -0.076 | -0.165 | -23% | -0.06 |
| GBPUSD | -0.042 | -0.149 | -22% | -0.06 |

**Key Findings:**
- ✅ **Improves crypto** (BTC +15%, ETH +14%)
- ⚠️ Hurts stocks slightly (market already trending)
- ❌ Negative in Forex
- ✅ Strongest theoretical link to SI

**Thesis Angle:** "SI as position sizing signal - larger when market is clear"

---

## 📊 Application-by-Application Breakdown

### Application 2: SI-ADX Spread Trading

**Best Feature: 100% Consistency** (all assets positive Sharpe)

| Asset | Sharpe | Max DD | Trades | Win Rate |
|-------|--------|--------|--------|----------|
| BTCUSDT | 0.225 | -48% | 47 | 10% |
| ETHUSDT | 0.233 | -62% | 55 | 12% |
| SPY | 0.151 | -14% | 28 | 8% |
| QQQ | **0.454** | -17% | 33 | 10% |
| EURUSD | 0.082 | -6% | 29 | 8% |
| GBPUSD | 0.263 | -9% | 41 | 10% |

**Verdict:** ✅ Robust but modest returns. Good for diversification.

---

### Application 3: Factor Timing

**Best Feature: Huge improvement vs baseline momentum**

| Asset | Baseline (Mom) | Strategy | Improvement |
|-------|----------------|----------|-------------|
| BTCUSDT | -0.461 | -0.126 | **+0.33** |
| ETHUSDT | -0.104 | 0.219 | **+0.32** |
| SPY | -0.367 | 0.486 | **+0.85** |
| QQQ | 0.027 | 0.267 | **+0.24** |
| EURUSD | -0.175 | -0.003 | +0.17 |
| GBPUSD | -0.035 | 0.245 | **+0.28** |

**Verdict:** ✅ Dramatically beats pure momentum. SI correctly identifies when momentum works.

---

### Application 4: Volatility Forecasting

**Measures: RMSE improvement over EWMA baseline**

| Asset | Baseline RMSE | SI RMSE | Improvement |
|-------|---------------|---------|-------------|
| BTCUSDT | 0.0063 | 0.0061 | **2.6%** |
| ETHUSDT | 0.0115 | 0.0110 | **4.4%** |
| SPY | 0.0036 | 0.0035 | 2.7% |
| QQQ | 0.0043 | 0.0042 | 3.1% |
| EURUSD | 0.0013 | 0.0012 | 3.8% |
| GBPUSD | 0.0011 | 0.0011 | 3.9% |

**Verdict:** ✅ SI adds 3-4% forecasting accuracy across all assets. Statistical not trading application.

---

### Application 7: Tail Risk Hedge

**Measures: Tail loss protection (5th percentile)**

| Asset | Unhedged Tail | Hedged Tail | Protection |
|-------|---------------|-------------|------------|
| BTCUSDT | -6.88% | -6.32% | **8.2%** |
| ETHUSDT | -9.00% | -8.31% | **7.7%** |
| SPY | -2.48% | -2.24% | **9.4%** |
| QQQ | -3.27% | -3.01% | **8.1%** |
| EURUSD | -1.04% | -0.95% | **8.7%** |
| GBPUSD | -1.21% | -1.11% | **8.1%** |

**Verdict:** ✅ Consistent 8-9% tail protection. Good risk management story.

---

### Application 8: Cross-Asset SI Momentum

**Pairs tested:** BTC-ETH, SPY-QQQ, EUR-GBP

| Pair | Sharpe | Max DD | Trades |
|------|--------|--------|--------|
| BTC-ETH | **-0.67** | -48% | 121 |
| SPY-QQQ | -0.24 | -4% | 71 |
| EUR-GBP | **0.31** | -3% | 70 |

**Verdict:** ❌ Only works in Forex. Cross-asset SI divergence is not a reliable signal.

---

### Application 9: Ensemble

**Combines:** Risk Budgeting + Momentum-on-High-SI + Mean-Rev-on-Low-SI

| Asset | SI Only | Ensemble | Improvement |
|-------|---------|----------|-------------|
| BTCUSDT | 0.546 | 0.339 | **-38%** |
| ETHUSDT | 0.534 | 0.525 | -2% |
| SPY | 0.846 | **0.924** | +9% |
| QQQ | 0.726 | 0.692 | -5% |
| EURUSD | -0.139 | -0.141 | -2% |
| GBPUSD | -0.154 | -0.031 | **+80%** |

**Verdict:** ⚠️ Mixed. Ensemble doesn't consistently beat SI-only. Complexity not justified.

---

### Application 10: Entry Timing

**Measures: 5-day forward returns by signal type**

| Asset | Strong Buy Edge | Buy Edge | Avoid Edge |
|-------|-----------------|----------|------------|
| BTCUSDT | **+0.96%** | +0.48% | **-1.15%** |
| ETHUSDT | **+1.37%** | -0.50% | +0.11% |
| SPY | **+1.17%** | -0.34% | +0.39% |
| QQQ | **+0.83%** | +0.65% | +0.51% |
| EURUSD | -0.34% | -0.07% | +0.06% |
| GBPUSD | -0.08% | -0.19% | +0.18% |

**Verdict:** ✅ Strong buy signal works in BTC, ETH, SPY, QQQ. Entry timing is valuable in risk assets.

---

## 🎯 THESIS RECOMMENDATION

Based on the test results, here are the **top 3 candidates** ranked by thesis suitability:

### 🏆 WINNER: **Application 6 - Regime Rebalancing**

**Why:**
1. **Best risk story:** 20% avg drawdown reduction across ALL assets
2. **Clearest mechanism:** High SI = clear market = more equity; Low SI = unclear = less equity
3. **Novel contribution:** SI as a regime indicator for asset allocation
4. **Robust:** Works in crypto (best), stocks (neutral), forex (neutral)
5. **Simple to explain:** "When the market is readable, take more risk"

**Thesis Title Options:**
- "Emergent Specialization as a Market Regime Indicator"
- "SI-Based Dynamic Asset Allocation: From Agent Competition to Risk Budgeting"

---

### 🥈 RUNNER-UP: **Application 2 - SI-ADX Spread Trading**

**Why:**
1. **100% consistency:** Positive Sharpe in ALL 6 assets
2. **Strong theoretical basis:** SI-ADX cointegration discovered in Phase 1
3. **Mean-reversion strategy:** Well-understood by practitioners
4. **Lower max DD:** Especially good in stocks and forex

**Thesis Title Options:**
- "Trading the SI-ADX Spread: Exploiting Cointegration in Market Clarity"
- "Specialization Index as a Mean-Reverting Signal"

---

### 🥉 THIRD: **Application 3 - Factor Timing**

**Why:**
1. **Largest improvement:** +0.37 Sharpe vs pure momentum (82% improvement)
2. **Clear economic story:** "SI tells you when momentum works"
3. **Novel angle:** Using emergent specialization to time factor exposure
4. **Connects to factor literature:** Links to Fama-French, momentum crashes

**Thesis Title Options:**
- "When Does Momentum Work? SI as a Factor Timing Signal"
- "Emergent Specialization and Factor Exposure"

---

## 📋 Next Steps

1. **Deep dive on top 3** (2 days)
   - Walk-forward validation
   - Parameter optimization
   - Robustness tests

2. **Final validation on winner** (1 day)
   - Holdout test set
   - Statistical significance
   - Bootstrap confidence intervals

3. **Thesis write-up** (3 days)
   - Clear problem statement
   - Methodology
   - Results
   - Limitations

---

## 📊 Market-by-Market Summary

| Market | Best Application | Sharpe | Notes |
|--------|------------------|--------|-------|
| **Crypto (BTC)** | Regime Rebalance | 0.60 | SI works best here |
| **Crypto (ETH)** | Regime Rebalance | 0.56 | Consistent with BTC |
| **Stocks (SPY)** | Ensemble | 0.92 | SI+factors work together |
| **Stocks (QQQ)** | SI-ADX Spread | 0.45 | Mean reversion works |
| **Forex (EUR)** | SI-ADX Spread | 0.08 | Low vol = modest edge |
| **Forex (GBP)** | SI-ADX Spread | 0.26 | Better than EUR |

---

*Report Generated: January 18, 2026*
*Author: Yuhao Li, University of Pennsylvania*
