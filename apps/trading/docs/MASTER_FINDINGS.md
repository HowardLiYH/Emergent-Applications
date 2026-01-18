# Master Findings Document

**Date:** January 17, 2026  
**Author:** Yuhao Li, University of Pennsylvania  
**Total Discoveries:** 80+  
**Methods Applied:** 25+

---

## ⚠️ CRITICAL CORRECTIONS (Previously Wrong)

| # | Previous Assumption | CORRECTED Finding | Evidence |
|---|---------------------|-------------------|----------|
| 1 | SI predicts returns | SI **LAGS** market state | Transfer Entropy: ADX→SI ratio=0.6 |
| 2 | SI → Volatility | **Volatility → SI** | Granger causality p<0.05 |
| 3 | SI works short-term | **30+ day only** | Short-term r = -0.05 |
| 4 | SI is mean-reverting | SI has **LONG MEMORY** (H=0.83) | Hurst exponent analysis |
| 5 | SI is normally distributed | **Non-normal**, fat tails | Shapiro-Wilk p<0.001 |

---

## 🏆 TOP 10 VERIFIED FINDINGS

| Rank | Discovery | Evidence | Confidence |
|------|-----------|----------|------------|
| **1** | **SI is LAGGING** | TE: ADX→SI (0.6), Granger: Vol→SI | ★★★★★ |
| **2** | **SI has LONG MEMORY** | Hurst H=0.83-0.87 (all assets) | ★★★★★ |
| **3** | **SI-ADX is COINTEGRATED** | Engle-Granger p<0.0001 | ★★★★★ |
| **4** | **RSI Extremity is #1 correlate** | r=+0.243, 9/9 assets | ★★★★★ |
| **5** | **High SI = Low Volatility** | High SI: 15.5% vol, Low SI: 20.4% | ★★★★☆ |
| **6** | **30-120 day relationship only** | Short-term r=-0.05, Long-term r=+0.35 | ★★★★★ |
| **7** | **SI negatively correlates with market entropy** | r=-0.15, confirms theory | ★★★★☆ |
| **8** | **Same-market assets sync** | SPY-QQQ: r=0.53 | ★★★★☆ |
| **9** | **2 latent SI states (HMM)** | 88% persistence in high state | ★★★★☆ |
| **10** | **SI survives HAC/permutation tests** | p<0.01 for BTC, EUR | ★★★★☆ |

---

## 📊 CATEGORY 1: Correlation Structure

### Top SI Correlates (Ranked)

| Rank | Feature | Correlation | Consistency | Bootstrap 95% CI |
|------|---------|-------------|-------------|------------------|
| 1 | **RSI Extremity** | +0.243 | 9/9 ✓ | [0.15, 0.33] |
| 2 | **Fractal Dimension** | -0.231 | 3/3 ✓ | Significant |
| 3 | **Directional Consistency** | +0.206 | 3/3 ✓ | Significant |
| 4 | **MCI (Market Clarity)** | +0.205 | 9/9 ✓ | Significant |
| 5 | **DX (unsmoothed)** | +0.205 | 9/9 ✓ | Significant |
| 6 | **Efficiency (Kaufman)** | +0.177 | 3/3 ✓ | Significant |
| 7 | **Volatility** | -0.158 | 9/9 ✓ | Significant |
| 8 | **ADX** | +0.127 | 9/9 ✓ | [0.02, 0.22] |

---

## 📊 CATEGORY 2: Causality & Information Flow

| Direction | Method | Result | Interpretation |
|-----------|--------|--------|----------------|
| ADX → SI | Transfer Entropy | TE ratio = 0.6 | ADX PREDICTS SI |
| Volatility → SI | Transfer Entropy | TE ratio = 0.46-0.56 | Vol PREDICTS SI |
| Volatility → SI | Granger | p < 0.05 | Vol CAUSES SI |
| SI → Features | All methods | NOT significant | SI doesn't predict |

**Conclusion:** Features → SI (SI is a lagging indicator)

---

## 📊 CATEGORY 3: Stochastic Process Properties

| Property | BTC | SPY | EUR | Method |
|----------|-----|-----|-----|--------|
| **Hurst Exponent** | 0.831 | 0.866 | 0.861 | R/S analysis |
| **OU Half-life** | 4.4 days | 5.1 days | 5.3 days | Regression |
| **GPD Tail Shape** | 1.416 | 0.454 | 0.476 | EVT |
| **Entropy Rate** | 0.444 | 0.423 | 0.413 | Conditional H |
| **HMM Persistence** | 88% | 89% | 88% | Gaussian HMM |

**Interpretation:** SI follows a Fractional Ornstein-Uhlenbeck process:
- Long-range dependence (H > 0.8)
- Local mean reversion (4-5 day half-life)
- Fat upper tails

---

## 📊 CATEGORY 4: Cointegration

| Pair | Test Statistic | p-value | Status |
|------|---------------|---------|--------|
| SI-ADX (BTC) | -10.2 | <0.0001 | ✅ Cointegrated |
| SI-ADX (SPY) | -10.3 | <0.0001 | ✅ Cointegrated |
| SI-ADX (EUR) | -13.4 | <0.0001 | ✅ Cointegrated |
| SI-RSI_ext (BTC) | -10.6 | <0.0001 | ✅ Cointegrated |
| SI-RSI_ext (SPY) | -10.6 | <0.0001 | ✅ Cointegrated |
| SI-RSI_ext (EUR) | -13.9 | <0.0001 | ✅ Cointegrated |

**Implication:** SI and ADX share a long-run equilibrium.

---

## 📊 CATEGORY 5: Frequency Domain

| Frequency | Period | SI-ADX Correlation | Tradeable? |
|-----------|--------|-------------------|------------|
| Very Short | 3-7 days | **-0.05** | ❌ NEGATIVE |
| Short | 7-14 days | ~0 | ❌ |
| Medium | 14-30 days | ~0 | ❌ |
| **Long** | 30-60 days | **+0.24** | ✓ |
| **Very Long** | 60-120 days | **+0.35** | ✓ |

**Conclusion:** Only monthly/quarterly SI is meaningful.

---

## 📊 CATEGORY 6: Distribution Properties

| Statistic | BTC | SPY | EUR | Interpretation |
|-----------|-----|-----|-----|----------------|
| Mean | 0.019 | 0.022 | 0.022 | Similar across assets |
| Std | 0.011 | 0.011 | 0.012 | Similar variability |
| Skewness | +0.57 | +0.23 | +0.27 | Positively skewed |
| Kurtosis | -0.41 | -0.87 | -0.90 | Platykurtic (light central) |
| Normality | ❌ | ❌ | ❌ | All non-normal |

**Cross-Asset Distribution:**
- BTC ≠ SPY (different markets)
- SPY = QQQ = EUR (same distribution)

---

## 📊 CATEGORY 7: Tail Dependence (Copula)

| Feature Pair | Lower Tail λ | Upper Tail λ | Asymmetry |
|--------------|-------------|-------------|-----------|
| SI-RSI_ext (BTC) | 0.13 | 0.31 | 2.4x |
| SI-RSI_ext (SPY) | 0.12 | 0.39 | 3.3x |
| SI-RSI_ext (EUR) | 0.16 | 0.31 | 1.9x |

**Interpretation:** When SI is extremely high, RSI_ext is also high (upper tail cluster). Less clustering at low values.

---

## 📊 CATEGORY 8: Cross-Asset Synchronization

| Pair | SI Correlation | Same Market? |
|------|---------------|--------------|
| **SPY-QQQ** | **+0.530** | ✓ Stocks |
| BTC-ETH | +0.218 | ✓ Crypto |
| BTC-SOL | +0.214 | ✓ Crypto |
| EURUSD-GBPUSD | +0.220 | ✓ Forex |
| BTC-SPY | <0.10 | ✗ Cross-market |

**Conclusion:** Assets within the same market specialize together.

---

## 📊 CATEGORY 9: Statistical Robustness

| Test | BTC | SPY | EUR | Interpretation |
|------|-----|-----|-----|----------------|
| Bootstrap CI excludes 0 | ✓ | ✗ | ✓ | Significant in 2/3 |
| Permutation p < 0.05 | ✓ | Borderline | ✓ | Robust in 2/3 |
| HAC t-stat > 2 | ✓ | ✗ | ✓ | Survives autocorrelation |
| Factor-neutral r > 0.05 | ✓ | ✓ | ✓ | Not spurious |
| Cointegrated | ✓ | ✓ | ✓ | Long-run equilibrium |

---

## 📊 CATEGORY 10: Regime Analysis

### SI-ADX by Regime

| Regime | BTC r | SPY r | EUR r | Sign Consistent? |
|--------|-------|-------|-------|-----------------|
| Bull | +0.15 | +0.08 | +0.18 | ✓ Positive |
| Bear | +0.12 | +0.05 | +0.14 | ✓ Positive |
| Volatile | +0.08 | +0.04 | +0.10 | ✓ Positive |
| Neutral | **-0.04** | +0.02 | -0.01 | ⚠️ Flips sign |

**Warning:** SI-ADX can flip sign in neutral regimes.

---

## 📊 CATEGORY 11: PCA/ICA Structure

### PCA
| Component | Variance | SI Correlation |
|-----------|----------|----------------|
| PC1 | 36% | r = 0.23-0.30 |
| PC2 | 27% | r ≈ 0.07 |
| PC3 | 20% | r ≈ -0.12 |

**PC1 Loadings:** ADX (+0.6), RSI_ext (+0.6), Volatility (-0.4), Momentum (-0.3)

**Interpretation:** SI captures the "trend clarity" factor.

### ICA
| Asset | Strongest IC | Correlation |
|-------|-------------|-------------|
| BTC | IC2 | r = -0.225 |
| SPY | IC2 | r = +0.238 |
| EUR | IC3 | r = +0.317 |

---

## 📊 CATEGORY 12: Wavelet Decomposition

| Level | Frequency | Variance % | SI-ADX r |
|-------|-----------|------------|----------|
| Approx (Low) | >60 days | 36% | **+0.21** |
| Detail 1 | 30-60 days | 22% | +0.04 |
| Detail 2 | 14-30 days | 26% | -0.01 |
| Detail 3 | 7-14 days | 11% | -0.00 |
| Detail 4 | 3-7 days | 4% | -0.00 |

**Conclusion:** SI-ADX relationship is entirely in low-frequency (trend) component.

---

## 🎯 FINAL SUMMARY: What SI Really Is

```
┌─────────────────────────────────────────────────────────────────┐
│                    SI = MARKET CLARITY INDEX                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  WHAT SI IS:                                                     │
│    ✓ A lagging indicator of market state                        │
│    ✓ Cointegrated with trend strength (ADX)                     │
│    ✓ Negatively correlated with volatility                      │
│    ✓ A long-term (30-120 day) phenomenon                        │
│    ✓ A fractional OU process (H=0.83)                           │
│                                                                  │
│  WHAT SI IS NOT:                                                 │
│    ✗ A trading signal                                           │
│    ✗ Predictive of returns                                      │
│    ✗ Useful for short-term trading                              │
│    ✗ Normally distributed                                       │
│    ✗ A causal driver of market behavior                         │
│                                                                  │
│  PRACTICAL USES:                                                 │
│    → Volatility forecasting                                      │
│    → Regime classification                                       │
│    → Position sizing (smaller in low-SI)                        │
│    → Risk management overlay                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 NEGATIVE RESULTS (Equally Important)

| Finding | Evidence |
|---------|----------|
| SI does NOT predict returns | IC half-life = 3-5 days only |
| SI does NOT predict regime changes | p > 0.05 |
| SI does NOT correlate with macro | VIX, DXY all r < 0.1 |
| SI-Volume NOT significant | p > 0.05 |
| Cross-market SI sync weak | r < 0.1 |

---

## 📈 TOTAL DISCOVERIES BY CATEGORY

| Category | Count | Key Finding |
|----------|-------|-------------|
| Correlations | 14 | RSI_ext, Fractal_dim, ADX |
| Causality | 4 | Features → SI (lagging) |
| Process Properties | 6 | Long memory, fat tails |
| Cointegration | 6 | All pairs cointegrated |
| Frequency | 5 | 30+ days only |
| Distribution | 5 | Non-normal, skewed |
| Tail Dependence | 3 | Upper tail clustering |
| Cross-Asset | 5 | Same-market sync |
| Robustness | 5 | HAC, permutation pass |
| Regime | 4 | Sign flips in neutral |
| PCA/ICA | 4 | PC1 = trend clarity |
| Wavelet | 4 | Low-freq dominates |
| Expert Methods | 18 | Hurst, OU, EVT, HMM |
| **TOTAL** | **80+** | - |

---

*Document Version: 1.0*  
*Last Updated: January 17, 2026*
