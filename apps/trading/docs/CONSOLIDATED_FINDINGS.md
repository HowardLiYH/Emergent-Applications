# Consolidated Findings & Corrections

**Date:** January 17, 2026  
**Status:** Phase 1 Complete - All findings verified

---

## ⚠️ CORRECTIONS TO PREVIOUS ASSUMPTIONS

### Correction 1: SI is NOT Predictive
| Previous Assumption | Corrected Understanding | Evidence |
|---------------------|------------------------|----------|
| SI predicts future returns | SI **lags** market state | Transfer Entropy: ADX→SI ratio=0.6 |
| SI → Volatility | **Volatility → SI** | Granger p<0.05 |
| SI can generate alpha | SI is a risk indicator only | IC half-life = 3-5 days |

### Correction 2: SI-ADX is Long-Term Only
| Previous Assumption | Corrected Understanding | Evidence |
|---------------------|------------------------|----------|
| SI-ADX works at all frequencies | Only at 30+ day frequency | Short-term r = -0.05 |
| 7-day SI window is optimal | **Window varies by asset** | BTC=14d, ETH=7d, SOL=30d |
| Daily SI changes are meaningful | Daily changes are **noise** | Detail level r ≈ 0 |

### Correction 3: Relationship is Regime-Dependent
| Previous Assumption | Corrected Understanding | Evidence |
|---------------------|------------------------|----------|
| SI-ADX always positive | **Flips sign in neutral regime** | BTC neutral: r=-0.04 |
| Correlation is constant | Varies 70-83% positive windows | Rolling beta analysis |

---

## ✅ VERIFIED FINDINGS

### Category 1: Correlation Structure

| Finding | Metric | Value | Cross-Asset | Confidence |
|---------|--------|-------|-------------|------------|
| RSI Extremity is #1 correlate | Spearman r | +0.243 | 9/9 ✓ | 95% CI [0.15, 0.33] |
| Fractal Dimension is #2 | Spearman r | -0.231 | 3/3 ✓ | Significant |
| Directional Consistency #3 | Spearman r | +0.206 | 3/3 ✓ | Significant |
| ADX correlation | Spearman r | +0.127 | 9/9 ✓ | 95% CI [0.02, 0.22] |
| Volatility is negative | Spearman r | -0.131 | 3/3 ✓ | Significant |

### Category 2: Statistical Properties

| Property | Value | Asset | Method |
|----------|-------|-------|--------|
| SI Mean | 0.019 | BTC | Distribution |
| SI Std | 0.003 | BTC | Distribution |
| SI Skewness | +0.57 | BTC | Distribution |
| SI Persistence | 4-5 days | All | Run-length analysis |
| SI Half-life | 3 days | All | Autocorrelation |

### Category 3: Causality & Information Flow

| Direction | Test | Result | p-value |
|-----------|------|--------|---------|
| Volatility → SI | Granger | Significant | <0.05 |
| ADX → SI | Transfer Entropy | TE ratio=0.6 | - |
| RSI_ext → SI | Granger (SOL) | Significant | <0.05 |
| SI → Features | All tests | **Not significant** | >0.05 |

### Category 4: Cointegration

| Pair | Engle-Granger Stat | p-value | Status |
|------|-------------------|---------|--------|
| SI-ADX (BTC) | -10.2 | <0.0001 | ✅ Cointegrated |
| SI-ADX (SPY) | -10.3 | <0.0001 | ✅ Cointegrated |
| SI-ADX (EUR) | -13.4 | <0.0001 | ✅ Cointegrated |
| SI-RSI_ext (All) | -10 to -14 | <0.0001 | ✅ Cointegrated |

### Category 5: Frequency Domain

| Frequency Band | Period | SI-ADX r | Interpretation |
|----------------|--------|----------|----------------|
| Very Short | 3-7 days | -0.05 | NEGATIVE |
| Short | 7-14 days | ~0 | No relationship |
| Medium | 14-30 days | ~0 | No relationship |
| **Long** | 30-60 days | +0.24 | Strong positive |
| **Very Long** | 60-120 days | +0.35 | Strongest |

### Category 6: Tail Dependence

| Feature | Lower Tail λ | Upper Tail λ | Asymmetry |
|---------|-------------|-------------|-----------|
| SI-RSI_ext (BTC) | 0.13 | 0.31 | 2.4x |
| SI-RSI_ext (SPY) | 0.12 | 0.39 | 3.3x |
| SI-RSI_ext (EUR) | 0.16 | 0.31 | 1.9x |

### Category 7: Cross-Asset Synchronization

| Pair | SI Correlation | Market Type |
|------|---------------|-------------|
| SPY-QQQ | +0.530 | Same (Stocks) |
| BTC-ETH | +0.218 | Same (Crypto) |
| BTC-SOL | +0.214 | Same (Crypto) |
| EURUSD-GBPUSD | +0.220 | Same (Forex) |
| BTC-SPY | <0.1 | Cross-market |

### Category 8: PCA Structure

| Component | Variance Explained | SI Correlation |
|-----------|-------------------|----------------|
| PC1 | 36% | r = 0.23-0.30 |
| PC2 | 27% | r ≈ 0.07 |
| PC3 | 20% | r ≈ -0.12 |

**PC1 Loadings:** ADX (+0.6), RSI_ext (+0.6), Volatility (-0.4), Momentum (-0.3)

---

## 📊 DISTRIBUTION SUMMARY

| Statistic | BTC | SPY | EURUSD |
|-----------|-----|-----|--------|
| Mean SI | 0.0192 | 0.0201 | 0.0195 |
| Std SI | 0.0028 | 0.0031 | 0.0025 |
| Skewness | +0.57 | +0.21 | +0.34 |
| Kurtosis | 0.8 | -0.2 | 0.3 |
| Min SI | 0.010 | 0.012 | 0.011 |
| Max SI | 0.030 | 0.029 | 0.027 |

---

## 🎯 KEY TAKEAWAYS

1. **SI = Market Clarity Index** - Not a trading signal
2. **Lagging indicator** - Driven by market, doesn't predict
3. **Long-term only** - 30+ day relationship
4. **Cointegrated with ADX** - Long-run equilibrium
5. **Regime-dependent** - Sign flips in neutral markets
6. **Cross-asset sync** - Same-market assets correlate

---

## 📝 OPEN QUESTIONS FOR PHASE 2

1. Can we formalize SI as entropy of regime distribution?
2. Is SI useful for volatility forecasting (not return prediction)?
3. How to use SI for position sizing in risk management?
4. What's the theoretical basis for the cointegration?
5. Why does SI-ADX flip in neutral regimes?
