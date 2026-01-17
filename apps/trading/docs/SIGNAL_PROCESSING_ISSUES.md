# Signal Processing Issues Diagnosis

**Date**: January 17, 2026
**Status**: CRITICAL - Must Fix Before Proceeding

---

## 🚨 ROOT CAUSE: HOURLY PARAMETERS APPLIED TO DAILY DATA

All code was designed for **HOURLY** data but we switched to **DAILY** data for cross-market comparison. This broke everything.

---

## Issue 1: Feature Rolling Windows Are Wrong

### The Problem

Features use hardcoded rolling windows designed for HOURLY data:

| Code | Intended (Hourly) | Actual (Daily) |
|------|-------------------|----------------|
| `rolling(24)` | 24 hours = 1 day | 24 days |
| `rolling(168)` | 168 hours = 7 days | 168 days (~5.6 months) |
| `rolling(720)` | 720 hours = 30 days | 720 days (~2.9 years!) |

### Impact

With only 1,826 daily rows (5 years):
- `volatility_30d` → actually `volatility_2.9years`
- Features need 720 rows warm-up → 40% of data unusable
- Most risk features (`var_95_30d`, `cvar_95_30d`, `sharpe_ratio_30d`) are computing multi-year metrics

### Evidence

```
volatility_30d = returns.rolling(720).std()  # ← 720 DAYS, not 30 days!
```

---

## Issue 2: SI Window Is Wrong

### The Problem

SI computation uses `window=168`:

```python
si = population.compute_si_timeseries(train, window=168)
```

This was designed for 168 HOURS (7 days). On daily data:
- SI needs 168 DAYS warm-up (~5.6 months)
- Validation set (272 rows) has only 272 - 168 = 104 usable rows
- After feature alignment, maybe 50-80 rows remain

### Impact

- Training: 1272 - 168 = 1104 SI values (OK)
- Validation: 272 - 168 = 104 SI values (BARELY USABLE)
- Test: 274 - 168 = 106 SI values (BARELY USABLE)

---

## Issue 3: Regime Classification Is Wrong

### The Problem

```python
def classify_regime(self, data: pd.DataFrame, idx: int, lookback: int = 168):
```

On daily data, looking back 168 DAYS (~8 months) for regime classification!

### Impact

- Regimes change too slowly (8-month lookback smooths everything)
- High-volatility detection becomes rare
- Specialization dynamics are dampened

---

## Issue 4: Strategy Parameters Are Wrong

### The Problem

Strategies have hardcoded lookback periods:

```python
DEFAULT_STRATEGIES = [
    MomentumStrategy(lookback=168),  # 168 hours = 7 days → BUT 168 days = 8 months!
    MomentumStrategy(lookback=72),   # 72 hours = 3 days → BUT 72 days = 3 months
    MeanReversionStrategy(lookback=24, threshold=2.0),  # 24 hours → 24 days
    MeanReversionStrategy(lookback=48, threshold=1.5),  # 48 hours → 48 days
    BreakoutStrategy(lookback=48),   # 48 hours → 48 days
    BreakoutStrategy(lookback=96),   # 96 hours → 96 days
]
```

### Impact

- Momentum signals based on 8-month trends (way too slow)
- Mean reversion signals based on 24-48 day extremes
- Competition dynamics are sluggish

---

## Issue 5: Validation Confirmation Rate = 0%

### The Problem

`validate_on_set()` returns 0% confirmation because:

1. Validation has only 272 daily rows
2. After `start_idx=50` competition warm-up: 222 rows
3. After SI `window=168` warm-up: 54 rows
4. After feature warm-up (up to 720): **NEGATIVE or ~0 rows**

### Evidence

```python
def validate_on_set(data, candidates, si_window, n_agents, frequency):
    if len(candidates) == 0 or len(data) < 300:  # ← 272 < 300!
        return {'confirmation_rate': 0}  # ← RETURNS HERE!
```

The validation set (272 rows) is **less than 300 rows**, so it **immediately returns 0%**!

---

## Issue 6: Correlation Signs Might Be Inverted

### The Problem

On daily data with 168-day SI window:
- SI captures VERY long-term specialization trends
- Features like `volatility_24h` (actually 24 days) also capture long-term patterns
- Correlations might be correct but NOT what we intended to measure

### Impact

We're measuring: "Does 6-month SI correlate with 1-month volatility?"
We wanted to measure: "Does 1-week SI correlate with 1-day volatility?"

---

## Issue 7: Backtest Uses Wrong Threshold

### The Problem

```python
si_pct = si.rolling(window=min(720, len(si) // 2)).rank(pct=True)
```

Using 720-period rolling percentile on daily data = 2.9 years!

### Impact

- SI percentile changes very slowly
- Trading signals are rare
- Backtest performance is mostly noise

---

## 📊 Summary of Parameter Mismatches

| Parameter | Designed For | Current Data | Mismatch Factor |
|-----------|--------------|--------------|-----------------|
| SI window (168) | 168 hours (7d) | 168 days | **24x** |
| Feature window (24) | 24 hours (1d) | 24 days | **24x** |
| Feature window (168) | 168 hours (7d) | 168 days | **24x** |
| Feature window (720) | 720 hours (30d) | 720 days | **24x** |
| Regime lookback (168) | 168 hours (7d) | 168 days | **24x** |
| Strategy lookback | 24-168 hours | 24-168 days | **24x** |
| Val set size check | 300 hours OK | 272 days FAIL | N/A |

---

## ✅ SOLUTION: Frequency-Aware Parameters

### Option A: Use Hourly Data Only (Simplest)

Keep using hourly crypto data (43,786 rows) for all analysis.
- Pros: Code works as designed
- Cons: Can't compare crypto to daily forex/stocks

### Option B: Adjust All Parameters for Daily Data (Recommended)

Create frequency-aware version:

```python
# For hourly data
HOURLY_PARAMS = {
    'si_window': 168,        # 7 days
    'feature_1d': 24,        # 1 day
    'feature_7d': 168,       # 7 days
    'feature_30d': 720,      # 30 days
    'regime_lookback': 168,  # 7 days
    'min_val_size': 300,     # 300 hours
}

# For daily data
DAILY_PARAMS = {
    'si_window': 7,          # 7 days
    'feature_1d': 1,         # 1 day
    'feature_7d': 7,         # 7 days
    'feature_30d': 30,       # 30 days
    'regime_lookback': 7,    # 7 days
    'min_val_size': 30,      # 30 days
}
```

---

## 🔧 Files That Need Fixing

1. **`src/analysis/features.py`** - Add frequency parameter
2. **`src/competition/niche_population.py`** - Add frequency-aware regime classification
3. **`src/agents/strategies.py`** - Add frequency-aware strategy parameters
4. **`experiments/run_fixed_analysis.py`** - Use frequency-aware parameters
5. **`src/data/loader_v2.py`** - Return frequency with data

---

## 🎯 Priority: Fix Order

1. **HIGH**: Fix feature calculator (frequency-aware rolling windows)
2. **HIGH**: Fix SI window (7 for daily, 168 for hourly)
3. **HIGH**: Fix validation threshold (30 for daily, 300 for hourly)
4. **MEDIUM**: Fix regime classification
5. **MEDIUM**: Fix strategy lookbacks
6. **LOW**: Fix backtest percentile windows

---

## ❓ Why Did We See 238 "Meaningful" Correlations?

Because we were correlating:
- 6-month SI trends with 2-year volatility patterns
- Both are slow-moving, so they naturally correlate
- But this is NOT what we intended to measure!

The correlations are **statistically real** but **semantically wrong**.

---

**Next Step**: Implement frequency-aware parameters across all modules.
