# Methodology Audit: SI Correlation Testing Plan

**Date**: January 17, 2026
**Purpose**: Identify methodological issues, gaps, and improvements needed

---

## 🚨 Executive Summary

| Category | Issues Found | Severity | Status |
|----------|--------------|----------|--------|
| **Statistical** | 8 issues | 🔴 High | Needs fixing |
| **Experimental Design** | 6 issues | 🔴 High | Needs fixing |
| **Data Handling** | 5 issues | 🟡 Medium | Needs fixing |
| **Feature Engineering** | 4 issues | 🟡 Medium | Needs attention |
| **Causality** | 3 issues | 🔴 High | Needs fixing |
| **Reproducibility** | 3 issues | 🟢 Low | Minor fixes |
| **Generalization** | 4 issues | 🟡 Medium | Acknowledge |

**Overall Assessment**: The plan has good ideas but several critical methodological gaps that must be addressed before running experiments.

---

## 🔴 Critical Issues (Must Fix)

### Issue 1: No Train/Test Split = Overfitting Risk

**Problem**: We plan to run ONE backtest on 6-12 months of data, compute correlations, and interpret. This is a recipe for finding spurious correlations.

**Why it matters**: With 70+ features, we WILL find significant correlations by chance. Without out-of-sample validation, we can't distinguish real patterns from noise.

**Solution**:
```
CURRENT PLAN (WRONG):
All Data → Compute SI → Correlate → Interpret

FIXED PLAN:
├── Training Set (70%) → Compute SI → Correlate → Find top features
├── Validation Set (15%) → Test top features → Confirm patterns
└── Test Set (15%) → Final validation → Report results
```

**Implementation**:
```python
# Split data temporally (NOT randomly - time series!)
train_end = int(len(data) * 0.7)
val_end = int(len(data) * 0.85)

train_data = data[:train_end]      # Discovery
val_data = data[train_end:val_end]  # Confirmation
test_data = data[val_end:]          # Final test

# Only report correlations that hold in BOTH train and validation
```

---

### Issue 2: Time Series Autocorrelation Ignored

**Problem**: Time series data has autocorrelation. Standard correlation tests assume independence. Our p-values are WRONG.

**Why it matters**: If SI at time t is correlated with SI at time t+1 (which it likely is), then having 1000 hourly observations ≠ having 1000 independent observations. Our effective sample size is much smaller.

**Solution**:
```python
# WRONG: Standard Pearson correlation
r, p = pearsonr(si, feature)  # Assumes independence

# RIGHT: Adjust for autocorrelation
def adjusted_correlation_test(x, y):
    """Correlation test with Newey-West standard errors."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    # Standardize
    x_std = (x - x.mean()) / x.std()
    y_std = (y - y.mean()) / y.std()

    # OLS with HAC standard errors
    model = OLS(y_std, add_constant(x_std))
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 10})

    return results.params[1], results.pvalues[1]

# OR: Block bootstrap for confidence intervals
def block_bootstrap_correlation(x, y, block_size=24, n_bootstrap=1000):
    """Block bootstrap preserving autocorrelation structure."""
    correlations = []
    n = len(x)
    n_blocks = n // block_size

    for _ in range(n_bootstrap):
        # Sample blocks with replacement
        block_indices = np.random.choice(n_blocks, n_blocks, replace=True)
        boot_x = np.concatenate([x[i*block_size:(i+1)*block_size] for i in block_indices])
        boot_y = np.concatenate([y[i*block_size:(i+1)*block_size] for i in block_indices])
        correlations.append(np.corrcoef(boot_x, boot_y)[0,1])

    return np.percentile(correlations, [2.5, 97.5])  # 95% CI
```

---

### Issue 3: Multiple Testing Underestimated

**Problem**: We test 70+ features. FDR correction helps, but we also:
- Test multiple SI variants (8 types)
- Test multiple windows (1d, 7d, 30d)
- Test multiple correlation types (Pearson, Spearman, MI, Granger)
- Test regime-stratified correlations

**True number of tests**: ~70 features × 8 SI variants × 3 windows × 4 methods × 3 regimes = **20,160 tests**

**Solution**:
```python
# Pre-register primary analysis
PRIMARY_TESTS = {
    'si_variant': 'si_rolling_7d',  # Pick ONE SI variant a priori
    'window': '7d',                   # Pick ONE window
    'method': 'spearman',             # Pick ONE method (robust to outliers)
    'regime': 'all',                  # Don't stratify initially
}

# All other analyses are EXPLORATORY (not for hypothesis testing)
EXPLORATORY_TESTS = [...]  # Document these separately

# Apply correction only to primary tests (70 features)
# Report exploratory as "hypothesis-generating, not confirmatory"
```

---

### Issue 4: Circular Reasoning Risk

**Problem**: SI is computed from agent behavior. Agent behavior is influenced by market features. We're correlating SI with market features.

```
Market Features → Agent Behavior → SI
       ↓
SI correlated with Market Features?
       ↓
Obviously yes, because SI DERIVES from market-driven behavior!
```

**Why it matters**: Finding that SI correlates with volatility might just mean "agents react to volatility" - not that SI is useful.

**Solution**:
```python
# Test PREDICTIVE power, not concurrent correlation
# SI at time t should predict feature at time t+k (not t)

def test_predictive_power(si, feature, lags=[1, 6, 24]):
    """Test if SI predicts FUTURE feature values."""
    results = []
    for lag in lags:
        # SI today vs Feature tomorrow
        si_today = si.iloc[:-lag]
        feature_future = feature.iloc[lag:]

        r, p = spearmanr(si_today, feature_future)
        results.append({'lag': lag, 'r': r, 'p': p})
    return results

# Only consider SI valuable if it PREDICTS (leads) features
# Concurrent correlation may be circular
```

---

### Issue 5: No Power Analysis

**Problem**: We don't know if we have enough data to detect meaningful correlations.

**Why it matters**: With 6 months of hourly data (~4,380 points), after adjusting for autocorrelation, effective N might be ~100-200. What's the minimum detectable effect size?

**Solution**:
```python
from statsmodels.stats.power import TTestPower

def minimum_detectable_effect(n_eff, alpha=0.05, power=0.8):
    """What's the smallest correlation we can reliably detect?"""
    analysis = TTestPower()
    effect_size = analysis.solve_power(
        effect_size=None,
        nobs=n_eff,
        alpha=alpha,
        power=power,
        alternative='two-sided'
    )
    # Convert to correlation
    r = effect_size / np.sqrt(effect_size**2 + 4)
    return r

# Example: If effective N = 100, minimum detectable r ≈ 0.28
# Anything smaller is underpowered
```

**Document in report**:
```markdown
## Power Analysis
- Raw sample size: 4,380 hourly observations
- Effective sample size (adjusted for autocorrelation): ~150
- Minimum detectable correlation (α=0.05, power=0.8): r = 0.23
- Correlations weaker than r = 0.23 may be underpowered
```

---

### Issue 6: Look-Ahead Bias in Features

**Problem**: Some features use future information:
- `next_day_return`: By definition looks ahead
- `max_drawdown`: Uses future data to find recovery
- `drawdown_recovery_days`: Requires knowing when recovery happens

**Why it matters**: Can't use these features for SI prediction if they contain future info.

**Solution**:
```python
# Categorize features by temporal nature
FEATURES_NO_LOOKAHEAD = [
    'volatility', 'trend_strength', 'return_entropy',
    'volume', 'rsi', 'adx', 'agent_correlation', ...
]

FEATURES_WITH_LOOKAHEAD = [
    'next_day_return',  # Explicitly for prediction testing
    'next_day_volatility',
]

FEATURES_AMBIGUOUS = [
    'max_drawdown',  # Need to clarify: rolling or expanding?
    'drawdown_recovery_days',  # Clarify calculation
]

# For rolling metrics, specify:
# "max_drawdown_30d" = max drawdown in PAST 30 days (no lookahead)
# NOT max drawdown that will occur (lookahead!)
```

---

### Issue 7: Confounding Variables

**Problem**: Two variables might both correlate with SI not because they're related, but because they're both driven by a third variable.

**Example**:
```
Time → More training → Higher SI
Time → More data → Better features

SI correlates with feature quality, but both are just "time effects"
```

**Solution**:
```python
# Control for obvious confounders
CONFOUNDERS = ['time_index', 'training_iteration', 'cumulative_data']

def partial_correlation(si, feature, confounders):
    """Correlation controlling for confounders."""
    from scipy.stats import pearsonr
    import statsmodels.api as sm

    # Residualize SI
    si_model = sm.OLS(si, sm.add_constant(confounders)).fit()
    si_resid = si_model.resid

    # Residualize feature
    feat_model = sm.OLS(feature, sm.add_constant(confounders)).fit()
    feat_resid = feat_model.resid

    # Correlate residuals
    return pearsonr(si_resid, feat_resid)

# Report both raw and partial correlations
```

---

### Issue 8: Non-Stationarity

**Problem**: Financial time series are non-stationary. Correlations computed on non-stationary data can be spurious.

**Why it matters**: Two random walks can show strong correlation (spurious regression problem).

**Solution**:
```python
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series, name):
    """Check if series is stationary."""
    result = adfuller(series.dropna())
    p_value = result[1]
    is_stationary = p_value < 0.05

    if not is_stationary:
        print(f"WARNING: {name} is non-stationary (p={p_value:.3f})")
        print(f"Consider using differenced series or returns")

    return is_stationary

# For non-stationary series, use:
# 1. First differences: diff(SI) vs diff(feature)
# 2. Returns: SI_return vs feature_return
# 3. Cointegration test if both are I(1)
```

---

## 🟡 Medium Issues (Should Fix)

### Issue 9: Feature Multicollinearity

**Problem**: Many features are highly correlated with each other:
- `volatility_24h` ↔ `volatility_7d` ↔ `volatility_30d`
- `sharpe_ratio` ↔ `sortino_ratio`
- `rsi` ↔ `fear_greed_proxy`

**Why it matters**: Reporting "SI correlates with 10 features" is misleading if those 10 features are really just 3 independent concepts.

**Solution**:
```python
# Cluster correlated features and pick one representative
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def cluster_features(features, threshold=0.7):
    """Group features that are >70% correlated."""
    corr_matrix = features.corr().abs()
    distance_matrix = 1 - corr_matrix
    linkage_matrix = linkage(squareform(distance_matrix), method='average')
    clusters = fcluster(linkage_matrix, t=1-threshold, criterion='distance')

    # Pick one representative per cluster
    representatives = {}
    for cluster_id in np.unique(clusters):
        cluster_features = features.columns[clusters == cluster_id]
        # Pick feature with highest variance (most informative)
        variances = features[cluster_features].var()
        representatives[cluster_id] = variances.idxmax()

    return representatives

# Report: "After clustering, 70 features reduce to 25 independent concepts"
```

---

### Issue 10: Regime Definition Circular

**Problem**: We plan to test if SI correlates with "regime" features. But how do we define regimes?
- If we use volatility thresholds → We're testing SI vs volatility (already done)
- If we use HMM → Different HMM specifications give different regimes

**Solution**:
```python
# Use MULTIPLE regime definitions and check consistency
REGIME_DEFINITIONS = {
    'volatility_based': lambda data: define_vol_regimes(data),
    'trend_based': lambda data: define_trend_regimes(data),
    'hmm_2state': lambda data: fit_hmm(data, n_states=2),
    'hmm_3state': lambda data: fit_hmm(data, n_states=3),
    'structural_breaks': lambda data: detect_breaks(data),
}

# Only trust findings that hold across MULTIPLE definitions
```

---

### Issue 11: Missing Baseline Comparison

**Problem**: If we find SI correlates with volatility (r=0.3), is that good? We have no baseline.

**Solution**:
```python
# Compare SI correlations to RANDOM SI
def compute_random_baseline(features, n_permutations=1000):
    """What correlations do we get with random SI?"""
    random_correlations = []
    for _ in range(n_permutations):
        random_si = np.random.permutation(len(features))
        random_si = pd.Series(random_si, index=features.index)
        for col in features.columns:
            r = np.corrcoef(random_si, features[col])[0,1]
            random_correlations.append({'feature': col, 'r': r})

    # 95th percentile of random correlations
    baseline = pd.DataFrame(random_correlations).groupby('feature')['r'].quantile(0.95)
    return baseline

# Real SI correlation must exceed random baseline to be meaningful
```

---

### Issue 12: Single Asset Class

**Problem**: Only testing on crypto (BTC, ETH, SOL). Results may not generalize to:
- Equities
- Forex
- Commodities
- Different market structures

**Solution**:
```python
# Acknowledge limitation explicitly
LIMITATIONS = """
## Generalization Limitations
- All tests conducted on cryptocurrency data (BTC, ETH, SOL)
- Crypto markets have:
  - 24/7 trading
  - High retail participation
  - Different liquidity dynamics
- Results may not generalize to traditional markets
- Future work: Test on SPY, FX pairs, commodities
"""
```

---

### Issue 13: Survivorship Bias

**Problem**: BTC and ETH are "winners" - they survived and grew. Testing on survivors introduces bias.

**Why it matters**: Our strategies might only work on assets that went up. Would fail on assets that crashed.

**Solution**:
```python
# Include "failed" or poor-performing assets
TEST_ASSETS = {
    'survivors': ['BTC', 'ETH', 'SOL'],
    'strugglers': ['XRP', 'LTC', 'DOGE'],  # Less stellar performance
    # If possible, include delisted assets
}

# Test if SI findings hold on both groups
```

---

## 🟢 Minor Issues (Document/Acknowledge)

### Issue 14: Reproducibility

**Problem**: Need to document all parameters for reproducibility.

**Solution**:
```python
CONFIG = {
    'random_seed': 42,
    'data_source': 'bybit',
    'date_range': ('2025-01-01', '2025-12-31'),
    'assets': ['BTC', 'ETH', 'SOL'],
    'granularity': '1h',
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'si_window': 24,
    'feature_windows': [24, 168, 720],
    'fdr_alpha': 0.05,
    'min_samples_for_correlation': 30,
}

# Save config with results
import json
with open('results/config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)
```

---

### Issue 15: Effect Size Interpretation

**Problem**: We report correlations but don't interpret what they mean practically.

**Solution**:
```markdown
## Effect Size Guidelines

| Correlation | Interpretation | R² (Variance Explained) |
|-------------|---------------|-------------------------|
| r = 0.1     | Trivial       | 1%                      |
| r = 0.3     | Small         | 9%                      |
| r = 0.5     | Medium        | 25%                     |
| r = 0.7     | Large         | 49%                     |

**Practical threshold**: r > 0.3 to be actionable
**Minimum for publication**: r > 0.2 with p < 0.05 (FDR-corrected)
```

---

### Issue 16: Reporting Standards

**Problem**: Need clear reporting standards for the final report.

**Solution**:
```markdown
## Reporting Checklist

For each significant finding, report:
- [ ] Effect size (r) and confidence interval
- [ ] P-value (raw and FDR-corrected)
- [ ] Sample size and effective sample size
- [ ] Train/validation/test performance separately
- [ ] Visualization (scatter plot with CI)
- [ ] Robustness checks (multiple windows, methods)
- [ ] Comparison to random baseline
- [ ] Partial correlation (controlling for confounders)
- [ ] Interpretation in plain language
```

---

## 📋 Revised Test Plan

Based on this audit, here's the corrected methodology:

### Phase 0: Pre-Registration (NEW)

```python
# Document BEFORE looking at data:
PRE_REGISTRATION = {
    'primary_hypothesis': "SI correlates with at least one feature",
    'primary_si_variant': 'si_rolling_7d',
    'primary_correlation_method': 'spearman',
    'primary_significance_level': 0.05,
    'correction_method': 'fdr_bh',
    'minimum_effect_size': 0.2,
    'train_val_test_split': [0.7, 0.15, 0.15],
}
```

### Phase 1: Data Preparation (UPDATED)

```python
# 1. Split data FIRST (before any analysis)
train, val, test = split_temporal(data, [0.7, 0.15, 0.15])

# 2. Check stationarity
for feature in features:
    check_stationarity(feature)

# 3. Estimate effective sample size
n_eff = estimate_effective_n(data)

# 4. Power analysis
min_detectable_r = power_analysis(n_eff)
print(f"Minimum detectable correlation: {min_detectable_r}")
```

### Phase 2: Discovery (Training Set Only)

```python
# Only use training data for discovery
si_train = compute_si(train)
features_train = compute_features(train)

# Compute correlations with proper standard errors
correlations = []
for feature in features_train.columns:
    r, p = adjusted_correlation_test(si_train, features_train[feature])
    correlations.append({'feature': feature, 'r': r, 'p': p})

# Apply FDR correction
corrected = apply_fdr(correlations)

# Identify candidates (FDR < 0.05 AND |r| > min_detectable_r)
candidates = corrected[
    (corrected['p_fdr'] < 0.05) &
    (abs(corrected['r']) > min_detectable_r)
]
```

### Phase 3: Confirmation (Validation Set)

```python
# Test ONLY the candidates on validation data
si_val = compute_si(val)
features_val = compute_features(val)

confirmed = []
for feature in candidates['feature']:
    r, p = adjusted_correlation_test(si_val, features_val[feature])
    if p < 0.05 and np.sign(r) == np.sign(candidates.loc[feature, 'r']):
        confirmed.append(feature)

print(f"Confirmed {len(confirmed)}/{len(candidates)} candidates")
```

### Phase 4: Final Test

```python
# Report performance on held-out test set
si_test = compute_si(test)
features_test = compute_features(test)

final_results = []
for feature in confirmed:
    r, p = adjusted_correlation_test(si_test, features_test[feature])
    final_results.append({
        'feature': feature,
        'train_r': candidates.loc[feature, 'r'],
        'val_r': ...,
        'test_r': r,
        'test_p': p
    })

# Only report features that hold in ALL three sets
```

### Phase 5: Robustness Checks

```python
ROBUSTNESS_CHECKS = [
    'different_si_window',      # 1d, 7d, 30d
    'different_correlation',    # Pearson, Spearman, MI
    'different_regime',         # Trending, Ranging, Volatile
    'different_asset',          # BTC only, ETH only, SOL only
    'partial_correlation',      # Control for time, vol
    'block_bootstrap_ci',       # 95% confidence interval
    'random_baseline',          # Compare to permutation null
]
```

---

## ✅ Updated Checklist

### Before Running Experiments

- [ ] Pre-register primary analysis
- [ ] Split data into train/val/test
- [ ] Check stationarity of all series
- [ ] Estimate effective sample size
- [ ] Conduct power analysis
- [ ] Document all parameters

### During Analysis

- [ ] Use adjusted correlation tests (HAC standard errors)
- [ ] Apply FDR correction
- [ ] Compute partial correlations (control confounders)
- [ ] Test predictive power (SI leads feature)
- [ ] Compare to random baseline

### Before Reporting

- [ ] Confirm findings on validation set
- [ ] Final test on held-out test set
- [ ] Run all robustness checks
- [ ] Document effect sizes and CIs
- [ ] Acknowledge limitations

---

## 📊 Summary of Required Changes

| Current Plan | Required Change | Priority |
|--------------|-----------------|----------|
| Single backtest | Train/val/test split | 🔴 Critical |
| Standard correlation | HAC-adjusted or block bootstrap | 🔴 Critical |
| 70 features tested | Pre-register primary, rest exploratory | 🔴 Critical |
| Concurrent correlation | Test predictive (lagged) correlation | 🔴 Critical |
| No power analysis | Add power analysis | 🔴 Critical |
| No baseline | Compare to random permutation | 🟡 High |
| No confounder control | Partial correlations | 🟡 High |
| Non-stationarity ignored | Test stationarity, use differences | 🟡 High |
| Multicollinear features | Cluster and pick representatives | 🟡 Medium |
| Single asset class | Acknowledge limitation | 🟢 Low |

---

*This audit ensures our methodology is rigorous and findings are credible.*

*Last Updated: January 17, 2026*
