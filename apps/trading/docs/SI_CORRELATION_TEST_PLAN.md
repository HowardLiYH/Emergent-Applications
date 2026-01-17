# SI Correlation Test Plan: 70+ Features

**Date**: January 17, 2026
**Purpose**: Systematic plan to discover what SI actually correlates with

---

## ⚠️ Methodology Audit Applied

See `METHODOLOGY_AUDIT.md` for initial audit (16 issues).
See `EXPERT_PANEL_FINAL_REVIEW.md` for expert review (18 additional recommendations).

### Initial Audit Fixes (16 issues):

| Issue | Fix Applied |
|-------|-------------|
| No train/test split | ✅ 70/15/15 temporal split |
| Autocorrelation ignored | ✅ HAC-adjusted tests, block bootstrap |
| Multiple testing | ✅ Pre-registered primary analysis |
| Circular reasoning | ✅ Test predictive (lagged) correlations |
| No power analysis | ✅ Estimate min detectable effect |
| Look-ahead bias | ✅ Categorized features |
| Confounders | ✅ Partial correlations |
| Non-stationarity | ✅ Stationarity checks |

### Expert Panel Additions (18 recommendations):

| Recommendation | Fix Applied |
|----------------|-------------|
| Effective N formula | ✅ Bartlett/AR(1) specified |
| Rolling validation | ✅ Multiple fold validation |
| Cluster on train only | ✅ No data leakage |
| Liquidity control | ✅ Amihud, volume as confounders |
| Signal decay analysis | ✅ Half-life estimation |
| Negative controls | ✅ Random noise, moon phase |
| Transaction costs | ✅ Sensitivity analysis |
| Publish pre-registration | ✅ GitHub commit before analysis |
| Expand confounders | ✅ Macro, volume, liquidity added |

---

## 🎯 Goal

Discover what SI correlates with using **rigorous methodology**:
1. Pre-register primary hypothesis
2. Train/validate/test split
3. Proper statistical tests for time series
4. Control for confounders
5. Robustness checks

---

## ⚠️ Feature Pipeline Audit: Three Separate Pipelines Required

See `FEATURE_PIPELINE_AUDIT.md` for full analysis. Key finding:

**Not all features can use the same pipeline!**

```
┌─────────────────────────────────────────────────────────────┐
│                    THREE SEPARATE PIPELINES                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PIPELINE 1: DISCOVERY (46 features)                         │
│  Question: "What does SI correlate with?"                    │
│  Features: Market, Risk, Safe Agent, Liquidity               │
│  Method: Spearman + HAC + Bootstrap                          │
│                                                              │
│  PIPELINE 2: PREDICTION (2 features)                         │
│  Question: "Does SI predict future outcomes?"                │
│  Features: next_day_return, next_day_volatility              │
│  Method: Lagged correlation, Granger causality               │
│                                                              │
│  PIPELINE 3: SI DYNAMICS (9 features)                        │
│  Question: "How should we use SI?"                           │
│  Features: dSI/dt, SI_std, SI at different scales            │
│  Method: Time series analysis, NOT correlation discovery     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Features REMOVED from Discovery Pipeline (Circular)

```python
CIRCULAR_FEATURES_REMOVED = [
    # SI-derived (testing SI against itself)
    'dsi_dt', 'si_acceleration', 'si_rolling_std',
    'si_1h', 'si_4h', 'si_1d', 'si_1w', 'si_percentile',

    # Agent features that ARE specialization
    'strategy_concentration',   # This IS SI
    'niche_affinity_entropy',   # This IS SI (different formula)

    # Lookahead (moved to Prediction pipeline)
    'next_day_return', 'next_day_volatility',
]
```

---

## 📋 Phase 0: Pre-Registration (NEW - Before Any Analysis)

```python
PRE_REGISTRATION = {
    # Primary hypothesis (pre-specified)
    'primary_hypothesis': "SI correlates with at least one feature after correction",

    # Primary analysis specification
    'primary_si_variant': 'si_rolling_7d',
    'primary_correlation_method': 'spearman',
    'primary_significance_level': 0.05,
    'correction_method': 'fdr_bh',
    'minimum_effect_size': 0.2,

    # Data split (temporal, not random)
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,

    # What counts as success
    'success_criteria': {
        'train': 'p_fdr < 0.05 AND |r| > 0.2',
        'val': 'same sign AND p < 0.05',
        'test': 'same sign AND p < 0.05'
    }
}

# Document this BEFORE looking at any data!
```

---

## 📊 Phase 1: Data Infrastructure (Day 1-2)

### 1.1 Data Requirements

| Data Type | Source | Timeframe | Granularity |
|-----------|--------|-----------|-------------|
| **Price Data** | Bybit/PopAgent | 12+ months | 1-hour candles |
| **Assets** | BTC, ETH, SOL | Multi-asset | |
| **Features** | Computed | Rolling windows | 1d, 7d, 30d |

### 1.2 Data Split (CRITICAL - Temporal, Not Random)

```python
# IMPORTANT: Split BEFORE any analysis to prevent data leakage
def temporal_split(data, train_pct=0.70, val_pct=0.15, test_pct=0.15):
    """Split time series data temporally."""
    n = len(data)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    return {
        'train': data.iloc[:train_end],      # Discovery
        'val': data.iloc[train_end:val_end],  # Confirmation
        'test': data.iloc[val_end:],          # Final test
    }

# With 12 months of data:
# Train: ~8.4 months (discovery)
# Val: ~1.8 months (confirmation)
# Test: ~1.8 months (final report)
```

### 1.3 Power Analysis (Before Running)

```python
def power_analysis(n_eff, alpha=0.05, power=0.8):
    """What's the minimum correlation we can detect?"""
    from statsmodels.stats.power import TTestPower

    analysis = TTestPower()
    effect = analysis.solve_power(nobs=n_eff, alpha=alpha, power=power)
    min_r = effect / np.sqrt(effect**2 + 4)

    return min_r

# Example with 12 months hourly data:
# Raw N = 8,760
# Effective N (autocorrelation adjusted) ≈ 200-300
# Minimum detectable r ≈ 0.18-0.22
```

### 1.4 Survivorship Bias Mitigation (Issue #13)

```python
# Include assets with different performance profiles
ASSET_GROUPS = {
    'survivors': ['BTC', 'ETH'],        # Strong performers
    'mixed': ['SOL', 'AVAX'],           # Moderate
    'strugglers': ['XRP', 'LTC'],       # Underperformers
}

# Test if findings hold across ALL groups
def check_survivorship_robustness(results_by_group):
    """Verify findings aren't survivorship-biased."""
    findings = {}
    for group, results in results_by_group.items():
        findings[group] = results[results['sig_fdr']]['feature'].tolist()

    # Robust = significant in at least 2/3 groups
    from collections import Counter
    all_features = [f for fs in findings.values() for f in fs]
    counts = Counter(all_features)
    robust = [f for f, c in counts.items() if c >= 2]

    return robust
```

### 1.5 Feature Multicollinearity Handling (Issue #9)

```python
def cluster_and_select_features(features: pd.DataFrame, threshold=0.7):
    """
    Cluster highly correlated features and select representatives.
    Fixes Issue #9: Multicollinearity
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Compute correlation matrix
    corr_matrix = features.corr().abs()

    # Convert to distance
    distance_matrix = 1 - corr_matrix
    np.fill_diagonal(distance_matrix.values, 0)

    # Hierarchical clustering
    linkage_matrix = linkage(squareform(distance_matrix), method='average')
    clusters = fcluster(linkage_matrix, t=1-threshold, criterion='distance')

    # Select representative from each cluster (highest variance)
    representatives = {}
    feature_to_cluster = dict(zip(features.columns, clusters))

    for cluster_id in np.unique(clusters):
        cluster_features = [f for f, c in feature_to_cluster.items() if c == cluster_id]
        variances = features[cluster_features].var()
        rep = variances.idxmax()
        representatives[cluster_id] = {
            'representative': rep,
            'cluster_members': cluster_features,
            'cluster_size': len(cluster_features)
        }

    print(f"Reduced {len(features.columns)} features to {len(representatives)} clusters")
    return representatives, feature_to_cluster
```

### 1.6 Multiple Regime Definitions (Issue #10)

```python
def define_regimes_multiple_ways(data: pd.DataFrame):
    """
    Use multiple regime definitions to avoid circular reasoning.
    Fixes Issue #10: Regime definition circular
    """
    regimes = {}

    # Method 1: Volatility-based (simple)
    vol = data['close'].pct_change().rolling(24).std()
    vol_median = vol.median()
    regimes['vol_based'] = (vol > vol_median).astype(int)

    # Method 2: Trend-based (ADX)
    adx = compute_adx(data, period=14)
    regimes['trend_based'] = (adx > 25).astype(int)  # Trending if ADX > 25

    # Method 3: Hidden Markov Model (2 states)
    from hmmlearn.hmm import GaussianHMM
    returns = data['close'].pct_change().dropna().values.reshape(-1, 1)
    hmm = GaussianHMM(n_components=2, random_state=42)
    hmm.fit(returns)
    regimes['hmm_2state'] = pd.Series(hmm.predict(returns), index=data.index[1:])

    # Method 4: Structural breaks
    # (Simplified: detect when rolling mean changes significantly)
    ma_short = data['close'].rolling(24).mean()
    ma_long = data['close'].rolling(168).mean()
    regimes['structural'] = (ma_short > ma_long).astype(int)

    return regimes

def test_across_regime_definitions(si, feature, regime_definitions):
    """Test if SI-feature correlation holds across different regime definitions."""
    results = {}
    for name, regimes in regime_definitions.items():
        for regime_val in regimes.unique():
            mask = regimes == regime_val
            if mask.sum() < 50:
                continue
            r, p = spearmanr(si[mask], feature[mask])
            results[f"{name}_regime{regime_val}"] = {'r': r, 'p': p}

    # Robust if consistent sign across most definitions
    signs = [np.sign(r['r']) for r in results.values() if not np.isnan(r['r'])]
    is_robust = abs(sum(signs)) > len(signs) * 0.7  # 70% agree on direction

    return results, is_robust
```

### 1.7 Random Baseline Comparison (Issue #11)

```python
def compute_random_baseline(features: pd.DataFrame, n_permutations=1000):
    """
    Compute what correlations we'd get with random SI.
    Fixes Issue #11: Missing baseline
    """
    random_correlations = {col: [] for col in features.columns}

    for _ in range(n_permutations):
        # Random SI (permuted)
        random_si = np.random.permutation(len(features))

        for col in features.columns:
            r = np.corrcoef(random_si, features[col].values)[0, 1]
            random_correlations[col].append(r)

    # 95th percentile for each feature (one-tailed)
    baselines = {
        col: np.percentile(np.abs(corrs), 95)
        for col, corrs in random_correlations.items()
    }

    return baselines

def exceeds_baseline(observed_r, feature, baselines):
    """Check if observed correlation exceeds random baseline."""
    return abs(observed_r) > baselines[feature]
```

### 1.8 Reproducibility Config (Issue #14)

```python
CONFIG = {
    # Reproducibility
    'random_seed': 42,
    'version': '1.0.0',
    'date': '2026-01-17',

    # Data
    'data_source': 'bybit',
    'date_range': ('2025-01-01', '2025-12-31'),
    'assets': ['BTC', 'ETH', 'SOL'],
    'granularity': '1h',

    # Split
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,

    # SI
    'si_window': 24,
    'si_variants': ['instant', 'rolling_1d', 'rolling_7d'],

    # Features
    'feature_windows': [24, 168, 720],
    'feature_clustering_threshold': 0.7,

    # Statistical
    'fdr_alpha': 0.05,
    'min_samples': 100,
    'block_size_hours': 24,
    'n_bootstrap': 1000,
    'n_permutations': 1000,

    # Thresholds
    'min_effect_size': 0.2,
    'confirmation_alpha': 0.05,
}

def save_config(config, path='results/config.json'):
    import json
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {path}")
```

### 1.9 Effect Size Guidelines (Issue #15)

```python
EFFECT_SIZE_GUIDELINES = """
## Effect Size Interpretation

| Correlation (r) | Interpretation | R² (Variance Explained) | Actionability |
|-----------------|----------------|-------------------------|---------------|
| 0.00 - 0.10     | Negligible     | 0-1%                    | Ignore        |
| 0.10 - 0.20     | Weak           | 1-4%                    | Note only     |
| 0.20 - 0.30     | Modest         | 4-9%                    | Investigate   |
| 0.30 - 0.50     | Moderate       | 9-25%                   | Actionable    |
| 0.50 - 0.70     | Strong         | 25-49%                  | High priority |
| 0.70 - 1.00     | Very strong    | 49-100%                 | Verify (suspicious) |

## Decision Thresholds

- **Minimum for reporting**: |r| > 0.15 AND p_fdr < 0.05
- **Minimum for action**: |r| > 0.25 AND confirmed in val/test
- **Suspicious if**: |r| > 0.70 (check for data leakage)
"""
```

### 1.10 Reporting Standards (Issue #16)

```python
REPORTING_CHECKLIST = """
## Required for Each Significant Finding

### Statistical
- [ ] Effect size (r) with 95% confidence interval
- [ ] P-value: raw, Bonferroni-corrected, FDR-corrected
- [ ] Sample size: raw N and effective N
- [ ] Stationarity: both series stationary?

### Validation
- [ ] Train set result
- [ ] Validation set result (confirmation)
- [ ] Test set result (final)
- [ ] Random baseline comparison

### Robustness
- [ ] Different SI windows (1d, 7d, 30d)
- [ ] Different assets (per-asset check)
- [ ] Different time periods (rolling)
- [ ] Partial correlation (confounders controlled)
- [ ] Multiple regime definitions

### Interpretation
- [ ] Plain language description
- [ ] Hypothesis supported (H1-H40)
- [ ] Practical implication
- [ ] Limitations specific to this finding

### Visualization
- [ ] Scatter plot with regression line and CI
- [ ] Time series of both variables
- [ ] Distribution of bootstrap correlations
"""
```

---

## 📊 Phase 1.5: Expert Panel Additions (NEW)

### 1.11 Effective Sample Size Estimation (Expert Recommendation)

```python
def estimate_effective_n(series, method='bartlett'):
    """
    Estimate effective sample size adjusting for autocorrelation.

    From Expert Panel: Prof. Statistics
    """
    n = len(series)

    if method == 'bartlett':
        # Bartlett's formula: N_eff = N / (1 + 2*Σρ_k)
        from statsmodels.tsa.stattools import acf
        nlags = min(50, n // 4)
        acf_vals = acf(series.dropna(), nlags=nlags, fft=True)

        # Sum autocorrelations (significant ones only)
        rho_sum = 0
        for k in range(1, nlags + 1):
            # Stop at first non-significant lag (approximate)
            if abs(acf_vals[k]) < 2 / np.sqrt(n):
                break
            rho_sum += acf_vals[k]

        n_eff = n / (1 + 2 * rho_sum)

    elif method == 'ar1':
        # AR(1) approximation: N_eff = N * (1-ρ)/(1+ρ)
        rho = series.autocorr(lag=1)
        if np.isnan(rho) or abs(rho) >= 1:
            rho = 0
        n_eff = n * (1 - abs(rho)) / (1 + abs(rho))

    return max(10, int(n_eff))  # At least 10

# Example usage:
# n_raw = 8760  # 12 months hourly
# n_eff = estimate_effective_n(si_series)  # Might be ~200-500
# min_r = power_analysis(n_eff)  # Minimum detectable correlation
```

### 1.12 Rolling/Expanding Validation (Expert Recommendation)

```python
def rolling_cross_validation(data, n_folds=5, train_months=6, val_months=1, test_months=1):
    """
    Rolling window cross-validation for robust results.

    From Expert Panel: Prof. ML
    """
    results_by_fold = []
    total_months = 12
    step = (total_months - train_months - val_months - test_months) // (n_folds - 1)

    for fold in range(n_folds):
        start_month = fold * step
        train_end = start_month + train_months
        val_end = train_end + val_months
        test_end = val_end + test_months

        # Get data slices
        train_data = get_months(data, start_month, train_end)
        val_data = get_months(data, train_end, val_end)
        test_data = get_months(data, val_end, test_end)

        # Run analysis
        si_train = compute_si(train_data)
        features_train = compute_features(train_data)

        # Discovery on train
        correlations = run_correlation_analysis(si_train, features_train)
        candidates = correlations[correlations['p_fdr'] < 0.05]

        # Confirm on val
        si_val = compute_si(val_data)
        features_val = compute_features(val_data)
        confirmed = confirm_on_validation(candidates, si_val, features_val)

        # Test on test
        si_test = compute_si(test_data)
        features_test = compute_features(test_data)
        final = test_on_holdout(confirmed, si_test, features_test)

        results_by_fold.append({
            'fold': fold,
            'train_period': f"M{start_month}-M{train_end}",
            'candidates': len(candidates),
            'confirmed': len(confirmed),
            'final': final
        })

    # Aggregate: Only report features significant in majority of folds
    return aggregate_across_folds(results_by_fold)
```

### 1.13 Negative Controls (Expert Recommendation)

```python
def run_negative_controls(si: pd.Series):
    """
    SI should NOT correlate with these meaningless variables.
    If it does, something is wrong.

    From Expert Panel: Research Methodologist
    """
    results = {}
    n = len(si)

    # 1. Pure random noise
    np.random.seed(42)
    random_noise = pd.Series(np.random.randn(n), index=si.index)
    r, p = spearmanr(si, random_noise)
    results['random_noise'] = {'r': r, 'p': p}
    if p < 0.05:
        print("⚠️ WARNING: SI correlates with random noise! Check for bug.")

    # 2. Day of week (no expected relation)
    day_of_week = si.index.dayofweek
    r, p = spearmanr(si, day_of_week)
    results['day_of_week'] = {'r': r, 'p': p}

    # 3. Time index (should be controlled for)
    time_index = np.arange(n)
    r, p = spearmanr(si, time_index)
    results['time_index'] = {'r': r, 'p': p}
    if p < 0.05 and abs(r) > 0.3:
        print("⚠️ WARNING: SI has strong time trend. Control for this!")

    # 4. Shuffled SI (sanity check)
    shuffled_si = si.sample(frac=1, random_state=42)
    r, p = spearmanr(si, shuffled_si)
    results['shuffled_self'] = {'r': r, 'p': p}
    # Should NOT be correlated after shuffling

    return results
```

### 1.14 Liquidity Control (Expert Recommendation)

```python
def compute_liquidity_features(data: pd.DataFrame):
    """
    Liquidity features to control for.

    From Expert Panel: Prof. Quant Finance
    """
    features = {}

    # 1. Volume z-score (relative to recent history)
    vol_mean = data['volume'].rolling(168).mean()
    vol_std = data['volume'].rolling(168).std()
    features['volume_z'] = (data['volume'] - vol_mean) / (vol_std + 1e-8)

    # 2. Amihud illiquidity (price impact per volume)
    returns = data['close'].pct_change().abs()
    features['amihud'] = returns / (data['volume'] + 1e-8)
    features['amihud_log'] = np.log1p(features['amihud'] * 1e9)  # Scale for numerical stability

    # 3. Volume volatility (liquidity stability)
    features['volume_volatility'] = data['volume'].rolling(24).std() / (data['volume'].rolling(24).mean() + 1e-8)

    # 4. Turnover (if market cap available)
    if 'market_cap' in data.columns:
        features['turnover'] = data['volume'] * data['close'] / data['market_cap']

    return features

# Add to confounders
CONFOUNDERS = [
    'time_index',
    'training_iteration',
    'volume_z',           # Liquidity level
    'amihud_log',         # Illiquidity
    'volume_volatility',  # Liquidity stability
]
```

### 1.15 Signal Decay Analysis (Expert Recommendation)

```python
def analyze_signal_decay(si: pd.Series, future_returns: pd.Series, max_lag_hours=168):
    """
    How quickly does SI signal decay?

    From Expert Panel: Hedge Fund Quant
    """
    decay_curve = []

    for lag in range(1, max_lag_hours + 1):
        # SI at time t vs Return at time t+lag
        si_past = si.iloc[:-lag]
        returns_future = future_returns.iloc[lag:]

        # Align indices
        common_idx = si_past.index.intersection(returns_future.index)
        if len(common_idx) < 100:
            continue

        r, p = spearmanr(si_past.loc[common_idx], returns_future.loc[common_idx])
        decay_curve.append({
            'lag_hours': lag,
            'correlation': r,
            'p_value': p,
            'significant': p < 0.05
        })

    df = pd.DataFrame(decay_curve)

    # Find half-life (lag where |correlation| drops to 50% of max)
    if len(df) > 0 and df['correlation'].abs().max() > 0:
        max_corr = df['correlation'].abs().max()
        half_life_df = df[df['correlation'].abs() < max_corr * 0.5]
        half_life = half_life_df['lag_hours'].iloc[0] if len(half_life_df) > 0 else max_lag_hours
    else:
        half_life = np.nan

    # Find optimal lag (highest absolute correlation)
    if len(df) > 0:
        optimal_lag = df.loc[df['correlation'].abs().idxmax(), 'lag_hours']
        optimal_corr = df['correlation'].abs().max()
    else:
        optimal_lag, optimal_corr = np.nan, np.nan

    return {
        'decay_curve': df,
        'half_life_hours': half_life,
        'optimal_lag_hours': optimal_lag,
        'optimal_correlation': optimal_corr
    }
```

### 1.16 Transaction Cost Sensitivity (Expert Recommendation)

```python
def transaction_cost_sensitivity(si_signal: pd.Series, returns: pd.Series,
                                  cost_bps_range=[5, 10, 20, 50, 100]):
    """
    Does the SI signal survive transaction costs?

    From Expert Panel: Hedge Fund Quant
    """
    results = []

    # Simple strategy: Long when SI > median, flat otherwise
    si_median = si_signal.median()
    position = (si_signal > si_median).astype(int)

    # Count trades
    trades = (position.diff().abs() > 0).sum()

    for cost_bps in cost_bps_range:
        cost_per_trade = cost_bps / 10000

        # Gross return
        strategy_return = (position.shift(1) * returns).sum()

        # Trading costs
        total_cost = trades * cost_per_trade

        # Net return
        net_return = strategy_return - total_cost

        results.append({
            'cost_bps': cost_bps,
            'gross_return': strategy_return,
            'trading_cost': total_cost,
            'net_return': net_return,
            'trades': trades,
            'breakeven_cost_bps': (strategy_return / trades * 10000) if trades > 0 else np.inf
        })

    return pd.DataFrame(results)
```

### 1.17 Pre-Registration Public Commit (Expert Recommendation)

```python
def create_preregistration():
    """
    Create and commit pre-registration before analysis.

    From Expert Panel: Prof. Behavioral Economics
    """
    import datetime
    import hashlib

    preregistration = {
        'title': 'SI Correlation Discovery Study',
        'date': datetime.datetime.now().isoformat(),
        'registered_before_analysis': True,

        'hypotheses': {
            'primary': 'SI correlates with at least one feature after FDR correction',
            'secondary': 'See SI_EXPLORATION.md for 40 specific hypotheses'
        },

        'analysis_plan': {
            'primary_si_variant': 'si_rolling_7d',
            'primary_correlation_method': 'spearman',
            'primary_alpha': 0.05,
            'correction_method': 'fdr_bh',
            'minimum_effect_size': 0.2,
        },

        'data_plan': {
            'assets': ['BTC', 'ETH', 'SOL'],
            'date_range': '12 months',
            'granularity': '1h',
            'train_pct': 0.70,
            'val_pct': 0.15,
            'test_pct': 0.15,
        },

        'stopping_rules': {
            'stop_if': 'No features significant in validation after discovering in train',
            'continue_if': 'At least 3 features confirmed in validation'
        },

        'commitment': {
            'publish_null_results': True,
            'no_post_hoc_hypothesis_changes': True,
        }
    }

    # Create hash for integrity
    content = json.dumps(preregistration, sort_keys=True)
    preregistration['hash'] = hashlib.sha256(content.encode()).hexdigest()

    # Save
    with open('results/preregistration.json', 'w') as f:
        json.dump(preregistration, f, indent=2)

    print(f"Pre-registration saved with hash: {preregistration['hash']}")
    print("COMMIT THIS FILE BEFORE RUNNING ANY ANALYSIS!")

    return preregistration
```

### 1.2 Directory Structure

```
apps/trading/
├── src/
│   ├── data/
│   │   ├── loader.py          # Load price data
│   │   └── features.py        # Compute all features
│   ├── agents/
│   │   ├── base.py            # Agent base class
│   │   ├── strategies.py      # 3 strategies
│   │   └── thompson.py        # Thompson Sampling
│   ├── competition/
│   │   ├── niche_population.py  # NichePopulation algorithm
│   │   └── si_calculator.py     # Compute SI
│   ├── analysis/
│   │   ├── correlation.py     # Correlation analysis
│   │   ├── visualization.py   # Plots
│   │   └── interpretation.py  # Auto-interpret results
│   └── backtest/
│       └── runner.py          # Main backtest loop
├── experiments/
│   └── phase0_si_discovery.py # Main experiment script
├── results/
│   └── si_correlations/       # Output directory
└── data/
    └── bybit/                 # Raw price data
```

### 1.3 Core Classes

```python
# src/data/features.py

class FeatureCalculator:
    """Compute all 70+ features from price data and agent states."""

    def __init__(self, price_data: pd.DataFrame, window_sizes: List[int] = [24, 168, 720]):
        self.price_data = price_data
        self.windows = window_sizes  # 1d, 7d, 30d in hours

    def compute_all_features(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Compute all features at a given timestamp."""
        features = {}

        # Category 1-3: Market State & Microstructure
        features.update(self._compute_market_state_features(timestamp))

        # Category 6: Risk Metrics
        features.update(self._compute_risk_features(timestamp))

        # Category 7: Timing Features
        features.update(self._compute_timing_features(timestamp))

        # Category 8: Behavioral Features
        features.update(self._compute_behavioral_features(timestamp))

        # Category 9: Factor Features
        features.update(self._compute_factor_features(timestamp))

        return features
```

---

## 📊 Phase 2: Feature Computation (Day 3-4)

### 2.1 Feature Categories & Formulas

#### Category 1-3: Market State & Microstructure (15 features)

| Feature | Formula | Window | Hypothesis |
|---------|---------|--------|------------|
| `volatility` | `std(returns) * sqrt(periods)` | 24h, 7d | H1, H7 |
| `trend_strength` | `abs(return) / volatility` | 7d | H1 |
| `return_autocorr` | `corr(r_t, r_{t-1})` | 7d | H2 |
| `hurst_exponent` | R/S analysis | 30d | H2 |
| `return_entropy` | `-sum(p * log(p))` | 7d | H2 |
| `regime_duration` | Days since last vol regime change | N/A | H3 |
| `volume` | `mean(volume)` | 24h | H7 |
| `volume_volatility` | `std(volume) / mean(volume)` | 7d | H7 |
| `jump_frequency` | Count of |r| > 3σ | 7d | H8 |
| `variance_ratio` | `var(r_k) / (k * var(r_1))` | 7d | H8 |
| `adx` | Average Directional Index | 14d | H24 |
| `bb_width` | Bollinger Band width | 20d | H24 |
| `rsi` | Relative Strength Index | 14d | H13 |
| `macd_hist` | MACD histogram | 12/26/9 | H1 |
| `atr` | Average True Range | 14d | H7 |

```python
def _compute_market_state_features(self, timestamp: pd.Timestamp) -> Dict[str, float]:
    features = {}

    for window in self.windows:
        window_data = self._get_window(timestamp, window)
        returns = window_data['close'].pct_change().dropna()

        # Volatility
        features[f'volatility_{window}h'] = returns.std() * np.sqrt(window)

        # Trend Strength
        total_return = (window_data['close'].iloc[-1] / window_data['close'].iloc[0]) - 1
        features[f'trend_strength_{window}h'] = abs(total_return) / (returns.std() + 1e-8)

        # Return Autocorrelation
        if len(returns) > 2:
            features[f'return_autocorr_{window}h'] = returns.autocorr(lag=1)

        # Return Entropy (discretize returns into bins)
        bins = pd.cut(returns, bins=10, labels=False)
        probs = bins.value_counts(normalize=True)
        features[f'return_entropy_{window}h'] = -np.sum(probs * np.log(probs + 1e-8))

        # Volume features
        features[f'volume_{window}h'] = window_data['volume'].mean()
        features[f'volume_volatility_{window}h'] = window_data['volume'].std() / (window_data['volume'].mean() + 1e-8)

        # Jump frequency (|return| > 3 sigma)
        threshold = returns.std() * 3
        features[f'jump_frequency_{window}h'] = (np.abs(returns) > threshold).sum()

    # Hurst Exponent (longer window needed)
    features['hurst_exponent'] = self._compute_hurst(timestamp)

    # Variance Ratio
    features['variance_ratio'] = self._compute_variance_ratio(timestamp)

    # Technical Indicators
    features['adx'] = self._compute_adx(timestamp)
    features['bb_width'] = self._compute_bollinger_width(timestamp)
    features['rsi'] = self._compute_rsi(timestamp)
    features['atr'] = self._compute_atr(timestamp)

    return features
```

#### Category 4-5: Agent Behavior (10 features)

| Feature | Formula | Source | Hypothesis |
|---------|---------|--------|------------|
| `agent_correlation` | Mean pairwise corr of agent returns | Agents | H4 |
| `effective_n` | 1 / sum(w_i^2) | Agents | H4 |
| `winner_consistency` | % same winner in rolling window | Agents | H6 |
| `winner_spread` | Best return - Worst return | Agents | H9 |
| `viable_agent_count` | Agents with positive return | Agents | H9 |
| `agent_confidence_mean` | Mean Thompson posterior width | Agents | H5 |
| `niche_affinity_entropy` | Entropy of niche assignments | Agents | H6 |
| `strategy_concentration` | HHI of strategy usage | Agents | H18 |
| `position_correlation` | Corr of agent positions | Agents | H18 |
| `return_dispersion` | Std of agent returns | Agents | H9 |

```python
def compute_agent_features(self, agents: List[Agent], period_returns: Dict[int, float]) -> Dict[str, float]:
    features = {}

    returns = np.array(list(period_returns.values()))

    # Agent correlation (pairwise)
    if len(agents) >= 2:
        correlations = []
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                # Use historical returns for correlation
                corr = np.corrcoef(agents[i].return_history[-30:],
                                   agents[j].return_history[-30:])[0,1]
                correlations.append(corr)
        features['agent_correlation'] = np.mean(correlations)

    # Effective N (diversification)
    weights = np.array([a.capital / sum(a.capital for a in agents) for a in agents])
    features['effective_n'] = 1.0 / np.sum(weights ** 2)

    # Winner spread
    features['winner_spread'] = returns.max() - returns.min()

    # Viable agents
    features['viable_agent_count'] = (returns > 0).sum()

    # Return dispersion
    features['return_dispersion'] = returns.std()

    # Strategy concentration (HHI)
    strategy_counts = Counter(a.current_strategy for a in agents)
    total = sum(strategy_counts.values())
    shares = [c / total for c in strategy_counts.values()]
    features['strategy_concentration'] = sum(s**2 for s in shares)

    return features
```

#### Category 6: Risk Metrics (10 features)

| Feature | Formula | Window | Hypothesis |
|---------|---------|--------|------------|
| `max_drawdown` | Max peak-to-trough | 30d | H21, H22 |
| `var_95` | 5th percentile of returns | 30d | H21 |
| `cvar_95` | Mean of returns below VaR | 30d | H21 |
| `volatility_of_volatility` | Std of rolling volatility | 30d | H21 |
| `tail_ratio` | 95th percentile / |5th| | 30d | H21 |
| `drawdown_recovery_days` | Days to recover from DD | History | H22 |
| `win_rate` | % of winning trades | 30d | H19 |
| `profit_factor` | Gross profit / Gross loss | 30d | H19 |
| `sharpe_ratio` | Mean / Std of returns | 30d | H23 |
| `sortino_ratio` | Mean / Downside Std | 30d | H23 |

```python
def _compute_risk_features(self, returns: pd.Series, equity_curve: pd.Series) -> Dict[str, float]:
    features = {}

    # Max Drawdown
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    features['max_drawdown'] = drawdowns.min()

    # VaR and CVaR
    features['var_95'] = np.percentile(returns, 5)
    features['cvar_95'] = returns[returns <= features['var_95']].mean()

    # Volatility of Volatility
    rolling_vol = returns.rolling(24).std()
    features['volatility_of_volatility'] = rolling_vol.std()

    # Tail Ratio
    upper_tail = np.percentile(returns, 95)
    lower_tail = abs(np.percentile(returns, 5))
    features['tail_ratio'] = upper_tail / (lower_tail + 1e-8)

    # Win Rate
    features['win_rate'] = (returns > 0).mean()

    # Profit Factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    features['profit_factor'] = gross_profit / (gross_loss + 1e-8)

    # Sharpe Ratio
    features['sharpe_ratio'] = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252 * 24)

    # Sortino Ratio
    downside = returns[returns < 0].std()
    features['sortino_ratio'] = returns.mean() / (downside + 1e-8) * np.sqrt(252 * 24)

    return features
```

#### Category 7-8: Timing & Behavioral (8 features)

| Feature | Formula | Window | Hypothesis |
|---------|---------|--------|------------|
| `next_day_return` | Return at t+1 | Lead | H14, H23 |
| `next_day_volatility` | Volatility at t+1 | Lead | H14, H23 |
| `holding_period_avg` | Average trade duration | History | H25 |
| `momentum_return` | Return of momentum strategy | 7d | H24 |
| `meanrev_return` | Return of mean-rev strategy | 7d | H24 |
| `fear_greed_proxy` | RSI deviation from 50 | 14d | H13, H26 |
| `extreme_high` | SI > 90th percentile | SI | H26 |
| `extreme_low` | SI < 10th percentile | SI | H26 |

#### Category 9: Factor & Cross-Asset (8 features)

| Feature | Formula | Window | Hypothesis |
|---------|---------|--------|------------|
| `momentum_factor` | Long winners, short losers | 30d | H29 |
| `value_factor` | Price vs MA deviation | 30d | H29 |
| `si_btc` | SI for BTC agents | Current | H30 |
| `si_eth` | SI for ETH agents | Current | H30 |
| `si_cross_asset_corr` | Correlation of SI across assets | 7d | H30 |
| `asset_return_corr` | Cross-asset return correlation | 7d | H30 |
| `rotation_signal` | Relative SI change | 7d | H31 |
| `relative_strength` | Asset return vs benchmark | 7d | H31 |

#### Category 10-11: Dynamics & Meta (9 features)

| Feature | Formula | Window | Hypothesis |
|---------|---------|--------|------------|
| `dsi_dt` | SI change (velocity) | 1d | H16 |
| `si_acceleration` | dSI/dt change | 1d | H16 |
| `si_rolling_std` | Stability of SI | 7d | H34 |
| `si_1h` | SI at 1-hour scale | 1h | H38 |
| `si_4h` | SI at 4-hour scale | 4h | H38 |
| `si_1d` | SI at daily scale | 1d | H38 |
| `si_1w` | SI at weekly scale | 1w | H38 |
| `prediction_accuracy` | Model correctness | 30d | H33 |
| `optimal_leverage` | Kelly criterion | 30d | H35 |

---

## 📊 Phase 3: SI Computation (Day 3-4)

### 3.1 SI Calculator

```python
# src/competition/si_calculator.py

class SICalculator:
    """Compute Specialization Index at various granularities."""

    def compute_si(self, agents: List[Agent], niche_assignments: Dict[int, str]) -> float:
        """
        Compute SI as entropy-based specialization measure.

        SI = 1 - H(niche|agent) / H_max

        Where H(niche|agent) is the entropy of niche assignments
        and H_max = log(num_niches)
        """
        # Count niche assignments per agent
        niche_counts = Counter(niche_assignments.values())
        total = sum(niche_counts.values())

        if total == 0:
            return 0.0

        # Compute entropy
        probs = [count / total for count in niche_counts.values()]
        entropy = -sum(p * np.log(p + 1e-8) for p in probs)

        # Normalize by max entropy
        num_niches = len(set(niche_assignments.values()))
        max_entropy = np.log(num_niches) if num_niches > 1 else 1.0

        # SI = 1 means perfect specialization (low entropy)
        # SI = 0 means no specialization (max entropy)
        si = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

        return max(0.0, min(1.0, si))

    def compute_si_timeseries(self,
                              agents: List[Agent],
                              history: List[Dict],
                              window: int = 24) -> pd.Series:
        """Compute SI over rolling window."""
        si_values = []
        timestamps = []

        for i in range(window, len(history)):
            window_data = history[i-window:i]
            # Get niche assignments from window
            niche_assignments = self._aggregate_niches(window_data)
            si = self.compute_si(agents, niche_assignments)
            si_values.append(si)
            timestamps.append(history[i]['timestamp'])

        return pd.Series(si_values, index=timestamps)
```

### 3.2 SI Variants to Compute

| SI Variant | Description | Purpose |
|------------|-------------|---------|
| `si_instant` | SI at each timestep | Raw signal |
| `si_rolling_1d` | 24h rolling average | Smooth signal |
| `si_rolling_7d` | 7d rolling average | Trend |
| `si_velocity` | dSI/dt | H16: Rate of change |
| `si_acceleration` | d²SI/dt² | H16: Momentum |
| `si_std_7d` | Rolling std of SI | H34: Stability |
| `si_percentile` | Current SI percentile in history | H26: Extremes |
| `si_per_asset` | SI computed per asset | H30, H31 |

---

## 📊 Phase 4: Correlation Analysis (Day 5)

### 4.1 Statistical Methods (Audit-Corrected)

```python
# src/analysis/correlation.py

class CorrelationAnalyzer:
    """
    Comprehensive correlation analysis with PROPER TIME SERIES methods.

    Key fixes from methodology audit:
    1. HAC standard errors for autocorrelation
    2. Block bootstrap for confidence intervals
    3. Stationarity checks
    4. Partial correlations for confounders
    5. Predictive (lagged) correlations
    """

    def __init__(self, block_size=24):
        self.block_size = block_size  # 24 hours for hourly data

    def analyze_all(self, si: pd.Series, features: pd.DataFrame,
                    confounders: pd.DataFrame = None) -> pd.DataFrame:
        """Run all correlation analyses with proper methods."""
        results = []

        for col in features.columns:
            feature = features[col].dropna()
            si_aligned = si.loc[feature.index]

            if len(si_aligned) < 100:  # Need enough for block bootstrap
                continue

            # Check stationarity
            is_stationary = self._check_stationarity(si_aligned, feature)

            result = {
                'feature': col,
                'n': len(si_aligned),
                'is_stationary': is_stationary,

                # PRIMARY: Spearman with HAC standard errors
                'spearman_r': self._spearman_hac(si_aligned, feature)[0],
                'spearman_p': self._spearman_hac(si_aligned, feature)[1],

                # Block bootstrap 95% CI
                'ci_lower': self._block_bootstrap_ci(si_aligned, feature)[0],
                'ci_upper': self._block_bootstrap_ci(si_aligned, feature)[1],

                # Mutual Information (nonlinear)
                'mutual_info': self._mutual_info(si_aligned, feature),

                # PREDICTIVE: SI leads feature by k periods
                'lead_1h_r': self._lagged_correlation(si_aligned, feature, lag=1),
                'lead_6h_r': self._lagged_correlation(si_aligned, feature, lag=6),
                'lead_24h_r': self._lagged_correlation(si_aligned, feature, lag=24),

                # Granger Causality
                'granger_p': self._granger_causality(si_aligned, feature),

                # Partial correlation (if confounders provided)
                'partial_r': self._partial_correlation(
                    si_aligned, feature, confounders
                ) if confounders is not None else np.nan,
            }
            results.append(result)

        return pd.DataFrame(results)

    def _check_stationarity(self, x, y):
        """Check if both series are stationary."""
        from statsmodels.tsa.stattools import adfuller
        x_stat = adfuller(x.dropna())[1] < 0.05
        y_stat = adfuller(y.dropna())[1] < 0.05
        return x_stat and y_stat

    def _spearman_hac(self, x, y):
        """Spearman correlation with HAC standard errors."""
        from scipy.stats import spearmanr
        import statsmodels.api as sm

        # Rank transform for Spearman
        x_rank = x.rank()
        y_rank = y.rank()

        # OLS with HAC standard errors
        X = sm.add_constant((x_rank - x_rank.mean()) / x_rank.std())
        Y = (y_rank - y_rank.mean()) / y_rank.std()

        model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': self.block_size})

        return model.params[1], model.pvalues[1]

    def _block_bootstrap_ci(self, x, y, n_bootstrap=1000, alpha=0.05):
        """Block bootstrap confidence interval for correlation."""
        correlations = []
        n = len(x)
        n_blocks = n // self.block_size

        for _ in range(n_bootstrap):
            # Sample blocks with replacement
            block_indices = np.random.choice(n_blocks, n_blocks, replace=True)
            boot_x = np.concatenate([
                x.iloc[i*self.block_size:(i+1)*self.block_size].values
                for i in block_indices
            ])
            boot_y = np.concatenate([
                y.iloc[i*self.block_size:(i+1)*self.block_size].values
                for i in block_indices
            ])

            from scipy.stats import spearmanr
            r, _ = spearmanr(boot_x, boot_y)
            correlations.append(r)

        return np.percentile(correlations, [alpha/2*100, (1-alpha/2)*100])

    def _lagged_correlation(self, x, y, lag):
        """Test if x at time t predicts y at time t+lag."""
        from scipy.stats import spearmanr
        if lag > 0:
            return spearmanr(x.iloc[:-lag], y.iloc[lag:])[0]
        else:
            return spearmanr(x, y)[0]

    def _partial_correlation(self, x, y, confounders):
        """Correlation controlling for confounders."""
        import statsmodels.api as sm

        # Residualize x
        X_conf = sm.add_constant(confounders.loc[x.index])
        x_resid = sm.OLS(x, X_conf).fit().resid

        # Residualize y
        y_resid = sm.OLS(y, X_conf).fit().resid

        from scipy.stats import spearmanr
        return spearmanr(x_resid, y_resid)[0]

    def _mutual_info(self, x, y):
        from sklearn.feature_selection import mutual_info_regression
        return mutual_info_regression(
            x.values.reshape(-1, 1),
            y.values,
            random_state=42
        )[0]

    def _granger_causality(self, x, y, max_lag=5):
        """Test if x Granger-causes y."""
        from statsmodels.tsa.stattools import grangercausalitytests
        data = pd.DataFrame({'x': x, 'y': y}).dropna()
        try:
            result = grangercausalitytests(data[['y', 'x']], maxlag=max_lag, verbose=False)
            # Return minimum p-value across lags
            p_vals = [result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
            return min(p_vals)
        except:
            return np.nan
```

### 4.2 Multiple Testing Correction

```python
def apply_corrections(self, results: pd.DataFrame) -> pd.DataFrame:
    """Apply multiple testing corrections."""
    from statsmodels.stats.multitest import multipletests

    # Bonferroni correction
    results['pearson_p_bonferroni'] = results['pearson_p'] * len(results)
    results['pearson_p_bonferroni'] = results['pearson_p_bonferroni'].clip(upper=1.0)

    # FDR (Benjamini-Hochberg)
    _, results['pearson_p_fdr'], _, _ = multipletests(
        results['pearson_p'], method='fdr_bh'
    )

    # Significance flags
    results['sig_raw'] = results['pearson_p'] < 0.05
    results['sig_bonferroni'] = results['pearson_p_bonferroni'] < 0.05
    results['sig_fdr'] = results['pearson_p_fdr'] < 0.05

    return results
```

### 4.3 Output Format

```python
# Results structure
{
    'summary': {
        'total_features': 70,
        'significant_raw': 25,
        'significant_bonferroni': 8,
        'significant_fdr': 12,
        'top_10_features': [...],
        'interpretation': "..."
    },
    'correlations': pd.DataFrame,  # All correlations
    'regime_stratified': {
        'trending': pd.DataFrame,
        'ranging': pd.DataFrame,
        'volatile': pd.DataFrame
    },
    'temporal_analysis': {
        'granger_results': pd.DataFrame,
        'lead_lag_matrix': pd.DataFrame
    },
    'visualizations': {
        'heatmap_path': 'results/si_correlations/heatmap.png',
        'scatter_plots_path': 'results/si_correlations/scatters/',
        'time_series_path': 'results/si_correlations/timeseries.png'
    }
}
```

---

## 📊 Phase 5: Interpretation & Visualization (Day 6)

### 5.1 Auto-Interpretation

```python
# src/analysis/interpretation.py

class Interpreter:
    """Automatically interpret correlation results."""

    def interpret(self, results: pd.DataFrame) -> Dict:
        """Generate human-readable interpretation."""

        # Sort by absolute correlation
        results_sorted = results.sort_values('pearson_r', key=abs, ascending=False)

        # Top positive correlations
        top_positive = results_sorted[results_sorted['pearson_r'] > 0].head(5)

        # Top negative correlations
        top_negative = results_sorted[results_sorted['pearson_r'] < 0].head(5)

        # Significant after correction
        significant = results_sorted[results_sorted['sig_fdr']]

        interpretation = {
            'main_finding': self._generate_main_finding(significant),
            'top_positive': self._describe_correlations(top_positive, 'positive'),
            'top_negative': self._describe_correlations(top_negative, 'negative'),
            'hypothesis_support': self._map_to_hypotheses(significant),
            'recommended_path': self._recommend_path(significant),
            'caveats': self._generate_caveats(results)
        }

        return interpretation

    def _generate_main_finding(self, significant: pd.DataFrame) -> str:
        if len(significant) == 0:
            return "No statistically significant correlations found after FDR correction."

        top = significant.iloc[0]
        direction = "positively" if top['pearson_r'] > 0 else "negatively"

        return f"SI is most strongly {direction} correlated with {top['feature']} (r={top['pearson_r']:.3f}, p={top['pearson_p_fdr']:.4f})"

    def _map_to_hypotheses(self, significant: pd.DataFrame) -> Dict[str, str]:
        """Map significant correlations to our 40 hypotheses."""
        hypothesis_map = {
            'volatility': 'H1, H7',
            'return_entropy': 'H2',
            'regime_duration': 'H3',
            'agent_correlation': 'H4',
            'max_drawdown': 'H21, H22',
            'win_rate': 'H19',
            'next_day_return': 'H14, H23',
            'dsi_dt': 'H16',
            # ... more mappings
        }

        supported = {}
        for _, row in significant.iterrows():
            feature = row['feature']
            for key, hyp in hypothesis_map.items():
                if key in feature.lower():
                    supported[feature] = hyp

        return supported
```

### 5.2 Visualization Suite

```python
# src/analysis/visualization.py

class Visualizer:
    """Generate all visualizations for SI correlation analysis."""

    def generate_all(self, si: pd.Series, features: pd.DataFrame,
                     results: pd.DataFrame, output_dir: str):
        """Generate complete visualization suite."""

        # 1. Correlation Heatmap
        self._plot_heatmap(results, f"{output_dir}/heatmap.png")

        # 2. Top 10 Scatter Plots
        self._plot_top_scatters(si, features, results, f"{output_dir}/scatters/")

        # 3. SI Time Series with Annotations
        self._plot_si_timeseries(si, features, f"{output_dir}/si_timeseries.png")

        # 4. Lead-Lag Heatmap
        self._plot_lead_lag(results, f"{output_dir}/lead_lag.png")

        # 5. Regime-Stratified Correlations
        self._plot_regime_comparison(results, f"{output_dir}/regime_comparison.png")

        # 6. Effect Size Forest Plot
        self._plot_forest(results, f"{output_dir}/effect_sizes.png")

    def _plot_heatmap(self, results: pd.DataFrame, path: str):
        """Correlation heatmap sorted by absolute correlation."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Pivot for heatmap
        sorted_results = results.sort_values('pearson_r', key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(12, 20))

        # Create bar chart of correlations
        colors = ['green' if r > 0 else 'red' for r in sorted_results['pearson_r']]
        bars = ax.barh(range(len(sorted_results)), sorted_results['pearson_r'], color=colors)

        # Add significance markers
        for i, (_, row) in enumerate(sorted_results.iterrows()):
            if row['sig_fdr']:
                ax.annotate('***', (row['pearson_r'], i), fontsize=10)
            elif row['sig_raw']:
                ax.annotate('*', (row['pearson_r'], i), fontsize=10)

        ax.set_yticks(range(len(sorted_results)))
        ax.set_yticklabels(sorted_results['feature'])
        ax.set_xlabel('Pearson Correlation with SI')
        ax.set_title('SI Correlation with All Features\n(*** = FDR significant, * = raw significant)')
        ax.axvline(x=0, color='black', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
```

---

## 📊 Phase 6: Decision & Next Steps (Day 7)

### 6.1 Decision Framework

```python
def make_decision(interpretation: Dict) -> str:
    """Determine next steps based on findings."""

    significant = interpretation['hypothesis_support']

    # Check for direct profit path
    if any('profit' in f.lower() or 'return' in f.lower() for f in significant):
        return "DIRECT_PROFIT_PATH"

    # Check for risk management path
    if any('drawdown' in f.lower() or 'var' in f.lower() or 'risk' in f.lower() for f in significant):
        return "RISK_MANAGEMENT_PATH"

    # Check for diversification path
    if any('correlation' in f.lower() or 'effective_n' in f.lower() for f in significant):
        return "DIVERSIFICATION_PATH"

    # Check for timing path
    if any('next_day' in f.lower() or 'leading' in f.lower() for f in significant):
        return "TIMING_SIGNAL_PATH"

    # Check for regime path
    if any('regime' in f.lower() or 'trend' in f.lower() for f in significant):
        return "REGIME_DETECTION_PATH"

    # No clear path
    return "DEEPER_ANALYSIS_NEEDED"
```

### 6.2 Output Report Template

```markdown
# SI Correlation Discovery Report

## Executive Summary
[Auto-generated main finding]

## Top 10 Correlations

| Rank | Feature | Correlation | P-value (FDR) | Hypothesis Supported |
|------|---------|-------------|---------------|----------------------|
| 1    | ...     | ...         | ...           | ...                  |

## Interpretation

### What SI Measures
Based on the strongest correlations, SI appears to measure: [...]

### Path to Profit
[DIRECT / VIA RISK / VIA DIVERSIFICATION / VIA TIMING / ...]

### Recommended Next Steps
1. [...]
2. [...]
3. [...]

## Caveats
- Multiple testing: X/70 features significant after FDR correction
- Temporal stability: [...]
- Regime dependence: [...]

## Appendix
- Full correlation table
- All visualizations
- Code and reproducibility
```

---

---

## 📊 Three Pipelines Implementation

### Pipeline 1: Discovery (46 Features)

**Question**: What does SI correlate with (concurrent)?

```python
# Features for Discovery Pipeline
DISCOVERY_FEATURES = {
    'market': [
        'volatility_24h', 'volatility_7d', 'volatility_30d',
        'trend_strength_7d', 'return_autocorr_7d', 'hurst_exponent',
        'return_entropy_7d', 'volume_24h', 'volume_volatility_7d',
        'jump_frequency_7d', 'variance_ratio',
        'adx', 'bb_width', 'rsi', 'atr',
    ],
    'agent_safe': [
        'agent_correlation',      # Different from SI
        'winner_spread',          # Magnitude, not pattern
        'viable_agent_count',     # Count, not specialization
        'return_dispersion',      # Variance of returns
        'effective_n',            # Diversification
        'winner_consistency',     # Stability
    ],
    'risk': [
        'max_drawdown_30d', 'var_95_30d', 'cvar_95_30d',
        'volatility_of_volatility_30d', 'tail_ratio_30d',
        'drawdown_recovery_days', 'win_rate_30d',
        'profit_factor_30d', 'sharpe_ratio_30d', 'sortino_ratio_30d',
    ],
    'behavioral': [
        'fear_greed_proxy', 'holding_period_avg',
        'momentum_return_7d', 'meanrev_return_7d',
        'regime_duration', 'days_since_regime_change',
    ],
    'liquidity': [
        'volume_z', 'amihud_log', 'volume_volatility_24h', 'spread_proxy',
    ],
    'cross_asset': [
        'asset_return_corr', 'relative_strength', 'rotation_signal',
        'si_cross_asset_corr', 'si_other_asset',
    ],
}

def run_discovery_pipeline(si: pd.Series, features: pd.DataFrame,
                           train_idx, val_idx, test_idx) -> Dict:
    """
    Pipeline 1: What does SI correlate with?
    """
    results = {'train': {}, 'val': {}, 'test': {}}

    # 1. Discovery on TRAIN
    train_correlations = []
    for col in features.columns:
        if col in CIRCULAR_FEATURES:
            continue  # Skip circular features

        si_train = si.loc[train_idx]
        feat_train = features.loc[train_idx, col]

        r, p = spearman_hac(si_train, feat_train)
        ci = block_bootstrap_ci(si_train, feat_train)

        train_correlations.append({
            'feature': col,
            'r': r, 'p': p,
            'ci_lower': ci[0], 'ci_upper': ci[1]
        })

    # Apply FDR correction
    train_df = pd.DataFrame(train_correlations)
    train_df = apply_fdr(train_df)

    # Identify candidates
    candidates = train_df[
        (train_df['p_fdr'] < 0.05) &
        (abs(train_df['r']) > MIN_EFFECT_SIZE)
    ]
    results['train'] = candidates

    # 2. Confirm on VALIDATION
    confirmed = []
    for _, row in candidates.iterrows():
        col = row['feature']
        si_val = si.loc[val_idx]
        feat_val = features.loc[val_idx, col]

        r, p = spearman_hac(si_val, feat_val)

        # Confirm: same direction AND significant
        if np.sign(r) == np.sign(row['r']) and p < 0.05:
            confirmed.append({
                'feature': col,
                'train_r': row['r'],
                'val_r': r,
                'val_p': p
            })

    results['val'] = pd.DataFrame(confirmed)

    # 3. Final test on TEST
    final = []
    for _, row in results['val'].iterrows():
        col = row['feature']
        si_test = si.loc[test_idx]
        feat_test = features.loc[test_idx, col]

        r, p = spearman_hac(si_test, feat_test)
        ci = block_bootstrap_ci(si_test, feat_test)

        final.append({
            'feature': col,
            'train_r': row['train_r'],
            'val_r': row['val_r'],
            'test_r': r,
            'test_p': p,
            'test_ci': ci,
            'consistent': np.sign(r) == np.sign(row['train_r'])
        })

    results['test'] = pd.DataFrame(final)

    return results
```

### Pipeline 2: Prediction (2 Features)

**Question**: Does SI predict future outcomes?

```python
PREDICTION_FEATURES = ['next_day_return', 'next_day_volatility']

def run_prediction_pipeline(si: pd.Series, future_data: pd.DataFrame,
                            train_idx, val_idx, test_idx) -> Dict:
    """
    Pipeline 2: Does SI predict future outcomes?

    Key difference: We're testing SI(t) vs Outcome(t+k)
    """
    results = {}

    for target in PREDICTION_FEATURES:
        target_results = {
            'signal_decay': None,
            'optimal_lag': None,
            'granger': None,
            'train_val_test': {}
        }

        # 1. Signal decay analysis (how quickly does prediction decay?)
        decay = analyze_signal_decay(
            si.loc[train_idx],
            future_data.loc[train_idx, target],
            max_lag_hours=168
        )
        target_results['signal_decay'] = decay['decay_curve']
        target_results['optimal_lag'] = decay['optimal_lag_hours']
        target_results['half_life'] = decay['half_life_hours']

        # 2. Granger causality (does SI Granger-cause target?)
        granger_p = granger_causality_test(
            si.loc[train_idx],
            future_data.loc[train_idx, target],
            max_lag=24
        )
        target_results['granger_p'] = granger_p

        # 3. Train/Val/Test at optimal lag
        optimal_lag = int(decay['optimal_lag_hours'])

        for split_name, split_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            si_split = si.loc[split_idx]
            target_split = future_data.loc[split_idx, target]

            # Lagged correlation
            r = lagged_correlation(si_split, target_split, lag=optimal_lag)
            ci = block_bootstrap_ci_lagged(si_split, target_split, lag=optimal_lag)

            target_results['train_val_test'][split_name] = {
                'r': r, 'ci': ci, 'lag': optimal_lag
            }

        results[target] = target_results

    return results
```

### Pipeline 3: SI Dynamics (9 Features)

**Question**: How should we use SI?

```python
SI_DYNAMICS_FEATURES = [
    'dsi_dt', 'si_acceleration', 'si_rolling_std',
    'si_1h', 'si_4h', 'si_1d', 'si_1w', 'si_percentile',
    'extreme_high', 'extreme_low'
]

def run_si_dynamics_pipeline(si_variants: Dict[str, pd.Series],
                              profit: pd.Series) -> Dict:
    """
    Pipeline 3: How should we use SI?

    This answers:
    - Which SI variant is best?
    - Does SI momentum (dSI/dt) matter?
    - Is SI stability important?
    - Do extremes predict reversals?

    NOT about correlation discovery - about optimal SI usage.
    """
    results = {}

    # 1. Compare SI variants (which correlates best with profit?)
    variant_performance = {}
    for variant_name, si in si_variants.items():
        r, p = spearmanr(si, profit)
        variant_performance[variant_name] = {'r': r, 'p': p}

    results['best_variant'] = max(variant_performance,
                                   key=lambda x: abs(variant_performance[x]['r']))
    results['variant_performance'] = variant_performance

    # 2. SI momentum analysis (does change in SI matter?)
    si_base = si_variants['si_rolling_7d']
    dsi = si_base.diff()

    # Does rising SI predict profit?
    rising_si = dsi > 0
    profit_when_rising = profit[rising_si].mean()
    profit_when_falling = profit[~rising_si].mean()

    results['momentum_effect'] = {
        'profit_when_rising': profit_when_rising,
        'profit_when_falling': profit_when_falling,
        'difference': profit_when_rising - profit_when_falling
    }

    # 3. SI stability analysis (does volatile SI matter?)
    si_std = si_base.rolling(168).std()

    stable_si = si_std < si_std.median()
    profit_when_stable = profit[stable_si].mean()
    profit_when_volatile = profit[~stable_si].mean()

    results['stability_effect'] = {
        'profit_when_stable': profit_when_stable,
        'profit_when_volatile': profit_when_volatile,
        'difference': profit_when_stable - profit_when_volatile
    }

    # 4. Extremes analysis (do SI extremes predict reversals?)
    si_pct = si_base.rank(pct=True)

    extreme_high = si_pct > 0.9
    extreme_low = si_pct < 0.1

    results['extremes_effect'] = {
        'profit_after_extreme_high': profit[extreme_high.shift(24)].mean(),
        'profit_after_extreme_low': profit[extreme_low.shift(24)].mean(),
        'profit_normal': profit[~extreme_high & ~extreme_low].mean()
    }

    return results
```

### Master Runner: All Three Pipelines

```python
def run_all_pipelines(data: pd.DataFrame, agents: List) -> Dict:
    """
    Run all three pipelines and compile results.
    """
    # 1. Compute SI and features
    si = compute_si_timeseries(agents, data)
    features = compute_all_features(data)
    si_variants = compute_si_variants(agents, data)

    # 2. Split data
    train_idx, val_idx, test_idx = temporal_split(data.index)

    # 3. Run Pipeline 1: Discovery
    print("Running Pipeline 1: Discovery...")
    discovery_results = run_discovery_pipeline(
        si, features, train_idx, val_idx, test_idx
    )

    # 4. Run Pipeline 2: Prediction
    print("Running Pipeline 2: Prediction...")
    prediction_results = run_prediction_pipeline(
        si, features[PREDICTION_FEATURES], train_idx, val_idx, test_idx
    )

    # 5. Run Pipeline 3: SI Dynamics
    print("Running Pipeline 3: SI Dynamics...")
    profit = features['profit'] if 'profit' in features else features['next_day_return']
    dynamics_results = run_si_dynamics_pipeline(si_variants, profit)

    # 6. Compile report
    results = {
        'discovery': discovery_results,
        'prediction': prediction_results,
        'dynamics': dynamics_results,
        'summary': generate_summary(discovery_results, prediction_results, dynamics_results)
    }

    return results

def generate_summary(discovery, prediction, dynamics):
    """Generate executive summary across all pipelines."""
    summary = {
        'discovery': {
            'question': "What does SI correlate with?",
            'features_tested': len(DISCOVERY_FEATURES),
            'significant_train': len(discovery['train']),
            'confirmed_val': len(discovery['val']),
            'held_test': len(discovery['test']),
            'top_correlates': discovery['test'].head(5)['feature'].tolist() if len(discovery['test']) > 0 else []
        },
        'prediction': {
            'question': "Does SI predict future outcomes?",
            'next_day_return': {
                'predictive': prediction['next_day_return']['granger_p'] < 0.05,
                'optimal_lag': prediction['next_day_return']['optimal_lag'],
                'half_life': prediction['next_day_return']['half_life']
            },
            'next_day_volatility': {
                'predictive': prediction['next_day_volatility']['granger_p'] < 0.05,
                'optimal_lag': prediction['next_day_volatility']['optimal_lag']
            }
        },
        'dynamics': {
            'question': "How should we use SI?",
            'best_variant': dynamics['best_variant'],
            'momentum_helps': dynamics['momentum_effect']['difference'] > 0,
            'stability_helps': dynamics['stability_effect']['difference'] > 0,
            'extremes_contrarian': dynamics['extremes_effect']['profit_after_extreme_high'] < dynamics['extremes_effect']['profit_normal']
        }
    }
    return summary
```

---

## 📅 Timeline Summary (Three Pipelines)

| Day | Phase | Tasks | Output |
|-----|-------|-------|--------|
| 0 | **Pre-Reg** | Document pre-registration | `pre_registration.json` |
| 1 | Setup | Data loading, temporal split | Train/Val/Test sets |
| 2 | Setup | Feature calculators (46 discovery + 2 prediction + 9 dynamics) | Categorized features |
| 3 | Compute | Backtest, compute SI variants | SI time series (8 variants) |
| 4 | Compute | All features, remove circular, stationarity checks | Clean features |
| 5 | **Pipeline 1** | Discovery on TRAIN → VAL → TEST | Top SI correlates |
| 6 | **Pipeline 2** | Prediction analysis (lagged, Granger, decay) | Predictive power |
| 7 | **Pipeline 3** | SI Dynamics (variants, momentum, stability) | How to use SI |
| 8 | **Report** | Compile three-pipeline report | Final conclusions |

---

## 🔧 Implementation Checklist (Three Pipelines)

### Day 0: Pre-Registration (CRITICAL - Before Looking at Data)
- [ ] Document primary hypothesis
- [ ] Specify primary SI variant
- [ ] Specify primary correlation method
- [ ] Define success criteria
- [ ] Save `pre_registration.json`
- [ ] **Commit to GitHub with timestamp**

### Day 1: Data Preparation
- [ ] Load 12+ months of data
- [ ] **Temporal split: 70/15/15** (train/val/test)
- [ ] Verify no data leakage
- [ ] Check data quality (missing values, outliers)
- [ ] Estimate effective sample size (autocorrelation)
- [ ] Conduct power analysis → minimum detectable r

### Day 2: Feature Infrastructure
- [ ] Implement `FeatureCalculator`
- [ ] **Categorize features into 3 pipelines:**
  - [ ] Discovery features (46): Market, Agent (safe), Risk, Behavioral, Liquidity
  - [ ] Prediction features (2): next_day_return, next_day_volatility
  - [ ] SI Dynamics features (9): dSI/dt, SI variants, extremes
- [ ] **Remove circular features** from discovery pipeline
- [ ] Check stationarity of each feature
- [ ] Cluster correlated features on TRAIN only

### Day 3-4: Computation
- [ ] Implement 3 strategies
- [ ] Implement `NichePopulation` algorithm
- [ ] Implement `SICalculator`
- [ ] Run backtest
- [ ] Compute SI variants (8 types)
- [ ] Compute all features (properly categorized)
- [ ] Run negative controls (random noise, shuffled SI)

### Day 5: Pipeline 1 - Discovery
- [ ] Run on TRAIN: correlate SI with 46 discovery features
- [ ] Apply HAC standard errors + block bootstrap
- [ ] Apply FDR correction
- [ ] Compare to random baseline
- [ ] Identify candidates (FDR < 0.05 AND |r| > min_effect)
- [ ] Confirm on VAL: same direction AND p < 0.05
- [ ] Final test on TEST
- [ ] Record: "SI correlates with X, Y, Z"

### Day 6: Pipeline 2 - Prediction
- [ ] Signal decay analysis (how fast does signal decay?)
- [ ] Find optimal lag for each prediction target
- [ ] Granger causality tests
- [ ] Test on TRAIN → VAL → TEST at optimal lag
- [ ] Record: "SI predicts return with lag K, half-life H"

### Day 7: Pipeline 3 - SI Dynamics
- [ ] Compare SI variants (which is best?)
- [ ] Momentum analysis (does dSI/dt matter?)
- [ ] Stability analysis (does SI variance matter?)
- [ ] Extremes analysis (do extreme SI values predict reversals?)
- [ ] Record: "Best SI variant is X, momentum effect is Y"

### Day 8: Final Report
- [ ] Compile three-pipeline report
- [ ] Generate visualizations for each pipeline
- [ ] Write executive summary
- [ ] Document limitations
- [ ] **Report pipelines SEPARATELY** (not merged)
- [ ] Determine path forward based on findings

---

## 🔍 Eight Additional Audits (Expert Panel Recommendations)

See `COMPREHENSIVE_AUDIT_IMPLEMENTATION.md` for full code implementations.

### Audit 1: Pre-Registration (BEFORE ANALYSIS)
**Priority**: 🔴 CRITICAL

```bash
# Commit pre-registration BEFORE looking at data
git add experiments/pre_registration.json
git commit -m "PRE-REGISTRATION: SI hypothesis ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
git push origin main
```

- [x] Pre-registration file created: `experiments/pre_registration.json`
- [ ] Committed to GitHub with timestamp
- [ ] Data NOT analyzed before commit

### Audit 2: Implementation Correctness
**Priority**: 🔴 HIGH

```bash
# Run unit tests
pytest tests/test_si_calculation.py -v
pytest tests/test_no_lookahead.py -v
pytest tests/test_numerical_stability.py -v
```

Tests to run:
- [ ] SI calculation unit tests (known input → known output)
- [ ] Time-travel tests (no lookahead bugs)
- [ ] Numerical stability (edge cases: zeros, extremes, NaN)

### Audit 3: Causal Inference
**Priority**: 🔴 HIGH

```python
from src.causal_tests import CausalInferenceAudit

causal = CausalInferenceAudit()

# For each significant correlation:
results = {
    'granger': causal.bidirectional_granger(si, feature),  # SI → Feature OR Feature → SI?
    'placebo': causal.placebo_test(si, feature),           # Random SI correlates?
    'permutation': causal.permutation_test(si, feature)    # Shuffle survives?
}
```

Tests to run:
- [ ] Bidirectional Granger causality
- [ ] Placebo test (random SI control)
- [ ] Permutation test (shuffle labels)

### Audit 4: Strategy Validity
**Priority**: 🔴 HIGH

```python
from src.strategy_validation import StrategyValidityAudit

strategy_audit = StrategyValidityAudit()

# Parameter sensitivity
results = strategy_audit.parameter_sensitivity(
    data, MomentumStrategy, 'lookback', [3, 7, 14, 30]
)

# Benchmark comparison
results = strategy_audit.benchmark_comparison(data, strategy_returns)

# Transaction costs
results = strategy_audit.transaction_cost_sensitivity(data, MomentumStrategy)
```

Tests to run:
- [ ] Parameter sensitivity (results stable across parameters?)
- [ ] Benchmark comparison (beats buy-and-hold? beats random?)
- [ ] Transaction cost sensitivity (profitable at 0.3% cost?)

### Audit 5: Reproducibility
**Priority**: 🟡 MEDIUM

```python
from src.reproducibility import ReproducibilityManifest, set_all_seeds

# Set all seeds
set_all_seeds(42)

# Create manifest
manifest = ReproducibilityManifest.create_manifest(
    data_files={'btc': 'data/btc.csv', 'eth': 'data/eth.csv'},
    config=CONFIG
)
ReproducibilityManifest.save_manifest(manifest, 'experiments/manifest.json')
```

Checks:
- [ ] All random seeds locked
- [ ] Package versions pinned in requirements.txt
- [ ] Data files hashed
- [ ] Git commit recorded

### Audit 6: Crypto-Specific
**Priority**: 🟡 MEDIUM

```python
from src.crypto_specific_audit import CryptoSpecificAudit

crypto = CryptoSpecificAudit()
results = crypto.run_full_crypto_audit(data, si, 'volatility_7d')

# Checks:
# - Time-of-day effects (Asian/European/US sessions)
# - Weekend effects (patterns differ?)
# - Liquidity regimes (high vs low liquidity)
# - Outlier analysis (flash crashes, hacks)
```

Tests to run:
- [ ] Time-of-day consistency
- [ ] Weekend vs weekday patterns
- [ ] High vs low liquidity regimes
- [ ] Outlier impact analysis

### Audit 7: Multi-Asset Generalization
**Priority**: 🟡 MEDIUM

```python
from src.multi_asset_audit import MultiAssetAudit

multi = MultiAssetAudit(assets=['BTC', 'ETH', 'SOL', 'BNB', 'XRP'])

# Cross-asset generalization
results = multi.cross_asset_generalization(data_dict, si_dict, 'volatility_7d')

# Time period robustness
results = multi.time_period_robustness(data, si, 'volatility_7d')

# Market regime analysis
results = multi.market_regime_analysis(data, si, 'volatility_7d')
```

Tests to run:
- [ ] Cross-asset: findings hold for 3+ of 5 assets
- [ ] Time periods: consistent across quarters/years
- [ ] Market regimes: bull/bear/sideways consistency

### Audit 8: Adversarial Testing
**Priority**: 🟡 MEDIUM

```python
from src.adversarial_audit import AdversarialAudit

adversarial = AdversarialAudit()
results = adversarial.run_full_adversarial_audit(data, si, 'volatility_7d')

# Devil's advocate: find counterexamples
# Random control: SI shouldn't correlate with random
# Permutation: shuffle SI, correlation should vanish
# Subset stability: correlation holds in random subsets
```

Tests to run:
- [ ] Devil's advocate (find counterexamples)
- [ ] Random feature control (false positive rate OK?)
- [ ] Permutation test (survives shuffle?)
- [ ] Subset stability (80%+ subsets consistent?)

---

## 📋 Master Audit Checklist

### Before Analysis (Day 0)
- [x] Create pre-registration file
- [ ] Commit pre-registration to GitHub
- [ ] Create reproducibility manifest
- [ ] Run all unit tests (implementation audit)
- [ ] Verify no lookahead bugs

### During Analysis (Days 1-7)
- [ ] Run causal inference tests for each significant finding
- [ ] Run crypto-specific checks
- [ ] Run strategy validity checks
- [ ] Document any deviations from pre-registration

### After Analysis (Day 8+)
- [ ] Run multi-asset generalization
- [ ] Run adversarial tests
- [ ] Report ALL results including nulls
- [ ] Write final report with all audit results

---

## 🚨 Methodology Safeguards

### Red Flags to Watch For:
- [ ] Correlation only significant in train, not val → **Overfitting**
- [ ] Effect size below minimum detectable → **Underpowered**
- [ ] Concurrent but not predictive correlation → **Circular**
- [ ] Disappears with confounders → **Spurious**
- [ ] Only in one regime → **Not robust**

### Required for Reporting:
- [ ] Pre-registration documented
- [ ] Train/Val/Test split clear
- [ ] Effective N and power analysis
- [ ] Confidence intervals (not just p-values)
- [ ] Robustness checks passed
- [ ] Limitations acknowledged

---

*This plan ensures systematic discovery of what SI actually measures.*

*Last Updated: January 17, 2026*
