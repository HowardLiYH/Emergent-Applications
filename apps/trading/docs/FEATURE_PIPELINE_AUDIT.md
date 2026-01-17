# Feature Pipeline Audit: Can All Features Use the Same Validation?

**Date**: January 17, 2026
**Purpose**: Audit whether all 70+ features can go through the same validation pipeline

---

## 🚨 Key Finding: NO, They Cannot All Use the Same Pipeline

Different feature types require **different treatment**. Using one pipeline for all is methodologically flawed.

---

## 📊 Feature Classification by Type

### Type A: Market Features (Exogenous)
**Can use standard pipeline: ✅ YES**

These are computed purely from market data, BEFORE SI is computed.

| Feature | Source | Timing | Issue |
|---------|--------|--------|-------|
| `volatility` | Price | Before SI | None |
| `trend_strength` | Price | Before SI | None |
| `return_entropy` | Price | Before SI | None |
| `volume` | Market | Before SI | None |
| `rsi`, `adx`, `atr` | Price | Before SI | None |
| `hurst_exponent` | Price | Before SI | None |

**Validation**: Standard correlation pipeline ✅

---

### Type B: Agent Features (Endogenous to SI)
**Can use standard pipeline: ⚠️ PROBLEMATIC**

These are computed from agent behavior, which is influenced by SI computation.

| Feature | Source | Issue | Severity |
|---------|--------|-------|----------|
| `agent_correlation` | Agent returns | Agents react to same market → SI | 🟡 Medium |
| `winner_spread` | Agent returns | Winner selection IS SI computation | 🔴 High |
| `viable_agent_count` | Agent states | Depends on competition | 🟡 Medium |
| `strategy_concentration` | Agent choices | This IS specialization | 🔴 High |
| `niche_affinity_entropy` | Agent states | This IS SI (different formula) | 🔴 Critical |

**Problem**: Testing SI against `strategy_concentration` or `niche_affinity_entropy` is testing SI against itself!

```
SI = f(agent behavior)
strategy_concentration = g(agent behavior)

Correlation(SI, strategy_concentration) is essentially:
Correlation(f(X), g(X)) where X = agent behavior

This is NOT testing if SI predicts anything external!
```

**Fix Required**:
- Remove `strategy_concentration`, `niche_affinity_entropy` from correlation tests
- For `agent_correlation`, `winner_spread`: test if they predict FUTURE profit, not just correlate with SI

---

### Type C: Risk Features (Outcome-Based)
**Can use standard pipeline: ⚠️ DEPENDS ON TIMING**

Risk features are computed from returns, which are outcomes.

| Feature | Source | Timing Issue | Fix |
|---------|--------|--------------|-----|
| `max_drawdown` | Past returns | Which period? | Clarify: past 30d only |
| `var_95` | Past returns | OK if historical | ✅ |
| `sharpe_ratio` | Past returns | OK if historical | ✅ |
| `win_rate` | Past trades | Which trades? | Must be PAST trades only |
| `profit_factor` | Past trades | Which trades? | Must be PAST trades only |

**Problem**: If `max_drawdown` includes the current period, it's using concurrent data.

**Fix Required**:
- All risk features must use STRICTLY PAST data
- Define clear cutoff: "risk features use data from t-30d to t-1"

---

### Type D: Lookahead Features (Future Data)
**Can use standard pipeline: ❌ NO - Requires Special Handling**

These features explicitly use FUTURE data.

| Feature | What It Uses | Purpose |
|---------|--------------|---------|
| `next_day_return` | Return at t+1 | Test if SI PREDICTS |
| `next_day_volatility` | Vol at t+1 | Test if SI PREDICTS |

**These are NOT for correlation discovery - they ARE the prediction test!**

**Current Pipeline Problem**:
```python
# WRONG: Treating lookahead like other features
correlate(SI, next_day_return)  # This is fine

# But then also doing:
correlate(SI, volatility)  # Concurrent
correlate(SI, next_day_return)  # Lookahead

# These answer DIFFERENT questions!
```

**Fix Required**:
- Separate lookahead features into "Prediction Tests" category
- Don't include in the same correlation matrix as concurrent features
- Report separately as "SI predictive power"

---

### Type E: SI-Derived Features (Circular)
**Can use standard pipeline: ❌ NO - Fundamentally Circular**

These features are computed FROM SI.

| Feature | Derivation | Issue |
|---------|------------|-------|
| `dsi_dt` | SI velocity | Derivative of SI |
| `si_acceleration` | SI acceleration | Second derivative |
| `si_rolling_std` | SI stability | Variance of SI |
| `si_1h`, `si_4h`, `si_1d` | SI at different scales | Same SI, different window |

**Problem**: Correlating SI with dSI/dt is correlating a variable with its own derivative!

```
Correlation(SI, dSI/dt) doesn't tell us anything external.
It just tells us about SI's time series properties.
```

**What These Actually Test**:
- `dsi_dt`: "Does SI momentum matter?" (not "What does SI correlate with?")
- `si_rolling_std`: "Does SI stability matter?" (different question)

**Fix Required**:
- Move to separate analysis: "SI dynamics analysis"
- Not part of "What does SI correlate with?" question
- These inform HOW to use SI, not WHAT SI measures

---

### Type F: Cross-Asset Features
**Can use standard pipeline: ⚠️ REQUIRES ALIGNMENT**

| Feature | Issue |
|---------|-------|
| `si_btc`, `si_eth` | Different SIs |
| `si_cross_asset_corr` | Correlation of SIs |
| `asset_return_corr` | Market feature, OK |

**Problem**: If running on BTC, correlating SI_BTC with SI_ETH is valid. But correlating SI_BTC with SI_BTC is 1.0 (trivial).

**Fix Required**:
- Only include cross-asset SI when analyzing multi-asset
- Don't include same-asset SI

---

## 📋 Revised Feature Categories

| Category | Features | Pipeline | Treatment |
|----------|----------|----------|-----------|
| **A: Market (Exogenous)** | 15 | Standard | ✅ Normal correlation |
| **B: Agent (Endogenous)** | 10 → **6** | Modified | Remove circular ones |
| **C: Risk (Outcome)** | 10 | Standard | Ensure past-only |
| **D: Lookahead (Prediction)** | 2 | **Separate** | Prediction test, not discovery |
| **E: SI-Derived (Circular)** | 9 → **0** | **Remove** | Separate SI dynamics analysis |
| **F: Cross-Asset** | 8 → **5** | Modified | Remove same-asset SI |

**Original: 70+ features → Revised: ~50 features** (for the correlation discovery pipeline)

---

## 🔧 Required Pipeline Modifications

### Modification 1: Split Into Three Pipelines

```
┌─────────────────────────────────────────────────────────────┐
│                    THREE SEPARATE PIPELINES                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PIPELINE 1: CORRELATION DISCOVERY                          │
│  ┌─────────────────────────────────────────┐                │
│  │ Features: Market (A) + Risk (C) + Safe Agent (B)        │
│  │ Question: What does SI correlate with?                   │
│  │ Method: Spearman + HAC + Bootstrap                       │
│  │ Output: Top correlates of SI                             │
│  └─────────────────────────────────────────┘                │
│                                                              │
│  PIPELINE 2: PREDICTION TESTING                              │
│  ┌─────────────────────────────────────────┐                │
│  │ Features: next_day_return, next_day_vol                  │
│  │ Question: Does SI predict future outcomes?               │
│  │ Method: Lagged correlation, Granger causality            │
│  │ Output: SI predictive power                              │
│  └─────────────────────────────────────────┘                │
│                                                              │
│  PIPELINE 3: SI DYNAMICS ANALYSIS                            │
│  ┌─────────────────────────────────────────┐                │
│  │ Features: dSI/dt, SI_std, SI at multiple scales          │
│  │ Question: How should we USE SI?                          │
│  │ Method: Time series analysis                             │
│  │ Output: Best SI variant, momentum effects                │
│  └─────────────────────────────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Modification 2: Remove Circular Features from Discovery

```python
# Features to REMOVE from correlation discovery
CIRCULAR_FEATURES = [
    # SI-derived (Type E)
    'dsi_dt', 'si_acceleration', 'si_rolling_std',
    'si_1h', 'si_4h', 'si_1d', 'si_1w',

    # Agent-derived that ARE SI (Type B circular)
    'strategy_concentration',  # This IS specialization
    'niche_affinity_entropy',  # This IS SI formula

    # Same-asset SI
    'si_btc' (when testing BTC),
    'si_eth' (when testing ETH),
]

# Keep these agent features (not circular)
VALID_AGENT_FEATURES = [
    'agent_correlation',      # Different question: are agents similar?
    'winner_spread',          # How much better is winner?
    'viable_agent_count',     # How many profitable?
    'return_dispersion',      # How different are returns?
]
```

### Modification 3: Enforce Temporal Ordering

```python
def validate_feature_timing(feature_name, feature_timestamp, si_timestamp):
    """
    Ensure features don't use data from after SI computation.
    """
    if feature_name in LOOKAHEAD_FEATURES:
        # These are SUPPOSED to look ahead
        assert feature_timestamp > si_timestamp
        return 'lookahead'

    elif feature_name in CONCURRENT_FEATURES:
        # Computed at same time as SI
        assert feature_timestamp == si_timestamp
        return 'concurrent'

    elif feature_name in LAGGED_FEATURES:
        # Uses only past data
        assert feature_timestamp < si_timestamp
        return 'lagged'

    else:
        raise ValueError(f"Unknown feature timing: {feature_name}")
```

### Modification 4: Separate Reporting

```python
def generate_report(results):
    """
    Report results from three pipelines separately.
    """
    report = {
        'correlation_discovery': {
            'title': 'What SI Correlates With (Concurrent)',
            'features_tested': len(DISCOVERY_FEATURES),
            'significant': results['discovery']['significant'],
            'top_10': results['discovery']['top_10'],
        },

        'prediction_testing': {
            'title': 'What SI Predicts (Lookahead)',
            'features_tested': len(PREDICTION_FEATURES),
            'significant': results['prediction']['significant'],
            'best_lag': results['prediction']['best_lag'],
        },

        'si_dynamics': {
            'title': 'How to Use SI (Dynamics)',
            'best_variant': results['dynamics']['best_variant'],
            'momentum_effect': results['dynamics']['momentum_effect'],
            'optimal_window': results['dynamics']['optimal_window'],
        },
    }
    return report
```

---

## 📊 Revised Feature List (50 Features for Discovery)

### Category A: Market Features (15) ✅
```python
MARKET_FEATURES = [
    'volatility_24h', 'volatility_7d', 'volatility_30d',
    'trend_strength_7d',
    'return_autocorr_7d',
    'hurst_exponent',
    'return_entropy_7d',
    'volume_24h', 'volume_volatility_7d',
    'jump_frequency_7d',
    'variance_ratio',
    'adx', 'bb_width', 'rsi', 'atr',
]
```

### Category B: Agent Features (6) ⚠️ (4 removed)
```python
AGENT_FEATURES = [
    'agent_correlation',      # Keep: different from SI
    'winner_spread',          # Keep: magnitude, not pattern
    'viable_agent_count',     # Keep: count, not specialization
    'return_dispersion',      # Keep: variance of returns
    'effective_n',            # Keep: diversification measure
    'winner_consistency',     # Keep: stability of winners
]

REMOVED_AGENT_FEATURES = [
    'strategy_concentration',  # REMOVE: IS specialization
    'niche_affinity_entropy',  # REMOVE: IS SI
    'agent_confidence_mean',   # REMOVE: endogenous
    'position_correlation',    # REMOVE: endogenous
]
```

### Category C: Risk Features (10) ✅
```python
RISK_FEATURES = [
    'max_drawdown_30d',       # Past 30 days only
    'var_95_30d',
    'cvar_95_30d',
    'volatility_of_volatility_30d',
    'tail_ratio_30d',
    'drawdown_recovery_days',  # Historical only
    'win_rate_30d',
    'profit_factor_30d',
    'sharpe_ratio_30d',
    'sortino_ratio_30d',
]
```

### Category D: Lookahead Features (2) → **SEPARATE PIPELINE**
```python
PREDICTION_FEATURES = [
    'next_day_return',
    'next_day_volatility',
]
# NOT in discovery pipeline!
```

### Category E: SI-Derived (9) → **SEPARATE PIPELINE**
```python
SI_DYNAMICS_FEATURES = [
    'dsi_dt',
    'si_acceleration',
    'si_rolling_std',
    'si_1h', 'si_4h', 'si_1d', 'si_1w',
    'si_percentile',
]
# NOT in discovery pipeline!
```

### Category F: Cross-Asset Features (5) ⚠️ (3 removed)
```python
CROSS_ASSET_FEATURES = [
    'asset_return_corr',      # Keep: market feature
    'relative_strength',      # Keep: market feature
    'rotation_signal',        # Keep: relative SI change
    'si_cross_asset_corr',    # Keep: different assets
    # For multi-asset analysis only:
    'si_other_asset',         # Keep: e.g., SI_ETH when testing BTC
]

REMOVED_CROSS_ASSET = [
    'si_btc',  # REMOVE when testing BTC (circular)
    'si_eth',  # REMOVE when testing ETH (circular)
    'momentum_factor',  # Move to market features
]
```

### Category G: Behavioral/Timing Features (8) ✅
```python
BEHAVIORAL_FEATURES = [
    'fear_greed_proxy',
    'holding_period_avg',
    'momentum_return_7d',
    'meanrev_return_7d',
    'regime_duration',
    'days_since_regime_change',
]

# Note: 'extreme_high' and 'extreme_low' are SI-derived, move to dynamics
```

### Category H: Liquidity Features (4) ✅ (Added from expert review)
```python
LIQUIDITY_FEATURES = [
    'volume_z',
    'amihud_log',
    'volume_volatility_24h',
    'spread_proxy',  # If available
]
```

---

## 📋 Final Feature Count

| Pipeline | Category | Count | Purpose |
|----------|----------|-------|---------|
| **Discovery** | Market | 15 | What SI correlates with |
| **Discovery** | Agent (safe) | 6 | Agent behavior patterns |
| **Discovery** | Risk | 10 | Risk relationship |
| **Discovery** | Cross-Asset | 5 | Multi-asset patterns |
| **Discovery** | Behavioral | 6 | Timing patterns |
| **Discovery** | Liquidity | 4 | Liquidity control |
| | **Total Discovery** | **46** | |
| **Prediction** | Lookahead | 2 | Does SI predict? |
| **Dynamics** | SI-derived | 9 | How to use SI? |
| | **Total All** | **57** | |

---

## ✅ Validation Checklist

### Before Running Discovery Pipeline:
- [ ] Remove all SI-derived features (dsi_dt, si_std, etc.)
- [ ] Remove circular agent features (strategy_concentration, niche_affinity_entropy)
- [ ] Remove same-asset SI features
- [ ] Verify all risk features use PAST data only
- [ ] Verify temporal ordering of all features

### Separate Pipelines:
- [ ] Run Discovery Pipeline (46 features)
- [ ] Run Prediction Pipeline (2 features)
- [ ] Run SI Dynamics Pipeline (9 features)
- [ ] Report results SEPARATELY

### Interpretation:
- [ ] Discovery: "SI correlates with X"
- [ ] Prediction: "SI predicts Y"
- [ ] Dynamics: "Use SI as Z"

---

## 🔴 Summary of Required Changes

| Change | Reason | Priority |
|--------|--------|----------|
| Split into 3 pipelines | Different questions | 🔴 Critical |
| Remove SI-derived features from discovery | Circular | 🔴 Critical |
| Remove circular agent features | Not testing externals | 🔴 Critical |
| Enforce past-only for risk features | Temporal validity | 🔴 Critical |
| Remove same-asset SI | Trivial correlation | 🟡 High |
| Report pipelines separately | Clarity | 🟡 High |

---

*This audit reveals that the original plan conflated three different questions into one pipeline. Separating them is essential for valid conclusions.*

*Last Updated: January 17, 2026*
