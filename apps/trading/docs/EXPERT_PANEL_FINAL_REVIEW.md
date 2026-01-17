# Expert Panel Final Review: SI Correlation Methodology

**Date**: January 17, 2026
**Purpose**: Final audit by experts and professors after all issues fixed

---

## 📋 Context for Reviewers

We have developed a methodology to discover what Specialization Index (SI) correlates with in trading. After an initial audit identified 16 issues, all have been addressed. We now request a final review.

### Documents to Review:
1. `SI_EXPLORATION.md` - 40 hypotheses about what SI might measure
2. `SI_CORRELATION_TEST_PLAN.md` - Detailed test plan with all fixes
3. `METHODOLOGY_AUDIT.md` - Initial audit and solutions implemented

---

## 🔬 Expert Panel

### Reviewers

| # | Expert | Domain | Focus Area |
|---|--------|--------|------------|
| 1 | 📊 Prof. Statistics | Econometrics | Time series, multiple testing |
| 2 | 🧪 Prof. Experimental Design | Causal inference | Validity, confounding |
| 3 | 💹 Prof. Quantitative Finance | Market microstructure | Trading-specific issues |
| 4 | 🤖 Prof. Machine Learning | Model validation | Overfitting, generalization |
| 5 | 🧠 Prof. Behavioral Economics | Cognitive biases | Researcher bias |
| 6 | 📈 Hedge Fund Quant | Practical trading | Real-world applicability |
| 7 | 🔢 Risk Manager | Financial risk | Risk metric validity |
| 8 | 🎯 Research Methodologist | Scientific rigor | Reproducibility |

---

## ✅ Issues Already Addressed

The following 16 issues were identified and fixed:

| # | Issue | Fix Implemented |
|---|-------|-----------------|
| 1 | No train/test split | 70/15/15 temporal split |
| 2 | Autocorrelation ignored | HAC standard errors + block bootstrap |
| 3 | Multiple testing underestimated | Pre-registration + FDR |
| 4 | Circular reasoning | Predictive (lagged) correlations |
| 5 | No power analysis | Effective N estimation + min r |
| 6 | Look-ahead bias | Feature categorization |
| 7 | Confounding variables | Partial correlations |
| 8 | Non-stationarity | Stationarity checks |
| 9 | Feature multicollinearity | Hierarchical clustering |
| 10 | Regime definition circular | Multiple regime definitions |
| 11 | No baseline comparison | Random permutation baseline |
| 12 | Single asset class | Acknowledged limitation |
| 13 | Survivorship bias | Include mixed performers |
| 14 | Reproducibility | Config file saved |
| 15 | Effect size interpretation | Guidelines added |
| 16 | Reporting standards | Checklist added |

---

## 🎤 Expert Reviews

### 📊 Prof. Statistics (Econometrics)

**Review of statistical methodology:**

✅ **Strengths:**
- HAC standard errors are appropriate for autocorrelated time series
- Block bootstrap is the correct approach for confidence intervals
- FDR correction is standard for multiple testing
- Pre-registration addresses p-hacking concerns

⚠️ **Concerns:**

1. **Effective sample size estimation needs specification**
   - How exactly will you estimate effective N?
   - Recommend: Use Bartlett's formula: N_eff = N / (1 + 2*Σρ_k)
   - Or fit AR(1) and use: N_eff = N * (1-ρ)/(1+ρ)

2. **Block size selection is arbitrary**
   - Block size = 24 hours is assumed, but should be data-driven
   - Recommend: Use optimal block length selection (e.g., Politis & White 2004)

3. **Granger causality assumptions**
   - Granger test requires stationarity of BOTH series
   - If one is non-stationary, use Toda-Yamamoto procedure

**Recommendation**: Add block size optimization and specify effective N formula.

---

### 🧪 Prof. Experimental Design (Causal Inference)

**Review of causal validity:**

✅ **Strengths:**
- Temporal split prevents data leakage
- Partial correlations control for confounders
- Predictive correlations address circular reasoning

⚠️ **Concerns:**

1. **Confounder list may be incomplete**
   - You control for "time_index, training_iteration, cumulative_data"
   - Missing: Market cap, overall market regime, trading volume surge
   - Missing: External events (halving, regulation news, macro)

2. **Selection of confounders is itself a researcher degree of freedom**
   - Document confounder selection rationale BEFORE analysis

3. **Mediation vs confounding not distinguished**
   - If SI → Volatility → Profit, volatility is a MEDIATOR not confounder
   - Don't control for mediators!
   - Recommend: Add mediation analysis framework

4. **Temporal precedence is necessary but not sufficient for causality**
   - SI leads Feature doesn't prove SI causes Feature
   - Could be: Unobserved Z causes both with different lags

**Recommendation**: Add mediation analysis, expand confounder list, document selection rationale.

---

### 💹 Prof. Quantitative Finance (Market Microstructure)

**Review of trading-specific issues:**

✅ **Strengths:**
- Multiple regime definitions is excellent
- Cross-asset analysis is valuable
- Survivorship consideration is important

⚠️ **Concerns:**

1. **Crypto-specific biases not fully addressed**
   - 24/7 trading: No market close effects to exploit
   - Retail-dominated: Different dynamics than institutional markets
   - Highly correlated: BTC/ETH/SOL move together
   - Recommend: Report intra-asset correlations

2. **Transaction costs ignored**
   - If SI correlation is r=0.3 with profit, does it survive transaction costs?
   - Recommend: Add transaction cost sensitivity analysis

3. **Liquidity variation**
   - SI might correlate with liquidity, not with anything tradeable
   - High liquidity → strategies work → high SI (spurious)
   - Recommend: Control for liquidity explicitly

4. **Time-of-day effects**
   - SI might vary by hour (Asian/European/US sessions)
   - Recommend: Check for time-of-day patterns

**Recommendation**: Add transaction cost analysis, liquidity control, time-of-day analysis.

---

### 🤖 Prof. Machine Learning (Model Validation)

**Review of validation approach:**

✅ **Strengths:**
- Train/val/test split is standard
- Robustness checks across assets/time periods
- Random baseline comparison is excellent

⚠️ **Concerns:**

1. **Single split is not robust**
   - One 70/15/15 split may hit lucky/unlucky periods
   - Recommend: Rolling/expanding window validation
   ```
   Split 1: [Train: M1-M8] [Val: M9-M10] [Test: M11-M12]
   Split 2: [Train: M2-M9] [Val: M10-M11] [Test: M12-M13]
   ...
   Report: Mean and std across splits
   ```

2. **Feature selection before split is data leakage**
   - If you cluster features using ALL data, then split, you've leaked
   - Recommend: Cluster using ONLY train data

3. **Hyperparameter selection not addressed**
   - Block size, window sizes, FDR threshold are hyperparameters
   - If you tune these on val set, you need a separate test set
   - Recommend: Fix all hyperparameters BEFORE looking at any data

4. **Model complexity not penalized**
   - Testing 70 features is "complex" vs testing 10
   - Recommend: Consider using AIC/BIC or regularization

**Recommendation**: Add rolling validation, fix hyperparameters before analysis, cluster only on train.

---

### 🧠 Prof. Behavioral Economics (Cognitive Biases)

**Review of researcher bias:**

✅ **Strengths:**
- Pre-registration addresses confirmation bias
- Multiple testing correction is good
- Effect size guidelines prevent overinterpretation

⚠️ **Concerns:**

1. **Hypothesis generation bias**
   - 40 hypotheses were generated AFTER seeing trading data patterns
   - Even if not looking at SI correlations, familiarity with data biases
   - Recommend: Document when hypotheses were generated

2. **Selective reporting risk**
   - Plan to "identify candidates" then "confirm" is fine
   - But temptation to adjust criteria post-hoc is strong
   - Recommend: Publish pre-registration publicly (OSF, GitHub commit)

3. **Confirmation bias in interpretation**
   - If SI correlates with volatility, you might interpret as "regime clarity"
   - But it could just mean "when vol is low, SI is high" (trivial)
   - Recommend: Have adversarial interpretation step

4. **Anchoring on initial results**
   - First significant result will anchor your thinking
   - Recommend: Analyze in random order, not sorted by p-value

**Recommendation**: Publish pre-registration publicly, add adversarial interpretation.

---

### 📈 Hedge Fund Quant (Practical Trading)

**Review of practical applicability:**

✅ **Strengths:**
- Focus on actionability is right
- Effect size thresholds are realistic
- Multiple asset check is important

⚠️ **Concerns:**

1. **Stale SI problem**
   - SI is computed from past N hours of competition
   - By the time you know SI is high, the opportunity may be gone
   - Recommend: Test how quickly SI changes, signal decay analysis

2. **Execution gap**
   - Even if SI predicts next-day return, can you capture it?
   - Recommend: Simulate realistic execution (slippage, partial fills)

3. **Capacity constraints**
   - Crypto liquidity is limited, especially for SOL
   - If SI signal works, how much capital can it support?
   - Recommend: Estimate strategy capacity

4. **Regime change detection**
   - Historical analysis assumes you know the regime
   - In real-time, regime detection is noisy
   - Recommend: Use only real-time-available features

5. **Alpha decay**
   - If this research is published, alpha will decay
   - Recommend: Estimate uniqueness/crowdedness of finding

**Recommendation**: Add signal decay analysis, execution simulation, capacity estimation.

---

### 🔢 Risk Manager (Financial Risk)

**Review of risk considerations:**

✅ **Strengths:**
- VaR and CVaR included as features
- Tail ratio is useful
- Drawdown recovery time is practical

⚠️ **Concerns:**

1. **Risk metrics computed on same window as SI**
   - SI uses 24h window, risk uses 30d
   - Mismatch could cause spurious correlations
   - Recommend: Align windows or document mismatch

2. **Crisis periods underrepresented**
   - 12 months might not include major crash
   - Recommend: If possible, include COVID crash, 2022 crypto winter

3. **Tail risk correlation is most valuable but hardest**
   - Few tail events = low power for tail analysis
   - Recommend: Report tail analysis separately with power caveat

4. **Position sizing implications unclear**
   - If SI correlates with risk, HOW should position sizes adjust?
   - Recommend: Add practical position sizing recommendations

**Recommendation**: Include crisis periods if data available, align windows, add position sizing guidance.

---

### 🎯 Research Methodologist (Scientific Rigor)

**Review of scientific rigor:**

✅ **Strengths:**
- Pre-registration is excellent
- Checklist approach ensures consistency
- Config file ensures reproducibility

⚠️ **Concerns:**

1. **Lack of negative controls**
   - You test if SI correlates with meaningful things
   - But do you test if SI correlates with MEANINGLESS things?
   - Recommend: Add negative controls (e.g., SI vs yesterday's weather)

2. **Publication bias potential**
   - If you find nothing, will you publish?
   - Recommend: Commit to publishing null results

3. **Code and data availability**
   - Reproducibility requires sharing code AND data
   - Recommend: Plan for code release, data access instructions

4. **Version control**
   - Analysis code should be version controlled
   - Recommend: Commit analysis code BEFORE running on test set

5. **External validation**
   - Even with train/val/test, all is same time period, same market
   - Recommend: Plan for external validation (different time, different market)

**Recommendation**: Add negative controls, commit to publishing null results, version control analysis code.

---

## 📋 Summary of New Recommendations

| # | Recommendation | Source | Priority |
|---|----------------|--------|----------|
| 1 | Specify effective N formula (Bartlett/AR(1)) | Statistics | 🔴 High |
| 2 | Optimize block size data-driven | Statistics | 🟡 Medium |
| 3 | Add mediation analysis framework | Experimental Design | 🟡 Medium |
| 4 | Expand confounder list (market cap, macro) | Experimental Design | 🔴 High |
| 5 | Transaction cost sensitivity | Quant Finance | 🔴 High |
| 6 | Control for liquidity explicitly | Quant Finance | 🔴 High |
| 7 | Time-of-day analysis | Quant Finance | 🟡 Medium |
| 8 | Rolling/expanding validation | ML | 🔴 High |
| 9 | Cluster features on train only | ML | 🔴 High |
| 10 | Fix hyperparameters before analysis | ML | 🔴 High |
| 11 | Publish pre-registration publicly | Behavioral | 🔴 High |
| 12 | Adversarial interpretation step | Behavioral | 🟡 Medium |
| 13 | Signal decay analysis | Hedge Fund | 🔴 High |
| 14 | Execution simulation | Hedge Fund | 🟡 Medium |
| 15 | Include crisis periods in data | Risk | 🟡 Medium |
| 16 | Add negative controls | Methodologist | 🔴 High |
| 17 | Commit to publishing null results | Methodologist | 🟢 Low |
| 18 | Version control analysis code | Methodologist | 🟢 Low |

---

## 🔴 Critical Additions Required

Based on expert feedback, these must be added:

### 1. Effective Sample Size Formula

```python
def estimate_effective_n(series, method='bartlett'):
    """Estimate effective sample size adjusting for autocorrelation."""
    n = len(series)

    if method == 'bartlett':
        # Bartlett's formula: N_eff = N / (1 + 2*Σρ_k)
        from statsmodels.tsa.stattools import acf
        acf_vals = acf(series, nlags=min(50, n//4), fft=True)
        # Sum until first non-significant lag
        rho_sum = sum(acf_vals[1:])
        n_eff = n / (1 + 2 * rho_sum)

    elif method == 'ar1':
        # AR(1) approximation: N_eff = N * (1-ρ)/(1+ρ)
        rho = series.autocorr(lag=1)
        n_eff = n * (1 - rho) / (1 + rho)

    return max(1, int(n_eff))
```

### 2. Rolling Validation

```python
def rolling_validation(data, window_months=8, val_months=2, test_months=2):
    """Rolling window validation for robust results."""
    results = []
    total_months = 12
    step = 1  # Slide by 1 month

    for start in range(0, total_months - window_months - val_months - test_months + 1, step):
        train_end = start + window_months
        val_end = train_end + val_months
        test_end = val_end + test_months

        train = data[start:train_end]
        val = data[train_end:val_end]
        test = data[val_end:test_end]

        # Run analysis
        result = run_correlation_analysis(train, val, test)
        results.append(result)

    # Aggregate across folds
    summary = aggregate_results(results)
    return summary
```

### 3. Negative Controls

```python
NEGATIVE_CONTROLS = [
    'random_noise',           # Pure random numbers
    'lagged_unrelated_asset', # S&P 500 lagged 30 days
    'moon_phase',             # Literally the moon phase
    'day_of_week',            # Categorical, no expected relation
]

def run_negative_controls(si, data):
    """SI should NOT correlate with these."""
    results = {}

    # Random noise
    random_noise = np.random.randn(len(si))
    r, p = spearmanr(si, random_noise)
    results['random_noise'] = {'r': r, 'p': p, 'expected_sig': False}

    # If SI correlates with random noise, something is wrong
    if p < 0.05:
        print("WARNING: SI correlates with random noise!")

    return results
```

### 4. Liquidity Control

```python
def add_liquidity_features(data):
    """Add liquidity features to control for."""
    features = {}

    # Volume-based
    features['volume_z'] = (data['volume'] - data['volume'].rolling(168).mean()) / data['volume'].rolling(168).std()

    # Bid-ask proxy (Corwin-Schultz)
    if 'high' in data.columns and 'low' in data.columns:
        features['spread_proxy'] = compute_corwin_schultz_spread(data)

    # Amihud illiquidity
    features['amihud'] = abs(data['close'].pct_change()) / data['volume']

    return features

# Add to confounders list
CONFOUNDERS = ['time_index', 'training_iteration', 'volume_z', 'amihud']
```

### 5. Signal Decay Analysis

```python
def analyze_signal_decay(si, future_returns, max_lag=168):
    """How quickly does SI signal decay?"""
    decay_curve = []

    for lag in range(1, max_lag + 1):
        si_lagged = si.iloc[:-lag]
        returns_future = future_returns.iloc[lag:]

        r, p = spearmanr(si_lagged, returns_future)
        decay_curve.append({'lag_hours': lag, 'correlation': r, 'p_value': p})

    df = pd.DataFrame(decay_curve)

    # Find half-life (lag where correlation drops to 50%)
    max_corr = df['correlation'].iloc[0]
    half_life = df[df['correlation'] < max_corr * 0.5]['lag_hours'].iloc[0] if any(df['correlation'] < max_corr * 0.5) else max_lag

    return df, half_life
```

---

## ✅ Final Checklist After Expert Review

### Before Running Any Analysis
- [ ] Pre-registration published (GitHub commit with timestamp)
- [ ] All hyperparameters fixed
- [ ] Effective N formula specified
- [ ] Confounder list documented with rationale
- [ ] Negative controls defined
- [ ] Crisis periods identified in data

### During Analysis
- [ ] Cluster features using TRAIN data only
- [ ] Use rolling validation (not single split)
- [ ] Include liquidity controls
- [ ] Run negative controls
- [ ] Analyze in random order (not by p-value)

### Interpretation
- [ ] Adversarial interpretation step
- [ ] Check for trivial explanations
- [ ] Signal decay analysis
- [ ] Transaction cost sensitivity
- [ ] Mediation vs confounding check

### Reporting
- [ ] Report all folds, not just aggregate
- [ ] Report null results too
- [ ] Provide code and data access
- [ ] Version control commit hash
- [ ] External validation plan

---

## 🎯 Revised Priority: Additional Issues

| Priority | Count | Issues |
|----------|-------|--------|
| 🔴 High | 8 | Effective N, expand confounders, transaction costs, liquidity control, rolling validation, cluster on train, fix hyperparameters, publish pre-reg, signal decay, negative controls |
| 🟡 Medium | 5 | Optimize block size, mediation analysis, time-of-day, adversarial interpretation, crisis periods, execution simulation |
| 🟢 Low | 2 | Publish null results, version control |

---

*This expert review adds 18 new recommendations on top of the 16 issues already fixed.*

*Total improvements incorporated: 34*

*Status: Ready for implementation after addressing high-priority additions*

*Last Updated: January 17, 2026*
