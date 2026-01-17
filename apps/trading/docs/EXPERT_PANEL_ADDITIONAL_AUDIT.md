# Expert Panel: Additional Audit Recommendations

**Date**: January 17, 2026
**Purpose**: Experts suggest what additional audits are needed before implementation

---

## 📋 Audit Summary So Far

| Audit | Issues Found | Status |
|-------|--------------|--------|
| Methodology Audit | 16 | ✅ Fixed |
| Expert Panel Review | 18 | ✅ Incorporated |
| Feature Pipeline Audit | 6 | ✅ Fixed |
| **Total Fixed** | **40** | |

---

## 🎓 Expert Panel Convened

### Panel Composition

| Expert | Domain | Focus Area |
|--------|--------|------------|
| Prof. Causal Inference | Statistics | Causality vs correlation |
| Prof. Experimental Economics | Behavioral | Incentive structures |
| Prof. Financial Econometrics | Finance | Time series validity |
| Prof. Software Engineering | CS | Implementation correctness |
| Quant Researcher (Hedge Fund) | Industry | Practical trading |
| ML Ops Engineer | Industry | Reproducibility |
| Research Integrity Officer | Academic | Ethical concerns |
| Domain Expert (Crypto) | Industry | Crypto-specific issues |

---

## 🔍 Additional Audit Areas Identified

### 1. CAUSAL INFERENCE AUDIT (Prof. Causal Inference)

**Current Gap**: We're testing correlations, but claiming SI is a "signal." Signals imply causality.

**Questions to Audit**:

| Question | Current Answer | Issue |
|----------|----------------|-------|
| Does SI → Feature or Feature → SI? | Unknown | Need Granger both directions |
| Could there be a common cause? | Not tested | Need instrumental variables |
| Is the correlation spurious? | Partially addressed | Need more robustness |

**Recommended Additional Tests**:

```python
# 1. Bidirectional Granger causality
def audit_causality_direction(si, feature):
    """
    Test: SI → Feature vs Feature → SI
    """
    si_causes_feature = granger_causality(si, feature)
    feature_causes_si = granger_causality(feature, si)

    return {
        'si_causes_feature': si_causes_feature['p'] < 0.05,
        'feature_causes_si': feature_causes_si['p'] < 0.05,
        'interpretation': interpret_causality(si_causes_feature, feature_causes_si)
    }

# 2. Placebo test with synthetic SI
def audit_placebo(real_si, features, n_placebos=100):
    """
    Generate random walks with same properties as SI.
    If random SI also correlates, our finding is spurious.
    """
    placebo_correlations = []
    for _ in range(n_placebos):
        fake_si = generate_random_walk_like(real_si)
        r, _ = spearmanr(fake_si, features)
        placebo_correlations.append(r)

    # Real correlation should be outside 95% of placebos
    return {
        'real_r': spearmanr(real_si, features)[0],
        'placebo_95_ci': np.percentile(placebo_correlations, [2.5, 97.5]),
        'significant': abs(real_r) > np.percentile(placebo_correlations, 97.5)
    }

# 3. Instrumental variable (if available)
# Hard in trading context, but consider:
# - Exchange-specific liquidity shocks as instrument for volatility
# - Time zone effects as instrument for volume
```

**Priority**: 🔴 HIGH - Without causality audit, we can't claim SI is a "signal"

---

### 2. IMPLEMENTATION CORRECTNESS AUDIT (Prof. Software Engineering)

**Current Gap**: We have plans but no code review.

**Questions to Audit**:

| Area | Risk | Audit Method |
|------|------|--------------|
| SI calculation | Off-by-one errors | Unit tests |
| Feature timing | Lookahead bugs | Time travel tests |
| Data alignment | Misaligned timestamps | Assertion checks |
| Numerical stability | NaN/Inf propagation | Edge case tests |

**Recommended Audits**:

```python
# 1. Unit test for SI calculation
def test_si_calculation():
    """
    Known input → Known output
    """
    # Create agents with known specialization
    agents = [
        MockAgent(niche_affinity=[1.0, 0.0, 0.0]),  # Specialist A
        MockAgent(niche_affinity=[0.0, 1.0, 0.0]),  # Specialist B
        MockAgent(niche_affinity=[0.33, 0.33, 0.34]),  # Generalist
    ]

    # SI should be high (2 specialists + 1 generalist)
    si = calculate_si(agents)

    # Known answer
    expected_si = 0.666  # Based on entropy formula
    assert abs(si - expected_si) < 0.01, f"SI wrong: {si} != {expected_si}"

# 2. Time travel test (CRITICAL)
def test_no_lookahead():
    """
    Ensure no feature uses future data.
    """
    data = load_data()

    # Truncate data at t
    t = data.index[1000]
    data_up_to_t = data.loc[:t]

    # Calculate features
    features_full = calculate_features(data)
    features_truncated = calculate_features(data_up_to_t)

    # Features at t should be IDENTICAL
    for col in features_truncated.columns:
        if col in LOOKAHEAD_FEATURES:
            continue  # Skip known lookahead

        full_val = features_full.loc[t, col]
        trunc_val = features_truncated.loc[t, col]

        assert full_val == trunc_val, f"LOOKAHEAD BUG in {col}!"

# 3. Numerical stability test
def test_edge_cases():
    """
    Test with extreme/edge inputs.
    """
    # All agents identical → SI should be 0 or NaN (handled)
    identical_agents = [MockAgent(niche=[0.5, 0.5])] * 10
    si = calculate_si(identical_agents)
    assert not np.isnan(si), "SI should handle identical agents"

    # Single agent → SI undefined
    single_agent = [MockAgent(niche=[1.0, 0.0])]
    si = calculate_si(single_agent)
    assert np.isnan(si) or si == 0, "SI should handle single agent"

    # Zero returns → features should handle
    zero_returns = pd.Series([0.0] * 1000)
    vol = calculate_volatility(zero_returns)
    assert not np.isnan(vol) and not np.isinf(vol), "Volatility should handle zeros"
```

**Priority**: 🔴 HIGH - Bugs in implementation invalidate all results

---

### 3. REPRODUCIBILITY AUDIT (ML Ops Engineer)

**Current Gap**: No reproducibility verification.

**Questions to Audit**:

| Area | Risk | Mitigation |
|------|------|------------|
| Random seeds | Different results each run | Lock all seeds |
| Package versions | Dependency drift | Lock requirements.txt |
| Data versioning | Data changed since run | Hash and version data |
| Code versioning | Code changed since run | Git commit hash in results |

**Recommended Checks**:

```python
# 1. Reproducibility manifest
def create_reproducibility_manifest():
    """
    Record everything needed to reproduce.
    """
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'git_dirty': len(subprocess.check_output(['git', 'status', '--porcelain'])) > 0,
        'python_version': sys.version,
        'package_versions': {
            pkg.key: pkg.version
            for pkg in pkg_resources.working_set
        },
        'random_seeds': {
            'numpy': np.random.get_state()[1][0],
            'python': random.getstate()[1][0],
        },
        'data_hashes': {
            'btc': hashlib.md5(btc_data.to_csv().encode()).hexdigest(),
            'eth': hashlib.md5(eth_data.to_csv().encode()).hexdigest(),
        },
        'config': CONFIG,
    }

    with open('reproducibility_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest

# 2. Reproducibility test
def test_reproducibility():
    """
    Run experiment twice with same seeds → same results.
    """
    set_all_seeds(42)
    results_1 = run_experiment()

    set_all_seeds(42)
    results_2 = run_experiment()

    assert results_1 == results_2, "Experiment not reproducible!"

# 3. requirements.txt with exact versions
# numpy==1.24.3
# pandas==2.0.3
# scipy==1.11.1
# ... (pin ALL versions)
```

**Priority**: 🟡 MEDIUM - Important for credibility but not correctness

---

### 4. CRYPTO-SPECIFIC AUDIT (Domain Expert - Crypto)

**Current Gap**: Treating crypto like traditional assets.

**Questions to Audit**:

| Issue | Traditional | Crypto | Impact |
|-------|-------------|--------|--------|
| Trading hours | 9:30-4:00 | 24/7 | Different volatility patterns |
| Weekends | Closed | Open | Weekend effects |
| Liquidity | Deep | Varies wildly | Slippage |
| Market structure | Centralized | Fragmented | Price discrepancies |
| Manipulation | Regulated | Common | Outliers |

**Recommended Checks**:

```python
# 1. Time-of-day effects
def audit_time_patterns():
    """
    Check if SI-feature correlations vary by time.
    """
    data['hour'] = data.index.hour

    # Asian hours (00:00-08:00 UTC)
    asian = data[data['hour'].between(0, 8)]
    # European hours (08:00-16:00 UTC)
    european = data[data['hour'].between(8, 16)]
    # US hours (16:00-24:00 UTC)
    us = data[data['hour'].between(16, 24)]

    results = {}
    for session, subset in [('asian', asian), ('european', european), ('us', us)]:
        r, p = spearmanr(subset['si'], subset['volatility'])
        results[session] = {'r': r, 'p': p}

    # Check if correlations are consistent
    return results

# 2. Weekend effects
def audit_weekend():
    """
    Check if patterns differ on weekends.
    """
    data['is_weekend'] = data.index.dayofweek >= 5

    weekday_r = spearmanr(data[~data['is_weekend']]['si'], ...)
    weekend_r = spearmanr(data[data['is_weekend']]['si'], ...)

    return {
        'weekday': weekday_r,
        'weekend': weekend_r,
        'significant_difference': test_difference(weekday_r, weekend_r)
    }

# 3. Liquidity regime check
def audit_liquidity_regimes():
    """
    Check if correlations change with liquidity.
    """
    high_liquidity = data[data['volume'] > data['volume'].median()]
    low_liquidity = data[data['volume'] <= data['volume'].median()]

    # Correlations may only hold in liquid markets
    return {
        'high_liq_r': spearmanr(high_liquidity['si'], ...),
        'low_liq_r': spearmanr(low_liquidity['si'], ...),
    }

# 4. Manipulation/outlier check
def audit_outliers():
    """
    Identify and assess impact of extreme events.
    """
    # Flash crashes, exchange hacks, etc.
    returns = data['close'].pct_change()
    extreme = abs(returns) > returns.std() * 5  # 5-sigma events

    # Correlation WITH outliers
    r_with = spearmanr(data['si'], data['feature'])

    # Correlation WITHOUT outliers
    r_without = spearmanr(data[~extreme]['si'], data[~extreme]['feature'])

    return {
        'with_outliers': r_with,
        'without_outliers': r_without,
        'outlier_driven': abs(r_with[0] - r_without[0]) > 0.1
    }
```

**Priority**: 🟡 MEDIUM - Crypto-specific issues may invalidate generalization

---

### 5. STRATEGY VALIDITY AUDIT (Quant Researcher)

**Current Gap**: Using "toy" strategies that may not reflect real trading.

**Questions to Audit**:

| Issue | Risk | Check |
|-------|------|-------|
| Strategy parameters | Overfitted | Out-of-sample test |
| Rebalancing frequency | Unrealistic | Transaction cost sensitivity |
| Position sizing | Arbitrary | Kelly criterion comparison |
| Benchmark | None | Compare to buy-and-hold |

**Recommended Checks**:

```python
# 1. Strategy parameter sensitivity
def audit_strategy_parameters():
    """
    Are results sensitive to strategy parameters?
    """
    results = {}

    # Vary momentum lookback
    for lookback in [3, 7, 14, 30]:
        strategy = MomentumStrategy(lookback=lookback)
        si = run_and_compute_si(strategy)
        r = spearmanr(si, volatility)
        results[f'momentum_{lookback}'] = r

    # If results vary wildly, findings are fragile
    r_values = [v[0] for v in results.values()]
    return {
        'results': results,
        'stable': max(r_values) - min(r_values) < 0.2
    }

# 2. Benchmark comparison
def audit_vs_benchmark():
    """
    Compare SI-based strategies to simple benchmarks.
    """
    benchmarks = {
        'buy_hold': lambda data: data['close'][-1] / data['close'][0] - 1,
        'equal_weight': lambda data: compute_equal_weight_return(data),
        'random': lambda data: simulate_random_trading(data),
    }

    results = {}
    for name, benchmark in benchmarks.items():
        results[name] = benchmark(data)

    results['our_best'] = run_best_si_strategy(data)

    return results

# 3. Transaction cost sensitivity
def audit_transaction_costs():
    """
    Does profitability survive realistic costs?
    """
    cost_scenarios = {
        'zero': 0.0,
        'low': 0.001,      # 0.1% (maker)
        'medium': 0.003,   # 0.3% (retail taker)
        'high': 0.005,     # 0.5% (adverse)
    }

    results = {}
    for name, cost in cost_scenarios.items():
        profit = run_with_costs(cost)
        results[name] = profit

    return {
        'results': results,
        'breakeven_cost': find_breakeven_cost(results)
    }
```

**Priority**: 🔴 HIGH - Invalid strategies → invalid SI

---

### 6. ETHICAL/INTEGRITY AUDIT (Research Integrity Officer)

**Current Gap**: No pre-registration, potential for p-hacking.

**Questions to Audit**:

| Risk | Mitigation | Status |
|------|------------|--------|
| P-hacking | Pre-registration | 📝 Planned |
| HARKing | Commit hypothesis before analysis | 📝 Planned |
| Selective reporting | Report ALL results | 📝 Planned |
| Data snooping | Strict train/val/test | ✅ Planned |

**Recommended Safeguards**:

```markdown
## Pre-Registration Checklist

### Before ANY Analysis:
- [ ] Document primary hypothesis
- [ ] Document analysis plan
- [ ] Commit to GitHub with timestamp
- [ ] Do NOT look at test set

### During Analysis:
- [ ] Follow pre-registered plan
- [ ] Document any deviations
- [ ] Record ALL results (including null)

### After Analysis:
- [ ] Report pre-registered results FIRST
- [ ] Clearly label exploratory vs confirmatory
- [ ] Include all null results
- [ ] Publish regardless of outcome
```

**Priority**: 🔴 HIGH - Without this, credibility is compromised

---

### 7. MULTI-ASSET VALIDITY AUDIT (Prof. Financial Econometrics)

**Current Gap**: Testing on BTC only or small set.

**Questions to Audit**:

| Issue | Risk | Check |
|-------|------|-------|
| Asset selection | Cherry-picking | Test all major coins |
| Time period | Lucky period | Test multiple periods |
| Market regime | Bull-only or bear-only | Test both |

**Recommended Checks**:

```python
# 1. Multi-asset generalization
ASSETS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOT', 'AVAX']

def audit_multi_asset():
    """
    Do findings generalize across assets?
    """
    results = {}
    for asset in ASSETS:
        data = load_data(asset)
        si = compute_si(data)
        r, p = spearmanr(si, data['volatility'])
        results[asset] = {'r': r, 'p': p}

    # Findings should be consistent
    positive_r = sum(1 for v in results.values() if v['r'] > 0)
    significant = sum(1 for v in results.values() if v['p'] < 0.05)

    return {
        'results': results,
        'consistent_direction': positive_r >= len(ASSETS) * 0.8,
        'mostly_significant': significant >= len(ASSETS) * 0.5
    }

# 2. Time period robustness
def audit_time_periods():
    """
    Do findings hold across different periods?
    """
    periods = {
        '2023_bull': ('2023-01-01', '2023-12-31'),
        '2022_bear': ('2022-01-01', '2022-12-31'),
        '2024_mixed': ('2024-01-01', '2024-12-31'),
    }

    results = {}
    for name, (start, end) in periods.items():
        data = load_data('BTC', start, end)
        si = compute_si(data)
        r, p = spearmanr(si, data['volatility'])
        results[name] = {'r': r, 'p': p}

    return results

# 3. Market regime analysis
def audit_market_regimes():
    """
    Check if SI correlations differ by market regime.
    """
    # Define regimes
    data['regime'] = classify_regime(data)  # bull/bear/sideways

    results = {}
    for regime in ['bull', 'bear', 'sideways']:
        subset = data[data['regime'] == regime]
        r, p = spearmanr(subset['si'], subset['volatility'])
        results[regime] = {'r': r, 'p': p}

    return results
```

**Priority**: 🟡 MEDIUM - Important for generalization claims

---

### 8. ADVERSARIAL AUDIT (Prof. Experimental Economics)

**Current Gap**: No adversarial testing.

**Questions to Audit**:

| Attack | Description | Defense |
|--------|-------------|---------|
| Confirmation bias | Only seeing what we want | Devil's advocate analysis |
| Overfitting | Finding noise | Cross-validation |
| Specification search | Trying until p<0.05 | Pre-registration |

**Recommended Tests**:

```python
# 1. Devil's advocate: Try to DISPROVE the hypothesis
def audit_devils_advocate():
    """
    Actively try to find counterexamples.
    """
    # Find conditions where SI-volatility correlation BREAKS
    subsets_where_negative = []

    for hour in range(24):
        for day in range(7):
            subset = data[(data.index.hour == hour) & (data.index.dayofweek == day)]
            if len(subset) < 100:
                continue
            r, p = spearmanr(subset['si'], subset['volatility'])
            if r < 0 and p < 0.05:
                subsets_where_negative.append((hour, day, r, p))

    return {
        'counterexamples_found': len(subsets_where_negative),
        'details': subsets_where_negative
    }

# 2. Random feature control
def audit_random_features():
    """
    Add random features. SI should NOT correlate with them.
    """
    for i in range(10):
        random_feature = np.random.randn(len(data))
        r, p = spearmanr(data['si'], random_feature)

        if p < 0.05:
            print(f"WARNING: SI correlates with random feature {i}!")

    # Should rarely happen (5% false positive rate)

# 3. Permutation test
def audit_permutation():
    """
    Shuffle SI labels and check if correlation persists.
    """
    real_r = spearmanr(data['si'], data['volatility'])[0]

    shuffled_rs = []
    for _ in range(1000):
        shuffled_si = np.random.permutation(data['si'])
        r = spearmanr(shuffled_si, data['volatility'])[0]
        shuffled_rs.append(r)

    p_permutation = np.mean(np.abs(shuffled_rs) >= np.abs(real_r))

    return {
        'real_r': real_r,
        'permutation_p': p_permutation,
        'significant': p_permutation < 0.05
    }
```

**Priority**: 🟡 MEDIUM - Builds confidence in findings

---

## 📋 Summary: Additional Audits Required

| Audit | Priority | Effort | When |
|-------|----------|--------|------|
| **Causal Inference** | 🔴 High | 2 days | Before claiming "signal" |
| **Implementation** | 🔴 High | 1 day | Before running experiments |
| **Strategy Validity** | 🔴 High | 1 day | Before trusting SI |
| **Ethical/Integrity** | 🔴 High | 0.5 day | Before any analysis |
| Reproducibility | 🟡 Medium | 0.5 day | Before final report |
| Crypto-Specific | 🟡 Medium | 1 day | During analysis |
| Multi-Asset | 🟡 Medium | 1 day | After initial findings |
| Adversarial | 🟡 Medium | 1 day | After initial findings |

### Total Additional Effort: ~8 days

---

## 🎯 Expert Panel Verdict

### Prof. Causal Inference:
> "The biggest gap is claiming SI is a 'signal' without testing causality. Add bidirectional Granger and placebo tests."

### Prof. Software Engineering:
> "No amount of statistical rigor matters if the code is buggy. Add unit tests for SI calculation and time-travel tests for lookahead."

### Quant Researcher:
> "Your strategies are academic toys. Add transaction cost sensitivity and benchmark comparisons before trusting SI."

### Research Integrity Officer:
> "Pre-registration is non-negotiable. Commit your hypothesis to GitHub BEFORE looking at any data."

### ML Ops Engineer:
> "Pin all package versions and save reproducibility manifests. Otherwise, you can't defend your results."

### Prof. Financial Econometrics:
> "Test across multiple assets and time periods. Findings on BTC alone are not generalizable."

### Domain Expert (Crypto):
> "Don't treat crypto like stocks. Check for time-of-day effects, weekend patterns, and liquidity regimes."

### Prof. Experimental Economics:
> "Add adversarial tests. If you don't try to disprove your hypothesis, reviewers will."

---

## ✅ Updated Audit Checklist

### Phase 1: Pre-Experiment (BEFORE ANALYSIS)
- [ ] Pre-register hypothesis (GitHub commit)
- [ ] Write unit tests for SI calculation
- [ ] Write time-travel tests for features
- [ ] Pin all package versions
- [ ] Create reproducibility manifest template

### Phase 2: During Experiment
- [ ] Run causal tests (bidirectional Granger, placebo)
- [ ] Run crypto-specific checks (time-of-day, weekend, liquidity)
- [ ] Run strategy validity checks (parameters, benchmarks, costs)
- [ ] Document all deviations from pre-registration

### Phase 3: Post-Experiment
- [ ] Run multi-asset generalization
- [ ] Run adversarial tests (devil's advocate, permutation)
- [ ] Report ALL results including null
- [ ] Save reproducibility manifest

---

*This expert panel review adds 8 additional audit areas with ~8 days of effort. Prioritize the HIGH items before proceeding.*

*Last Updated: January 17, 2026*
