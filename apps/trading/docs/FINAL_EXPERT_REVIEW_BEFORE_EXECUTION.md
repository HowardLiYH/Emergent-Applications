# Final Expert Review Before Execution

**Date**: January 17, 2026
**Purpose**: Last check before starting implementation

---

## 📋 Expert Panel Convened

| Expert | Domain | Focus |
|--------|--------|-------|
| Prof. A | Statistics | Statistical validity |
| Prof. B | ML/AI | Implementation correctness |
| Prof. C | Finance | Trading realism |
| Prof. D | Experimental Design | Methodology gaps |
| Dr. E | Quant Research | Practical concerns |
| Dr. F | Data Science | Data quality |
| Dr. G | Software Engineering | Code robustness |
| Dr. H | Research Ethics | Integrity checks |

---

## ✅ What's Ready (Approved)

| Component | Status | Expert Verdict |
|-----------|--------|----------------|
| Pre-registration | ✅ Created | "Good. Commit before analysis." |
| 3-pipeline design | ✅ Designed | "Correct separation of concerns." |
| 48 methodology fixes | ✅ Documented | "Comprehensive." |
| 8 audit implementations | ✅ Code ready | "Thorough." |
| Feature calculator | ✅ 46 features | "Good coverage." |
| Statistical tests | ✅ HAC, bootstrap, FDR | "Rigorous." |
| MASTER_PLAN structure | ✅ Linear, inline | "LLM-friendly." |

---

## ⚠️ Remaining Gaps Identified

### Gap 1: No Data Validation Step (Prof. F - Data Science)

**Issue**: MASTER_PLAN assumes data is clean. No validation step.

**Risk**: Garbage in, garbage out.

**Recommendation**:
```python
# Add to Phase 2, Step 2.4: Data Validation

def validate_data(df: pd.DataFrame) -> Dict:
    """Validate data quality before analysis."""
    issues = []

    # 1. Check for missing values
    missing_pct = df.isna().mean() * 100
    for col, pct in missing_pct.items():
        if pct > 5:
            issues.append(f"Column {col} has {pct:.1f}% missing values")

    # 2. Check for duplicates
    n_dups = df.index.duplicated().sum()
    if n_dups > 0:
        issues.append(f"Found {n_dups} duplicate timestamps")

    # 3. Check for gaps
    expected_freq = pd.Timedelta('1h')
    actual_gaps = df.index.to_series().diff()
    large_gaps = actual_gaps[actual_gaps > expected_freq * 2]
    if len(large_gaps) > 0:
        issues.append(f"Found {len(large_gaps)} gaps > 2 hours")

    # 4. Check for outliers (returns > 50% in 1 hour)
    returns = df['close'].pct_change()
    extreme = (abs(returns) > 0.5).sum()
    if extreme > 0:
        issues.append(f"Found {extreme} extreme returns (>50% in 1h)")

    # 5. Check date range
    days = (df.index.max() - df.index.min()).days
    if days < 180:
        issues.append(f"Only {days} days of data (recommend 365+)")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'n_rows': len(df),
        'date_range': f"{df.index.min()} to {df.index.max()}",
        'days': days
    }
```

**Priority**: 🔴 HIGH - Add before Phase 3

---

### Gap 2: No Power Analysis (Prof. A - Statistics)

**Issue**: We don't know if we have enough data to detect effects.

**Risk**: May run entire experiment and find nothing due to low power.

**Recommendation**:
```python
# Add to Phase 2, Step 2.5: Power Analysis

def power_analysis(n: int, alpha: float = 0.05, power: float = 0.80) -> Dict:
    """
    Calculate minimum detectable effect size.

    Uses formula: r_min = sqrt(chi2_inv(1-alpha, 1) / n) for correlation
    """
    from scipy.stats import chi2

    # Minimum detectable r for given n
    chi2_crit = chi2.ppf(1 - alpha, 1)
    r_min = np.sqrt(chi2_crit / n)

    # Effective N (accounting for autocorrelation)
    # Assume autocorrelation of 0.3 at lag 1
    rho = 0.3
    n_eff = n * (1 - rho) / (1 + rho)
    r_min_adj = np.sqrt(chi2_crit / n_eff)

    return {
        'n_raw': n,
        'n_effective': int(n_eff),
        'r_min_unadjusted': r_min,
        'r_min_adjusted': r_min_adj,
        'interpretation': (
            f"With {n} samples ({int(n_eff)} effective), "
            f"we can detect |r| >= {r_min_adj:.3f} at alpha={alpha}, power={power}"
        )
    }

# Example: 5000 hourly samples
# power_analysis(5000)
# → "We can detect |r| >= 0.05 at alpha=0.05"
# This is GOOD - our threshold is 0.15
```

**Priority**: 🟡 MEDIUM - Confirms we have enough data

---

### Gap 3: No Baseline Comparison for SI (Dr. E - Quant)

**Issue**: We compute SI but don't compare to a baseline.

**Risk**: SI might always be high/low. Need reference point.

**Recommendation**:
```python
# Add to Phase 3, Step 3.4: SI Baseline

def compute_si_baseline(n_simulations: int = 100) -> Dict:
    """
    Compute SI for random agent populations.
    This establishes what SI looks like WITHOUT meaningful specialization.
    """
    baseline_sis = []

    for _ in range(n_simulations):
        # Random niche affinities (no learning)
        random_affinities = np.random.dirichlet(np.ones(3), size=18)

        # Compute SI
        entropies = []
        for aff in random_affinities:
            p = aff + 1e-10
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(len(p))
            entropies.append(entropy / max_entropy)

        si = 1 - np.mean(entropies)
        baseline_sis.append(si)

    return {
        'baseline_mean': np.mean(baseline_sis),
        'baseline_std': np.std(baseline_sis),
        'baseline_95_ci': list(np.percentile(baseline_sis, [2.5, 97.5])),
        'interpretation': (
            f"Random SI: {np.mean(baseline_sis):.3f} ± {np.std(baseline_sis):.3f}. "
            f"Our SI should be significantly different from this."
        )
    }
```

**Priority**: 🟡 MEDIUM - Provides context for SI values

---

### Gap 4: No Regime Validation (Prof. C - Finance)

**Issue**: Regime classification is arbitrary (trending/mean-reverting/volatile).

**Risk**: If regimes are wrong, SI is meaningless.

**Recommendation**:
```python
# Add to Phase 3, Step 3.5: Regime Validation

def validate_regime_classification(data: pd.DataFrame, regimes: pd.Series) -> Dict:
    """
    Validate that regime classification makes sense.
    """
    returns = data['close'].pct_change()

    # Check regime distribution
    regime_counts = regimes.value_counts(normalize=True)

    # Check regime characteristics
    regime_stats = {}
    for regime in regimes.unique():
        mask = regimes == regime
        regime_stats[regime] = {
            'count': mask.sum(),
            'pct': mask.mean() * 100,
            'mean_return': returns[mask].mean(),
            'volatility': returns[mask].std(),
            'autocorr': returns[mask].autocorr(lag=1),
        }

    # Validation checks
    issues = []

    # 1. No regime should dominate (>80%)
    if regime_counts.max() > 0.80:
        issues.append(f"One regime dominates: {regime_counts.max():.0%}")

    # 2. Each regime should have different characteristics
    vols = [regime_stats[r]['volatility'] for r in regime_stats]
    if max(vols) / min(vols) < 1.5:
        issues.append("Regimes have similar volatility (not distinct)")

    # 3. Regimes should persist (not flip every hour)
    regime_changes = (regimes != regimes.shift()).sum()
    change_rate = regime_changes / len(regimes)
    if change_rate > 0.3:
        issues.append(f"Regimes change too often: {change_rate:.0%} of hours")

    return {
        'regime_stats': regime_stats,
        'valid': len(issues) == 0,
        'issues': issues
    }
```

**Priority**: 🟡 MEDIUM - Ensures regimes are meaningful

---

### Gap 5: No Test Set Lockout Mechanism (Dr. H - Ethics)

**Issue**: Nothing prevents accidentally looking at TEST data.

**Risk**: Data snooping, invalidates results.

**Recommendation**:
```python
# Add to Phase 2, Step 2.3b: Test Set Lockout

class DataLoaderWithLockout:
    """Data loader that prevents premature access to test set."""

    def __init__(self, data_dir: str = "data/bybit"):
        self.data_dir = Path(data_dir)
        self.test_locked = True
        self.lockout_file = Path("results/.test_lockout")

    def unlock_test(self, confirmation: str):
        """
        Unlock test set ONLY after validation is complete.
        Requires explicit confirmation string.
        """
        if confirmation != "I CONFIRM VALIDATION IS COMPLETE":
            raise ValueError("Invalid confirmation. Test remains locked.")

        self.test_locked = False

        # Log unlock event
        with open(self.lockout_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: Test set unlocked\n")

        print("⚠️  TEST SET UNLOCKED - Use responsibly!")

    def load_test(self) -> pd.DataFrame:
        """Load test data (only if unlocked)."""
        if self.test_locked:
            raise PermissionError(
                "TEST SET IS LOCKED!\n"
                "Complete validation first, then call:\n"
                "  loader.unlock_test('I CONFIRM VALIDATION IS COMPLETE')"
            )

        return self._load_split('test')
```

**Priority**: 🟡 MEDIUM - Prevents accidental snooping

---

### Gap 6: No Logging/Checkpointing (Dr. G - Software)

**Issue**: If experiment crashes, we lose all progress.

**Risk**: Wasted compute time, lost results.

**Recommendation**:
```python
# Add to all experiment scripts: Logging & Checkpointing

import logging
from datetime import datetime

def setup_experiment(name: str):
    """Setup logging and checkpointing for experiment."""

    # Create results directory
    results_dir = Path(f"results/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(results_dir / 'experiment.log'),
            logging.StreamHandler()
        ]
    )

    return results_dir

def checkpoint(results_dir: Path, name: str, data: dict):
    """Save checkpoint."""
    import json

    checkpoint_file = results_dir / f"checkpoint_{name}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    logging.info(f"Checkpoint saved: {name}")

def load_checkpoint(results_dir: Path, name: str) -> dict:
    """Load checkpoint if exists."""
    checkpoint_file = results_dir / f"checkpoint_{name}.json"

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)

    return None
```

**Priority**: 🟢 LOW - Nice to have, not critical

---

### Gap 7: No Multi-Asset Execution Plan (Prof. B - ML)

**Issue**: MASTER_PLAN only shows BTC. How to run on multiple assets?

**Risk**: Findings may not generalize.

**Recommendation**:
```python
# Add to Phase 4, Step 4.3: Multi-Asset Loop

ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']

def run_multi_asset_discovery():
    """Run discovery pipeline on multiple assets."""

    all_results = {}

    for asset in ASSETS:
        print(f"\n{'='*60}")
        print(f"Processing {asset}")
        print('='*60)

        try:
            # Load data
            loader = DataLoader()
            data = loader.load(asset)
            train, val, test = loader.temporal_split(data)

            # Run competition
            population = NichePopulation(DEFAULT_STRATEGIES)
            population.run(train, start_idx=200)
            si = population.compute_si_timeseries(train, window=168)

            # Compute features
            calc = FeatureCalculator()
            features = calc.compute_all(train)

            # Run discovery
            analyzer = CorrelationAnalyzer()
            results = analyzer.run_discovery(si, features, calc.get_discovery_features())

            all_results[asset] = {
                'n_significant': len(results[results['significant']]),
                'top_feature': results.iloc[0]['feature'],
                'top_r': float(results.iloc[0]['r']),
            }

        except Exception as e:
            all_results[asset] = {'error': str(e)}

    # Cross-asset summary
    print("\n" + "="*60)
    print("CROSS-ASSET SUMMARY")
    print("="*60)

    for asset, res in all_results.items():
        if 'error' in res:
            print(f"{asset}: ERROR - {res['error']}")
        else:
            print(f"{asset}: {res['n_significant']} significant, top={res['top_feature']} (r={res['top_r']:.3f})")

    return all_results
```

**Priority**: 🟢 LOW - Can add after initial BTC run

---

### Gap 8: No Visualization Pipeline (Prof. D - Experimental)

**Issue**: No figures planned. Hard to interpret results.

**Risk**: Miss patterns, harder to communicate findings.

**Recommendation**:
```python
# Add to Phase 8: Visualization

import matplotlib.pyplot as plt
import seaborn as sns

def generate_figures(results_dir: Path):
    """Generate all figures for report."""

    # Load results
    discovery = pd.read_csv(results_dir / "discovery_results.csv")

    # Figure 1: Top Correlations Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10 = discovery.head(10)
    colors = ['green' if r > 0 else 'red' for r in top_10['r']]
    ax.barh(top_10['feature'], top_10['r'], color=colors)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Correlation with SI')
    ax.set_title('Top 10 SI Correlations')
    plt.tight_layout()
    plt.savefig(results_dir / 'fig1_top_correlations.png', dpi=150)
    plt.close()

    # Figure 2: SI Time Series
    si = pd.read_csv(results_dir / "si_train.csv", index_col=0, parse_dates=True).squeeze()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(si.index, si.values, linewidth=0.5)
    ax.set_ylabel('Specialization Index')
    ax.set_title('SI Over Time')
    plt.tight_layout()
    plt.savefig(results_dir / 'fig2_si_timeseries.png', dpi=150)
    plt.close()

    # Figure 3: Correlation Heatmap
    features = pd.read_csv(results_dir / "features_train.csv", index_col=0, parse_dates=True)
    top_features = discovery.head(10)['feature'].tolist()
    corr_matrix = features[top_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(results_dir / 'fig3_feature_correlations.png', dpi=150)
    plt.close()

    print(f"Figures saved to {results_dir}")
```

**Priority**: 🟢 LOW - Add after main analysis

---

## 📋 Summary: What to Add Before Execution

| Gap | Priority | Effort | Add to Phase |
|-----|----------|--------|--------------|
| 1. Data Validation | 🔴 HIGH | 30 min | Phase 2 |
| 2. Power Analysis | 🟡 MEDIUM | 15 min | Phase 2 |
| 3. SI Baseline | 🟡 MEDIUM | 15 min | Phase 3 |
| 4. Regime Validation | 🟡 MEDIUM | 20 min | Phase 3 |
| 5. Test Set Lockout | 🟡 MEDIUM | 15 min | Phase 2 |
| 6. Logging | 🟢 LOW | 20 min | All phases |
| 7. Multi-Asset | 🟢 LOW | 30 min | Phase 4 |
| 8. Visualization | 🟢 LOW | 30 min | Phase 8 |

**Total additional effort**: ~3 hours

---

## 🎓 Expert Panel Verdict

### Prof. A (Statistics):
> "Plan is statistically sound. Add power analysis to confirm sample size is adequate. Otherwise, approved."

### Prof. B (ML/AI):
> "Implementation looks correct. Multi-asset loop should be added but can be done after initial run. Approved."

### Prof. C (Finance):
> "Regime classification needs validation. Otherwise realistic for research purposes. Approved with caveat."

### Prof. D (Experimental Design):
> "Methodology is rigorous. Add visualization pipeline for interpretation. Approved."

### Dr. E (Quant Research):
> "Missing SI baseline comparison. Critical for interpreting results. Add before execution."

### Dr. F (Data Science):
> "**Data validation is missing. Must add before any analysis.** Cannot approve without this."

### Dr. G (Software Engineering):
> "Add logging and checkpointing for robustness. Approved with recommendation."

### Dr. H (Research Ethics):
> "Pre-registration in place. Add test lockout mechanism for extra integrity. Approved."

---

## ✅ Final Checklist Before Execution

### Must Do (🔴 HIGH):
- [ ] Add data validation step to MASTER_PLAN
- [ ] Commit pre-registration to GitHub

### Should Do (🟡 MEDIUM):
- [ ] Add power analysis
- [ ] Add SI baseline computation
- [ ] Add regime validation
- [ ] Add test set lockout

### Nice to Have (🟢 LOW):
- [ ] Add logging/checkpointing
- [ ] Add multi-asset loop
- [ ] Add visualization pipeline

---

## 🚀 Recommendation

> **Add the HIGH priority gap (Data Validation) to MASTER_PLAN, then proceed with execution.**
>
> The MEDIUM and LOW items can be added incrementally during execution without blocking progress.

---

*Panel review complete. Ready for execution after data validation is added.*

*Last Updated: January 17, 2026*
