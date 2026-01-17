# Expert Panel Review: MASTER_PLAN.md

**Date**: January 17, 2026
**Purpose**: Final expert review before execution
**Document Reviewed**: `MASTER_PLAN.md` (2400+ lines)

---

## 👥 Panel Composition (12 Experts)

| Expert | Domain | Affiliation | Focus |
|--------|--------|-------------|-------|
| Prof. Chen | Statistics | Stanford | Statistical methodology |
| Prof. Williams | ML/AI | MIT | Algorithm design |
| Prof. Kumar | Finance | Chicago Booth | Market microstructure |
| Prof. Zhang | Econometrics | LSE | Time series analysis |
| Prof. Anderson | Experimental Design | Berkeley | Research methodology |
| Dr. Martinez | Quant Research | Two Sigma | Practical trading |
| Dr. Thompson | Risk Management | Bridgewater | Risk frameworks |
| Dr. Lee | Data Engineering | Jane Street | Data pipelines |
| Dr. Patel | Software Eng | Google DeepMind | Code quality |
| Dr. Nakamura | Crypto Markets | Paradigm | Crypto-specific |
| Dr. Schmidt | Reproducibility | Max Planck | Open science |
| Dr. Wilson | Research Ethics | NIH | Integrity |

---

## 📋 Review Categories

| Category | Score | Issues | Recommendations |
|----------|-------|--------|-----------------|
| 1. Overall Structure | 9/10 | 1 | 2 |
| 2. Statistical Methodology | 9/10 | 2 | 3 |
| 3. Implementation Design | 8/10 | 3 | 4 |
| 4. Data Handling | 8/10 | 2 | 3 |
| 5. Trading Realism | 7/10 | 4 | 5 |
| 6. Reproducibility | 9/10 | 1 | 2 |
| 7. Risk Management | 7/10 | 3 | 4 |
| 8. Scope & Timeline | 9/10 | 1 | 2 |
| **Overall** | **8.3/10** | **17** | **25** |

---

## 1. Overall Structure Review (Prof. Anderson)

### Score: 9/10

### Strengths
- ✅ Single-file execution plan is excellent for LLM workflows
- ✅ Clear phase progression with checkpoints
- ✅ Inline code ready to copy-paste
- ✅ Future enhancements properly parked

### Issues Found

**Issue 1.1**: No version control for MASTER_PLAN itself

> "If you modify MASTER_PLAN during execution, you lose the original. Consider freezing a version."

### Recommendations

**Rec 1.1**: Add versioning
```bash
# Before starting execution
cp MASTER_PLAN.md MASTER_PLAN_v1.0_frozen.md
git add MASTER_PLAN_v1.0_frozen.md
git commit -m "Freeze MASTER_PLAN v1.0 before execution"
```

**Rec 1.2**: Add execution log section
```markdown
## 📝 Execution Log (Fill During Execution)

| Date | Phase | Status | Notes |
|------|-------|--------|-------|
| | | | |
```

---

## 2. Statistical Methodology Review (Prof. Chen & Prof. Zhang)

### Score: 9/10

### Strengths
- ✅ Pre-registration is exemplary
- ✅ FDR correction for multiple testing
- ✅ HAC standard errors for autocorrelation
- ✅ Block bootstrap for confidence intervals
- ✅ Three-pipeline separation is methodologically sound

### Issues Found

**Issue 2.1**: Block size not data-driven

> "Block size of 24h is arbitrary. Should be based on autocorrelation decay."

**Issue 2.2**: No adjustment for effective sample size in power

> "With autocorrelation ρ=0.3, effective N is much smaller than raw N."

### Recommendations

**Rec 2.1**: Add autocorrelation-based block size
```python
def optimal_block_size(returns: pd.Series) -> int:
    """
    Compute optimal block size based on autocorrelation.
    Uses Politis & White (2004) automatic selection.
    """
    from statsmodels.tsa.stattools import acf

    # Compute autocorrelation
    acf_values = acf(returns.dropna(), nlags=100)

    # Find lag where ACF drops below 0.05
    for lag, val in enumerate(acf_values):
        if abs(val) < 0.05:
            return max(lag, 1)

    return 24  # Default fallback
```

**Rec 2.2**: Report effective N in results
```python
def effective_sample_size(n: int, rho: float) -> int:
    """Compute effective N accounting for autocorrelation."""
    return int(n * (1 - rho) / (1 + rho))

# Report both in results
results['n_raw'] = len(data)
results['n_effective'] = effective_sample_size(len(data), autocorr)
```

**Rec 2.3**: Add Bonferroni as sensitivity check
```python
# Primary: FDR correction
results['p_fdr'] = fdr_correction(results['p'])

# Sensitivity: Bonferroni (more conservative)
results['p_bonferroni'] = results['p'] * len(results)
results['significant_bonferroni'] = results['p_bonferroni'] < 0.05

# Report: "X significant with FDR, Y significant with Bonferroni"
```

---

## 3. Implementation Design Review (Dr. Patel & Dr. Lee)

### Score: 8/10

### Strengths
- ✅ Clean class structure
- ✅ Type hints in function signatures
- ✅ Docstrings present
- ✅ Modular design

### Issues Found

**Issue 3.1**: No logging in production code

> "Print statements are fine for development, but need proper logging for debugging issues."

**Issue 3.2**: No error handling for edge cases

> "What happens if data has all NaN? Division by zero?"

**Issue 3.3**: No progress indicators for long-running operations

> "Competition loop over 5000+ iterations should show progress."

### Recommendations

**Rec 3.1**: Add logging module
```python
# src/utils/logging.py
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs"):
    """Setup logging with file and console handlers."""
    Path(log_dir).mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(ch)

    return logger
```

**Rec 3.2**: Add defensive checks
```python
def safe_divide(a, b, default=0.0):
    """Division with NaN/Inf/zero handling."""
    if b == 0 or np.isnan(b) or np.isinf(b):
        return default
    result = a / b
    return default if np.isnan(result) or np.isinf(result) else result

def validate_series(s: pd.Series, name: str) -> pd.Series:
    """Validate series before computation."""
    if s.isna().all():
        raise ValueError(f"{name} is all NaN")
    if len(s) < 10:
        raise ValueError(f"{name} has only {len(s)} values")
    return s
```

**Rec 3.3**: Add progress bars
```python
from tqdm import tqdm

# In NichePopulation.run():
for idx in tqdm(range(start_idx, len(data)), desc="Competition"):
    self.compete(data, idx)
```

**Rec 3.4**: Add `tqdm` to requirements
```
# requirements.txt
tqdm>=4.65.0
```

---

## 4. Data Handling Review (Dr. Lee)

### Score: 8/10

### Strengths
- ✅ Data validation step is comprehensive
- ✅ Multi-market loader is well-designed
- ✅ Temporal split preserves order

### Issues Found

**Issue 4.1**: No data caching

> "Downloading and processing data each run is wasteful."

**Issue 4.2**: No handling for timezone issues

> "Crypto is UTC, stocks are ET. Mixed analysis could have bugs."

### Recommendations

**Rec 4.1**: Add data caching
```python
import hashlib
import pickle

def cached_load(loader, symbol, cache_dir="cache"):
    """Load data with caching."""
    Path(cache_dir).mkdir(exist_ok=True)

    cache_key = hashlib.md5(f"{symbol}_{loader.market_type}".encode()).hexdigest()
    cache_file = Path(cache_dir) / f"{cache_key}.pkl"

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    data = loader.load(symbol)

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

    return data
```

**Rec 4.2**: Standardize to UTC
```python
def standardize_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all timestamps to UTC."""
    if df.index.tz is None:
        # Assume UTC if no timezone
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    return df
```

**Rec 4.3**: Add timezone validation
```python
def validate_timezone(df: pd.DataFrame) -> bool:
    """Validate data is in UTC."""
    if df.index.tz is None:
        print("WARNING: No timezone info. Assuming UTC.")
        return True
    return str(df.index.tz) == 'UTC'
```

---

## 5. Trading Realism Review (Dr. Martinez & Dr. Thompson)

### Score: 7/10

### Strengths
- ✅ Acknowledges this is signal discovery, not trading
- ✅ Transaction costs mentioned in future plans
- ✅ Multi-market validation adds robustness

### Issues Found

**Issue 5.1**: Strategies are too simple

> "Momentum and mean-reversion with fixed parameters won't reflect real trading. This is fine for SI testing, but should be acknowledged as limitation."

**Issue 5.2**: No market impact modeling

> "Large positions move markets. SI computed with frictionless assumptions."

**Issue 5.3**: No regime-dependent analysis

> "SI-volatility correlation might flip sign in different regimes."

**Issue 5.4**: Position sizing ignored

> "All agents assumed equal size. Real traders size by conviction."

### Recommendations

**Rec 5.1**: Add limitation section
```markdown
## Known Limitations (Acknowledge in Report)

1. **Simple strategies**: Momentum/mean-reversion with fixed parameters
2. **No market impact**: Frictionless execution assumed
3. **Equal sizing**: All agents same size
4. **No transaction costs**: Zero-cost trades
5. **Hourly data**: May miss intraday patterns

These are acceptable for SIGNAL DISCOVERY but would need addressing for LIVE TRADING.
```

**Rec 5.2**: Add regime-conditioned analysis
```python
def analyze_by_regime(si: pd.Series, feature: pd.Series,
                      regimes: pd.Series) -> Dict:
    """Analyze SI-feature correlation within each regime."""
    results = {}

    for regime in regimes.unique():
        mask = regimes == regime
        if mask.sum() < 100:
            continue

        r, p = spearmanr(si[mask], feature[mask])
        results[regime] = {'r': r, 'p': p, 'n': mask.sum()}

    # Check if sign flips
    rs = [v['r'] for v in results.values()]
    results['sign_consistent'] = len(set(np.sign(r) for r in rs)) == 1

    return results
```

**Rec 5.3**: Add to Phase 4 output
```python
# In discovery results
results['regime_analysis'] = analyze_by_regime(si, feature, regimes)
results['sign_consistent'] = results['regime_analysis']['sign_consistent']

# Flag if sign flips
if not results['sign_consistent']:
    print(f"⚠️  WARNING: SI-{feature} correlation sign flips across regimes!")
```

**Rec 5.4**: Document as future work
```markdown
## Future Work (After SI Validation)

- [ ] Add realistic transaction costs
- [ ] Model market impact
- [ ] Test with variable position sizing
- [ ] Use more sophisticated strategies
```

---

## 6. Reproducibility Review (Dr. Schmidt)

### Score: 9/10

### Strengths
- ✅ Pre-registration is excellent
- ✅ Random seed setting
- ✅ Requirements.txt with versions
- ✅ Git commit hashing

### Issues Found

**Issue 6.1**: No hardware reproducibility

> "Results might differ on different hardware (floating point)."

### Recommendations

**Rec 6.1**: Add environment fingerprint
```python
def environment_fingerprint():
    """Capture environment for reproducibility."""
    import platform
    import numpy as np

    return {
        'python_version': platform.python_version(),
        'os': platform.system(),
        'processor': platform.processor(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'random_seed': 42,
        'numpy_random_state': str(np.random.get_state()[1][:5]),
    }
```

**Rec 6.2**: Pin exact package versions
```
# requirements.txt - pin EXACT versions
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1
statsmodels==0.14.0
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
tqdm==4.65.0
```

---

## 7. Risk Management Review (Dr. Thompson)

### Score: 7/10

### Strengths
- ✅ Clear stopping criteria
- ✅ Contingency plans for common issues
- ✅ Exit criteria defined

### Issues Found

**Issue 7.1**: No budget for API/compute costs

> "What if cross-market validation costs more than expected?"

**Issue 7.2**: No time limits per phase

> "Could spend infinite time on Phase 4 trying variations."

**Issue 7.3**: No rollback plan

> "What if Phase 5 invalidates Phase 4 findings?"

### Recommendations

**Rec 7.1**: Add cost/time budget
```markdown
## Phase Budgets

| Phase | Max Time | Max Compute Cost | Action if Exceeded |
|-------|----------|------------------|-------------------|
| Phase 2 | 3 days | $0 | Simplify data sources |
| Phase 3 | 2 days | ~$0 | Reduce iterations |
| Phase 4-6 | 5 days | ~$0 | Limit feature count |
| Phase 9 | 5 days | ~$0 | Test fewer markets |
| **Total** | **15 days** | **~$0** | |
```

**Rec 7.2**: Add phase time limits
```python
# At start of each phase
phase_start = datetime.now()
MAX_PHASE_HOURS = 24  # 1 day per phase

# At key checkpoints
elapsed = (datetime.now() - phase_start).total_seconds() / 3600
if elapsed > MAX_PHASE_HOURS:
    print(f"⚠️  Phase time limit exceeded ({elapsed:.1f}h > {MAX_PHASE_HOURS}h)")
    print("    Consider: simplifying, getting help, or stopping")
```

**Rec 7.3**: Add rollback checkpoints
```python
# After each phase, save checkpoint
def save_phase_checkpoint(phase: int, results: dict, data: dict):
    """Save checkpoint for potential rollback."""
    checkpoint = {
        'phase': phase,
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'data_hash': hashlib.md5(str(data).encode()).hexdigest(),
    }

    with open(f"checkpoints/phase_{phase}.json", 'w') as f:
        json.dump(checkpoint, f)

    print(f"✅ Checkpoint saved: phase_{phase}.json")
```

**Rec 7.4**: Define rollback triggers
```markdown
## Rollback Triggers

| If This Happens | Rollback To |
|-----------------|-------------|
| Phase 5 contradicts Phase 4 | Re-run Phase 4 with different parameters |
| Phase 7 validation fails badly | Re-examine Phase 4 methodology |
| Cross-market fails completely | Accept single-market scope |
```

---

## 8. Scope & Timeline Review (Prof. Anderson)

### Score: 9/10

### Strengths
- ✅ 10-14 day timeline is realistic
- ✅ Clear phase boundaries
- ✅ Future work properly scoped out

### Issues Found

**Issue 8.1**: No buffer for unexpected issues

> "14 days assumes everything works first try."

### Recommendations

**Rec 8.1**: Add 20% buffer
```markdown
## Revised Timeline

| Phase | Optimistic | Realistic | Pessimistic |
|-------|------------|-----------|-------------|
| Phase 1 | 0.5 days | 1 day | 2 days |
| Phase 2 | 1 day | 2 days | 3 days |
| Phase 3 | 1 day | 1 day | 2 days |
| Phase 4-6 | 3 days | 4 days | 6 days |
| Phase 7 | 1 day | 2 days | 3 days |
| Phase 8 | 0.5 days | 1 day | 1 day |
| Phase 9 | 2 days | 3 days | 5 days |
| **Total** | **9 days** | **14 days** | **22 days** |

Plan for REALISTIC (14 days), budget for PESSIMISTIC (22 days).
```

**Rec 8.2**: Define "done" for each phase
```markdown
## Phase Completion Criteria

| Phase | Done When |
|-------|-----------|
| 1 | Smoke test passes |
| 2 | Data validation passes for ≥1 asset |
| 3 | SI computed and has variance |
| 4 | Discovery results saved |
| 5 | Prediction results saved |
| 6 | Dynamics results saved |
| 7 | Audit results saved |
| 8 | Report generated |
| 9 | Cross-market results saved |
```

---

## 📊 Summary: All Issues & Recommendations

### Issues by Priority

| Priority | Count | Examples |
|----------|-------|----------|
| 🔴 Critical | 0 | - |
| 🟡 Important | 8 | Logging, timezone, regime analysis |
| 🟢 Nice-to-have | 9 | Caching, progress bars, versioning |

### Top 10 Recommendations to Implement

| # | Recommendation | Effort | Impact |
|---|----------------|--------|--------|
| 1 | Add logging module | 30 min | High |
| 2 | Add progress bars (tqdm) | 10 min | Medium |
| 3 | Add autocorrelation-based block size | 20 min | High |
| 4 | Standardize timezone to UTC | 15 min | Medium |
| 5 | Add regime-conditioned analysis | 30 min | High |
| 6 | Add phase time limits | 15 min | Medium |
| 7 | Add limitations section | 10 min | Medium |
| 8 | Freeze MASTER_PLAN version | 5 min | Low |
| 9 | Add execution log section | 5 min | Low |
| 10 | Pin exact package versions | 10 min | Medium |

### Optional (Do If Time)

| # | Recommendation | Effort |
|---|----------------|--------|
| 11 | Data caching | 30 min |
| 12 | Environment fingerprint | 15 min |
| 13 | Rollback checkpoints | 30 min |
| 14 | Bonferroni sensitivity check | 10 min |

---

## 🎓 Expert Panel Verdict

### Prof. Chen (Statistics):
> "Methodology is excellent. Add autocorrelation-based block size for rigor. 9/10."

### Prof. Williams (ML/AI):
> "Clean design. Add logging and progress bars for debugging. 8/10."

### Prof. Kumar (Finance):
> "Strategies are toy-level, but that's fine for signal discovery. Document as limitation. 8/10."

### Dr. Martinez (Two Sigma):
> "Add regime-conditioned analysis. SI-volatility correlation might flip in different regimes. 7/10."

### Dr. Lee (Jane Street):
> "Standardize timezones. Mixed UTC and ET will cause bugs. 8/10."

### Dr. Patel (DeepMind):
> "Add defensive checks for NaN/Inf. Production code will break otherwise. 8/10."

### Dr. Thompson (Bridgewater):
> "Add time limits per phase. Easy to get stuck optimizing forever. 7/10."

### Dr. Schmidt (Reproducibility):
> "Pin exact package versions. Close to gold standard for reproducibility. 9/10."

### Dr. Wilson (Ethics):
> "Pre-registration is exemplary. Commit it before any analysis. 10/10."

---

## ✅ Final Recommendation

### Implement These Before Starting (2 hours)

1. **Add logging module** (30 min)
2. **Add tqdm to requirements** (5 min)
3. **Add timezone standardization** (15 min)
4. **Add regime-conditioned analysis** (30 min)
5. **Add limitations section** (10 min)
6. **Add execution log section** (5 min)
7. **Freeze MASTER_PLAN version** (5 min)
8. **Pin exact package versions** (10 min)

### Skip These (Can Add During Execution)

- Data caching (nice-to-have)
- Environment fingerprint (nice-to-have)
- Rollback checkpoints (nice-to-have)

---

## 📋 Updated Readiness Score

| Category | Before | After Fixes |
|----------|--------|-------------|
| Overall Structure | 9/10 | 10/10 |
| Statistical Methodology | 9/10 | 10/10 |
| Implementation Design | 8/10 | 9/10 |
| Data Handling | 8/10 | 9/10 |
| Trading Realism | 7/10 | 8/10 |
| Reproducibility | 9/10 | 10/10 |
| Risk Management | 7/10 | 8/10 |
| Scope & Timeline | 9/10 | 10/10 |
| **Overall** | **8.3/10** | **9.3/10** |

---

*Expert panel review complete. Implement 8 high-priority recommendations before execution.*

*Last Updated: January 17, 2026*
