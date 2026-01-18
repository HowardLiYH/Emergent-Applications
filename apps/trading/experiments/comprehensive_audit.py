#!/usr/bin/env python3
"""
COMPREHENSIVE AUDIT CHECK

Systematically audit all aspects of our analysis for issues:
1. Data integrity
2. Look-ahead bias
3. SI computation correctness
4. Return calculation methodology
5. Transaction cost handling
6. Statistical significance
7. Overfitting detection
8. Cross-validation issues
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2
from src.data.loader_v2 import DataLoaderV2, MarketType

# ============================================================
# AUDIT FRAMEWORK
# ============================================================

class AuditResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.issues = []
        self.warnings = []
        self.fixes_applied = []
    
    def fail(self, issue: str):
        self.passed = False
        self.issues.append(issue)
    
    def warn(self, warning: str):
        self.warnings.append(warning)
    
    def fix(self, fix_description: str):
        self.fixes_applied.append(fix_description)
    
    def to_dict(self):
        return {
            'name': self.name,
            'passed': self.passed,
            'issues': self.issues,
            'warnings': self.warnings,
            'fixes_applied': self.fixes_applied,
        }

# ============================================================
# AUDIT 1: DATA INTEGRITY
# ============================================================

def audit_data_integrity(data: pd.DataFrame, asset: str) -> AuditResult:
    """Check data quality and integrity."""
    result = AuditResult(f"Data Integrity: {asset}")
    
    # Check 1: Missing values
    missing_pct = data.isnull().sum() / len(data)
    if missing_pct.max() > 0.01:
        result.fail(f"Missing values: {missing_pct.max():.2%} in {missing_pct.idxmax()}")
    elif missing_pct.max() > 0:
        result.warn(f"Minor missing values: {missing_pct.max():.4%}")
    
    # Check 2: Duplicate timestamps
    duplicates = data.index.duplicated().sum()
    if duplicates > 0:
        result.fail(f"Duplicate timestamps: {duplicates}")
    
    # Check 3: Non-monotonic index
    if not data.index.is_monotonic_increasing:
        result.fail("Index is not monotonically increasing")
    
    # Check 4: Extreme returns
    returns = data['close'].pct_change().dropna()
    extreme_count = (returns.abs() > 0.5).sum()
    if extreme_count > 0:
        result.warn(f"Extreme returns (|r| > 50%): {extreme_count} occurrences")
    
    # Check 5: Zero/negative prices
    if (data['close'] <= 0).any():
        result.fail("Zero or negative prices found")
    
    # Check 6: Gaps > 5 days
    gaps = data.index.to_series().diff()
    large_gaps = gaps[gaps > pd.Timedelta(days=5)]
    if len(large_gaps) > 0:
        result.warn(f"Large gaps (>5 days): {len(large_gaps)}")
    
    # Check 7: Stale data
    staleness = (data['close'] == data['close'].shift()).sum()
    if staleness / len(data) > 0.1:
        result.warn(f"Stale data (unchanged prices): {staleness} rows ({staleness/len(data):.1%})")
    
    return result

# ============================================================
# AUDIT 2: LOOK-AHEAD BIAS
# ============================================================

def audit_lookahead_bias() -> AuditResult:
    """Check for look-ahead bias in our code."""
    result = AuditResult("Look-Ahead Bias Check")
    
    # Read the test_all_applications.py file
    code_path = Path("experiments/test_all_applications.py")
    if not code_path.exists():
        result.fail("Cannot find test_all_applications.py")
        return result
    
    with open(code_path, 'r') as f:
        code = f.read()
    
    # Check 1: .shift(-N) without explicit forward reference
    if '.shift(-' in code:
        # Count occurrences
        import re
        forward_shifts = re.findall(r'\.shift\(-\d+\)', code)
        for shift in forward_shifts:
            # Check if it's used for forward returns (acceptable) or calculation (not acceptable)
            context = code[max(0, code.find(shift)-100):code.find(shift)+50]
            if 'fwd_' in context or 'future_' in context or 'forward_' in context:
                result.warn(f"Forward shift found but appears intentional for forward returns: {shift}")
            else:
                result.fail(f"Potential look-ahead bias: {shift} without clear future variable naming")
    
    # Check 2: Position uses current returns
    if 'position * returns' in code and 'shift(1)' not in code:
        result.warn("Position may not be shifted before applying to returns")
    
    # Check 3: SI computed on full data
    if 'compute_si(data' in code and 'train' not in code.split('compute_si')[0][-50:]:
        result.warn("SI may be computed on full data instead of train only")
    
    # Check 4: Rolling windows using future data
    # Pattern: .rolling(X).mean() without shift
    if '.rolling(' in code:
        # This is just a warning - rolling itself doesn't use future data
        result.warn("Rolling windows used - verify they're applied correctly")
    
    # Check position.shift(1) usage
    shift_count = code.count('.shift(1)')
    position_count = code.count('position')
    if shift_count < position_count / 3:
        result.warn(f"Low shift(1) count ({shift_count}) relative to position usage ({position_count})")
    
    return result

# ============================================================
# AUDIT 3: SI COMPUTATION
# ============================================================

def audit_si_computation(data: pd.DataFrame, asset: str) -> AuditResult:
    """Verify SI computation is correct."""
    result = AuditResult(f"SI Computation: {asset}")
    
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    
    # Run competition
    population.run(data)
    si = population.compute_si_timeseries(data, window=7)
    
    # Check 1: SI range
    if si.min() < 0 or si.max() > 1:
        result.fail(f"SI out of [0,1] range: min={si.min():.3f}, max={si.max():.3f}")
    
    # Check 2: SI variance
    if si.std() < 0.01:
        result.warn(f"SI has very low variance: std={si.std():.4f}")
    
    # Check 3: SI NaN count
    nan_pct = si.isnull().sum() / len(si)
    if nan_pct > 0.1:
        result.fail(f"Too many NaN in SI: {nan_pct:.1%}")
    elif nan_pct > 0.01:
        result.warn(f"Some NaN in SI: {nan_pct:.2%}")
    
    # Check 4: SI autocorrelation (should be high for a meaningful signal)
    autocorr = si.autocorr(lag=1)
    if autocorr < 0.5:
        result.warn(f"Low SI autocorrelation: {autocorr:.2f} (may be too noisy)")
    
    # Check 5: SI starts from correct point (after window warmup)
    first_valid = si.first_valid_index()
    expected_warmup = 7  # window size
    actual_warmup = data.index.get_loc(first_valid)
    if actual_warmup < expected_warmup - 1:
        result.warn(f"SI starts too early: index {actual_warmup} vs expected {expected_warmup}")
    
    return result

# ============================================================
# AUDIT 4: RETURN CALCULATION
# ============================================================

def audit_return_calculation() -> AuditResult:
    """Check return calculation methodology."""
    result = AuditResult("Return Calculation")
    
    code_path = Path("experiments/test_all_applications.py")
    with open(code_path, 'r') as f:
        code = f.read()
    
    # Check 1: Using pct_change for returns
    if "pct_change()" not in code:
        result.fail("Returns should use pct_change()")
    
    # Check 2: Returns aligned with positions
    if "position.shift(1)" in code or "position_shifted" in code:
        pass  # Good
    else:
        result.warn("Positions may not be shifted before multiplying with returns")
    
    # Check 3: Log returns vs simple returns
    if "np.log(" in code and "pct_change" in code:
        result.warn("Mixed log and simple returns - ensure consistency")
    
    # Check 4: Annualization factor
    if "252" in code:
        pass  # Correct for daily
    elif "365" in code:
        result.warn("Using 365 for annualization - should be 252 for trading days")
    
    return result

# ============================================================
# AUDIT 5: TRANSACTION COSTS
# ============================================================

def audit_transaction_costs() -> AuditResult:
    """Verify transaction costs are realistic and applied correctly."""
    result = AuditResult("Transaction Costs")
    
    code_path = Path("experiments/test_all_applications.py")
    with open(code_path, 'r') as f:
        code = f.read()
    
    # Check 1: Transaction costs defined
    if "TRANSACTION_COSTS" not in code and "cost_rate" not in code:
        result.fail("Transaction costs not defined")
    
    # Check 2: Costs are realistic
    # Crypto: 4-10 bps, Forex: 1-5 bps, Stocks: 2-10 bps
    if "0.0004" in code:  # 4 bps
        pass  # Reasonable for crypto
    if "0.01" in code:  # 100 bps = 1%
        result.fail("Transaction cost of 1% is too high for most markets")
    
    # Check 3: Costs applied to position changes, not all periods
    if "position_changes" in code or "diff().abs()" in code:
        pass  # Correct
    else:
        result.warn("Transaction costs may not be applied to position changes only")
    
    # Check 4: Round-trip costs
    if "* 2" in code or "round-trip" in code.lower():
        pass  # Considering round-trip
    else:
        result.warn("Ensure costs include both entry and exit (round-trip)")
    
    return result

# ============================================================
# AUDIT 6: STATISTICAL SIGNIFICANCE
# ============================================================

def audit_statistical_significance(results_path: str) -> AuditResult:
    """Check for proper statistical testing."""
    result = AuditResult("Statistical Significance")
    
    path = Path(results_path)
    if not path.exists():
        result.fail(f"Results file not found: {results_path}")
        return result
    
    with open(path, 'r') as f:
        results = json.load(f)
    
    # Check 1: Number of tests vs significance level
    n_applications = len(results.get('detailed_results', {}))
    n_assets = 6
    total_tests = n_applications * n_assets
    
    if total_tests > 20:
        result.warn(f"Multiple testing: {total_tests} tests - need FDR correction")
    
    # Check 2: Sample size per test
    # We should have at least 100 observations per test
    # This is implicit - just warn
    result.warn("Verify each test has sufficient sample size (>100 observations)")
    
    # Check 3: Look for confidence intervals
    detailed = results.get('detailed_results', {})
    has_ci = any('ci_' in str(v) for v in detailed.values())
    if not has_ci:
        result.warn("No confidence intervals computed - consider bootstrapping")
    
    # Check 4: P-values or t-stats
    has_pval = any('pval' in str(v).lower() or 'p_value' in str(v).lower() 
                   for v in detailed.values())
    if not has_pval:
        result.warn("No p-values computed - statistical significance not assessed")
    
    return result

# ============================================================
# AUDIT 7: OVERFITTING DETECTION
# ============================================================

def audit_overfitting(results_path: str) -> AuditResult:
    """Check for signs of overfitting."""
    result = AuditResult("Overfitting Detection")
    
    path = Path(results_path)
    if not path.exists():
        result.fail(f"Results file not found: {results_path}")
        return result
    
    with open(path, 'r') as f:
        results = json.load(f)
    
    detailed = results.get('detailed_results', {})
    
    # Check 1: In-sample vs out-of-sample comparison
    has_oos = any('oos' in str(v).lower() or 'out_of_sample' in str(v).lower() 
                  for v in detailed.values())
    if not has_oos:
        result.warn("No out-of-sample testing detected")
    
    # Check 2: Walk-forward validation
    has_walkforward = any('walk' in str(v).lower() for v in detailed.values())
    if not has_walkforward:
        result.warn("No walk-forward validation detected")
    
    # Check 3: Too many parameters
    # Check for parameter counts in applications
    code_path = Path("experiments/test_all_applications.py")
    with open(code_path, 'r') as f:
        code = f.read()
    
    # Count unique thresholds/parameters
    import re
    thresholds = re.findall(r'threshold\s*=\s*[\d.]+|lookback\s*=\s*\d+', code)
    if len(thresholds) > 20:
        result.warn(f"Many parameters ({len(thresholds)}) - risk of overfitting")
    
    # Check 4: Sharpe ratios too good to be true
    summary = results.get('summary', [])
    for app in summary:
        sharpe = app.get('avg_sharpe', 0)
        if sharpe > 2.0:
            result.fail(f"Suspiciously high Sharpe ({sharpe:.2f}) for {app['application']}")
    
    return result

# ============================================================
# AUDIT 8: CROSS-MARKET CONSISTENCY
# ============================================================

def audit_cross_market_consistency(results_path: str) -> AuditResult:
    """Check if results are consistent across markets."""
    result = AuditResult("Cross-Market Consistency")
    
    path = Path(results_path)
    if not path.exists():
        result.fail(f"Results file not found: {results_path}")
        return result
    
    with open(path, 'r') as f:
        results = json.load(f)
    
    detailed = results.get('detailed_results', {})
    
    # Group by market
    crypto = ['BTCUSDT', 'ETHUSDT']
    stocks = ['SPY', 'QQQ']
    forex = ['EURUSD', 'GBPUSD']
    
    for app_name, app_results in detailed.items():
        if isinstance(app_results, dict):
            sharpes = {}
            for asset, res in app_results.items():
                if isinstance(res, dict) and 'strategy_sharpe' in res:
                    sharpes[asset] = res['strategy_sharpe']
            
            if len(sharpes) >= 4:
                crypto_sharpes = [sharpes.get(a, 0) for a in crypto if a in sharpes]
                stock_sharpes = [sharpes.get(a, 0) for a in stocks if a in sharpes]
                forex_sharpes = [sharpes.get(a, 0) for a in forex if a in sharpes]
                
                # Check sign consistency within market
                if len(crypto_sharpes) >= 2:
                    if np.sign(crypto_sharpes[0]) != np.sign(crypto_sharpes[1]):
                        result.warn(f"{app_name}: Inconsistent crypto signs")
                
                if len(stock_sharpes) >= 2:
                    if np.sign(stock_sharpes[0]) != np.sign(stock_sharpes[1]):
                        result.warn(f"{app_name}: Inconsistent stock signs")
                
                # Check if all positive or all negative
                all_sharpes = list(sharpes.values())
                if all(s > 0 for s in all_sharpes):
                    pass  # Good
                elif all(s < 0 for s in all_sharpes):
                    result.warn(f"{app_name}: All negative - strategy may not work")
                else:
                    # Mixed - acceptable but note it
                    pass
    
    return result

# ============================================================
# AUDIT 9: APPLICATION-SPECIFIC CHECKS
# ============================================================

def audit_applications() -> AuditResult:
    """Audit specific application implementations."""
    result = AuditResult("Application-Specific Checks")
    
    code_path = Path("experiments/test_all_applications.py")
    with open(code_path, 'r') as f:
        code = f.read()
    
    # Check 1: Risk Budgeting - position scaling correct
    if "0.5 + si_rank * 1.0" in code:
        # Should result in [0.5, 1.5] range
        pass
    else:
        result.warn("Risk Budgeting: Verify position scaling range")
    
    # Check 2: Spread Trading - z-score computed correctly
    if "spread - spread_mean) / (spread_std" in code:
        pass
    else:
        result.warn("Spread Trading: Z-score calculation may be incorrect")
    
    # Check 3: Factor Timing - momentum signal correct
    if "np.sign(data['close'].pct_change(5))" in code:
        pass
    else:
        result.warn("Factor Timing: Momentum signal may be incorrect")
    
    # Check 4: Dynamic Stop - ATR calculation
    if "compute_atr" in code:
        pass
    else:
        result.warn("Dynamic Stop: ATR not computed")
    
    # Check 5: Tail Hedge - hedge returns inverted correctly
    if "-returns" in code or "* -1" in code:
        pass
    else:
        result.warn("Tail Hedge: Hedge returns may not be inverted")
    
    # Check 6: Entry Timing - forward returns calculation
    if ".shift(-5)" in code or "fwd_5d" in code:
        pass
    else:
        result.warn("Entry Timing: Forward returns may not be calculated")
    
    return result

# ============================================================
# AUDIT 10: CODE QUALITY
# ============================================================

def audit_code_quality() -> AuditResult:
    """Check code quality and potential bugs."""
    result = AuditResult("Code Quality")
    
    code_path = Path("experiments/test_all_applications.py")
    with open(code_path, 'r') as f:
        code = f.read()
    
    # Check 1: Division by zero protection
    if "+ 1e-10" in code or "+ 1e-8" in code or "clip(lower=" in code:
        pass  # Protected
    else:
        result.warn("Potential division by zero not protected")
    
    # Check 2: Empty series handling
    if "len(returns) < 10" in code or "dropna()" in code:
        pass
    else:
        result.warn("Empty series may not be handled")
    
    # Check 3: Index alignment
    if ".loc[common]" in code or ".intersection(" in code:
        pass
    else:
        result.warn("Index alignment may be missing")
    
    # Check 4: Error handling
    if "try:" in code and "except" in code:
        pass
    else:
        result.warn("Limited error handling in code")
    
    return result

# ============================================================
# FIX IDENTIFIED ISSUES
# ============================================================

def apply_fixes(audit_results: List[AuditResult]) -> List[str]:
    """Apply fixes for identified issues."""
    fixes = []
    
    for audit in audit_results:
        for issue in audit.issues:
            # Look-ahead bias fixes
            if "look-ahead" in issue.lower():
                fixes.append("FIX: Add .shift(1) before applying positions to returns")
            
            # Transaction cost fixes
            if "cost" in issue.lower() and "too high" in issue.lower():
                fixes.append("FIX: Reduce transaction costs to realistic levels")
            
            # Statistical significance fixes
            if "multiple testing" in issue.lower():
                fixes.append("FIX: Apply Benjamini-Hochberg FDR correction")
            
            # Overfitting fixes
            if "sharpe" in issue.lower() and "suspicious" in issue.lower():
                fixes.append("FIX: Add walk-forward validation to verify results")
    
    return fixes

# ============================================================
# MAIN AUDIT RUNNER
# ============================================================

def main():
    print("\n" + "="*70)
    print("  COMPREHENSIVE AUDIT CHECK")
    print("="*70)
    print(f"  Time: {datetime.now().isoformat()}")
    
    loader = DataLoaderV2()
    audit_results = []
    
    # Define assets
    assets = {
        'BTCUSDT': MarketType.CRYPTO,
        'ETHUSDT': MarketType.CRYPTO,
        'SPY': MarketType.STOCKS,
        'QQQ': MarketType.STOCKS,
        'EURUSD': MarketType.FOREX,
        'GBPUSD': MarketType.FOREX,
    }
    
    # ============================================================
    # RUN ALL AUDITS
    # ============================================================
    
    print("\n" + "-"*70)
    print("  AUDIT 1: Data Integrity")
    print("-"*70)
    
    for asset, market_type in assets.items():
        data = loader.load(asset, market_type)
        result = audit_data_integrity(data, asset)
        audit_results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {asset}: {status}")
        for issue in result.issues:
            print(f"    ❌ {issue}")
        for warning in result.warnings[:2]:
            print(f"    ⚠️ {warning}")
    
    print("\n" + "-"*70)
    print("  AUDIT 2: Look-Ahead Bias")
    print("-"*70)
    
    result = audit_lookahead_bias()
    audit_results.append(result)
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"  Status: {status}")
    for issue in result.issues:
        print(f"    ❌ {issue}")
    for warning in result.warnings[:3]:
        print(f"    ⚠️ {warning}")
    
    print("\n" + "-"*70)
    print("  AUDIT 3: SI Computation")
    print("-"*70)
    
    # Sample one asset for SI audit
    sample_data = loader.load('BTCUSDT', MarketType.CRYPTO)
    result = audit_si_computation(sample_data, 'BTCUSDT')
    audit_results.append(result)
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"  BTCUSDT: {status}")
    for issue in result.issues:
        print(f"    ❌ {issue}")
    for warning in result.warnings:
        print(f"    ⚠️ {warning}")
    
    print("\n" + "-"*70)
    print("  AUDIT 4: Return Calculation")
    print("-"*70)
    
    result = audit_return_calculation()
    audit_results.append(result)
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"  Status: {status}")
    for issue in result.issues:
        print(f"    ❌ {issue}")
    for warning in result.warnings:
        print(f"    ⚠️ {warning}")
    
    print("\n" + "-"*70)
    print("  AUDIT 5: Transaction Costs")
    print("-"*70)
    
    result = audit_transaction_costs()
    audit_results.append(result)
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"  Status: {status}")
    for issue in result.issues:
        print(f"    ❌ {issue}")
    for warning in result.warnings:
        print(f"    ⚠️ {warning}")
    
    print("\n" + "-"*70)
    print("  AUDIT 6: Statistical Significance")
    print("-"*70)
    
    results_path = "results/application_testing/full_results.json"
    result = audit_statistical_significance(results_path)
    audit_results.append(result)
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"  Status: {status}")
    for issue in result.issues:
        print(f"    ❌ {issue}")
    for warning in result.warnings:
        print(f"    ⚠️ {warning}")
    
    print("\n" + "-"*70)
    print("  AUDIT 7: Overfitting Detection")
    print("-"*70)
    
    result = audit_overfitting(results_path)
    audit_results.append(result)
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"  Status: {status}")
    for issue in result.issues:
        print(f"    ❌ {issue}")
    for warning in result.warnings:
        print(f"    ⚠️ {warning}")
    
    print("\n" + "-"*70)
    print("  AUDIT 8: Cross-Market Consistency")
    print("-"*70)
    
    result = audit_cross_market_consistency(results_path)
    audit_results.append(result)
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"  Status: {status}")
    for issue in result.issues:
        print(f"    ❌ {issue}")
    for warning in result.warnings:
        print(f"    ⚠️ {warning}")
    
    print("\n" + "-"*70)
    print("  AUDIT 9: Application-Specific Checks")
    print("-"*70)
    
    result = audit_applications()
    audit_results.append(result)
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"  Status: {status}")
    for issue in result.issues:
        print(f"    ❌ {issue}")
    for warning in result.warnings:
        print(f"    ⚠️ {warning}")
    
    print("\n" + "-"*70)
    print("  AUDIT 10: Code Quality")
    print("-"*70)
    
    result = audit_code_quality()
    audit_results.append(result)
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"  Status: {status}")
    for issue in result.issues:
        print(f"    ❌ {issue}")
    for warning in result.warnings:
        print(f"    ⚠️ {warning}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    print("\n" + "="*70)
    print("  AUDIT SUMMARY")
    print("="*70)
    
    total_passed = sum(1 for r in audit_results if r.passed)
    total_failed = len(audit_results) - total_passed
    total_issues = sum(len(r.issues) for r in audit_results)
    total_warnings = sum(len(r.warnings) for r in audit_results)
    
    print(f"\n  Total Audits: {len(audit_results)}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Total Issues: {total_issues}")
    print(f"  Total Warnings: {total_warnings}")
    
    # Apply fixes
    fixes = apply_fixes(audit_results)
    
    if total_issues > 0 or total_warnings > 0:
        print("\n" + "-"*70)
        print("  CRITICAL ISSUES TO FIX")
        print("-"*70)
        for r in audit_results:
            if r.issues:
                print(f"\n  {r.name}:")
                for issue in r.issues:
                    print(f"    ❌ {issue}")
        
        print("\n" + "-"*70)
        print("  RECOMMENDED FIXES")
        print("-"*70)
        for i, fix in enumerate(fixes, 1):
            print(f"  {i}. {fix}")
    
    # Save audit report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_audits': len(audit_results),
        'passed': total_passed,
        'failed': total_failed,
        'total_issues': total_issues,
        'total_warnings': total_warnings,
        'audits': [r.to_dict() for r in audit_results],
        'recommended_fixes': fixes,
    }
    
    out_path = Path("results/comprehensive_audit/audit_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Report saved: {out_path}")
    print("="*70 + "\n")
    
    return audit_results, fixes

if __name__ == "__main__":
    main()
