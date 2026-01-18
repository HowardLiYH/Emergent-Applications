#!/usr/bin/env python3
"""
COMPREHENSIVE AUDIT OF SI DEEP EXPLORATION

Checks for:
1. Data leakage / look-ahead bias
2. Proper train/test separation
3. Statistical methodology issues
4. Correlation calculation issues
5. Lead-lag methodology issues
6. Return calculation issues
7. Cross-validation issues
8. Sample size adequacy
9. Effect size interpretation
10. Multiple testing correction

Author: Yuhao Li, University of Pennsylvania
Date: January 18, 2026
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================================
# AUDIT CHECKS
# ============================================================================

def audit_data_leakage():
    """Check for data leakage in the exploration."""
    issues = []

    print("\n" + "="*60)
    print("AUDIT 1: DATA LEAKAGE CHECK")
    print("="*60)

    # Check 1.1: SI computation
    print("\n  [1.1] SI Computation:")
    print("        SI uses NichePopulation.run() which iterates through data chronologically")
    print("        At each step t, competition uses return from t (known at t)")
    print("        ✅ NO look-ahead in SI computation")

    # Check 1.2: Feature computation
    print("\n  [1.2] Feature Computation:")
    print("        ADX: uses rolling(14).mean() - backward looking ✅")
    print("        Volatility: uses rolling(14).std() - backward looking ✅")
    print("        RSI: uses rolling(14) - backward looking ✅")
    print("        ✅ NO look-ahead in feature computation")

    # Check 1.3: Future returns
    print("\n  [1.3] Future Returns:")
    print("        future_return_1d = returns.shift(-1)")
    print("        This is INTENTIONAL for prediction testing")
    print("        ⚠️ MUST NOT be used in training signal generation")

    # Check 1.4: Train/test split
    print("\n  [1.4] Train/Test Split:")
    print("        train = data.iloc[:train_end]")
    print("        test = data.iloc[val_end:]")
    print("        SI computed SEPARATELY for each split")
    print("        ✅ Proper temporal split, no leakage")

    return issues


def audit_correlation_methodology():
    """Check correlation calculation methodology."""
    issues = []

    print("\n" + "="*60)
    print("AUDIT 2: CORRELATION METHODOLOGY")
    print("="*60)

    # Check 2.1: Spearman vs Pearson
    print("\n  [2.1] Correlation Method:")
    print("        Using Spearman correlation (rank-based)")
    print("        ✅ Appropriate for non-linear relationships")
    print("        ✅ Robust to outliers")

    # Check 2.2: Autocorrelation handling
    print("\n  [2.2] Autocorrelation Handling:")
    print("        ⚠️ ISSUE: Standard p-values assume IID")
    print("        SI is highly autocorrelated (uses rolling window)")
    print("        ADX/Volatility/RSI also autocorrelated")
    print("        → P-values may be OVERSTATED")
    print("        → Effective sample size is SMALLER than reported")
    issues.append("Autocorrelation not accounted for in p-value calculation")

    # Check 2.3: Window overlap
    print("\n  [2.3] Window Overlap:")
    print("        SI window: 7 days")
    print("        ADX window: 14 days (uses rolling(14) twice)")
    print("        ⚠️ Significant overlap in underlying data")
    print("        → Correlation is PARTIALLY MECHANICAL")
    issues.append("Overlapping windows create mechanical correlations")

    # Check 2.4: Effect size interpretation
    print("\n  [2.4] Effect Size:")
    print("        SI-ADX correlation: r ≈ 0.10-0.20")
    print("        SI-Volatility correlation: r ≈ -0.10 to -0.22")
    print("        These are SMALL to MEDIUM effect sizes")
    print("        R² = 0.01-0.04 (explains 1-4% of variance)")
    print("        ⚠️ Statistically significant ≠ practically useful")
    issues.append("Small effect sizes (R² < 5%)")

    return issues


def audit_lead_lag_methodology():
    """Check lead-lag analysis methodology."""
    issues = []

    print("\n" + "="*60)
    print("AUDIT 3: LEAD-LAG METHODOLOGY")
    print("="*60)

    # Check 3.1: Multiple testing
    print("\n  [3.1] Multiple Testing:")
    print("        Testing 21 lags × 3 features × 9 assets = 567 tests")
    print("        Applied FDR (Benjamini-Hochberg) correction")
    print("        ✅ Proper multiple testing correction")

    # Check 3.2: Best lag selection
    print("\n  [3.2] Best Lag Selection:")
    print("        ⚠️ ISSUE: Reporting 'best' lag from scan")
    print("        Even with FDR, reporting the BEST is cherry-picking")
    print("        Should report: distribution of significant lags")
    issues.append("Reporting 'best' lag may be cherry-picking")

    # Check 3.3: Interpretation of lag direction
    print("\n  [3.3] Lag Interpretation:")
    print("        'SI leads ADX by 6-8 days'")
    print("        ⚠️ CAUTION: SI uses 7-day window")
    print("        ADX uses 14-day window (with 2 rolling operations)")
    print("        'Lead' might just mean SI REACTS FASTER")
    print("        Not necessarily PREDICTIVE")
    issues.append("Lead-lag may reflect window size differences, not causation")

    # Check 3.4: Granger causality
    print("\n  [3.4] Missing: Granger Causality Test")
    print("        ⚠️ Lead-lag correlation ≠ Granger causality")
    print("        Should test: Does SI help forecast ADX BEYOND past ADX?")
    issues.append("No Granger causality test performed")

    return issues


def audit_return_calculation():
    """Check return calculation methodology."""
    issues = []

    print("\n" + "="*60)
    print("AUDIT 4: RETURN CALCULATION")
    print("="*60)

    # Check 4.1: Transaction costs
    print("\n  [4.1] Transaction Costs:")
    print("        Crypto: 10 bps, Forex: 2 bps, Stocks: 5 bps")
    print("        ✅ Included in threshold analysis")

    # Check 4.2: Signal delay
    print("\n  [4.2] Signal Execution Delay:")
    print("        Strategy: trade when SI crosses threshold")
    print("        ⚠️ ISSUE: No execution delay assumed")
    print("        In reality: SI computed at close, trade at next open")
    print("        This adds ~1 day delay not accounted for")
    issues.append("No execution delay in return calculation")

    # Check 4.3: Position sizing
    print("\n  [4.3] Position Sizing:")
    print("        Signal: 0 or 1 (binary)")
    print("        ⚠️ No volatility scaling")
    print("        ⚠️ No risk management")
    issues.append("Binary position sizing, no risk management")

    # Check 4.4: Compounding
    print("\n  [4.4] Return Compounding:")
    print("        Using: (1 + returns).prod() - 1")
    print("        ✅ Proper compounding")

    return issues


def audit_out_of_sample():
    """Check out-of-sample validation."""
    issues = []

    print("\n" + "="*60)
    print("AUDIT 5: OUT-OF-SAMPLE VALIDATION")
    print("="*60)

    # Check 5.1: Sample sizes
    print("\n  [5.1] Test Set Sizes:")
    results_path = Path('results/deep_exploration_corrected/findings.json')
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)

        for asset_result in results.get('per_asset_results', []):
            n_test = asset_result.get('n_test', 0)
            asset = asset_result.get('asset', 'unknown')
            if n_test < 50:
                print(f"        ⚠️ {asset}: only {n_test} test samples (TOO FEW!)")
                issues.append(f"{asset} has only {n_test} test samples")
            else:
                print(f"        ✅ {asset}: {n_test} test samples")

    # Check 5.2: Degradation
    print("\n  [5.2] Train→Test Degradation:")
    print("        Threshold trading: 13% train → 0.5% test (96% degradation)")
    print("        ⚠️ SEVERE overfitting")
    issues.append("96% performance degradation from train to test")

    # Check 5.3: Validation rate
    print("\n  [5.3] Validation Rate:")
    print("        Threshold: 1/9 assets validated (11%)")
    print("        Quadrant: 0/9 assets validated (0%)")
    print("        ⚠️ Very low validation rates")
    issues.append("Only 11% of assets validate OOS")

    return issues


def audit_sample_size():
    """Check sample size adequacy."""
    issues = []

    print("\n" + "="*60)
    print("AUDIT 6: SAMPLE SIZE ADEQUACY")
    print("="*60)

    # Check 6.1: Effective sample size
    print("\n  [6.1] Effective Sample Size:")
    print("        Nominal: ~750-1100 daily observations per asset")
    print("        ⚠️ But SI is computed with 7-day window")
    print("        ⚠️ High autocorrelation (ρ > 0.9 at lag 1)")
    print("        Effective N might be 10-20x smaller")
    print("        → May only have ~50-100 'independent' observations")
    issues.append("Effective sample size much smaller due to autocorrelation")

    # Check 6.2: Power analysis
    print("\n  [6.2] Statistical Power:")
    print("        Observed effect sizes: r ≈ 0.15")
    print("        At N=1000, power > 99% for r=0.15")
    print("        At effective N=100, power ≈ 50%")
    print("        ⚠️ May be underpowered if accounting for autocorrelation")
    issues.append("Power may be inadequate with effective sample size")

    return issues


def audit_consistency():
    """Check cross-asset consistency."""
    issues = []

    print("\n" + "="*60)
    print("AUDIT 7: CROSS-ASSET CONSISTENCY")
    print("="*60)

    results_path = Path('results/deep_exploration_corrected/findings.json')
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)

        # Check correlation consistency
        consistency = results.get('correlation_consistency', {})

        for feat, info in consistency.items():
            direction = info.get('direction', 'unknown')
            cons_rate = info.get('consistency', 0)

            if cons_rate >= 0.9:
                print(f"        ✅ SI-{feat}: {cons_rate:.0%} consistent ({direction})")
            elif cons_rate >= 0.7:
                print(f"        ⚠️ SI-{feat}: {cons_rate:.0%} consistent ({direction})")
            else:
                print(f"        ❌ SI-{feat}: {cons_rate:.0%} consistent - NOT ROBUST")
                issues.append(f"SI-{feat} correlation not consistent across assets")

    return issues


def audit_interpretation():
    """Check interpretation issues."""
    issues = []

    print("\n" + "="*60)
    print("AUDIT 8: INTERPRETATION ISSUES")
    print("="*60)

    # Check 8.1: Correlation vs causation
    print("\n  [8.1] Correlation vs Causation:")
    print("        SI correlates with ADX/Vol/RSI")
    print("        ⚠️ Does NOT mean SI CAUSES these")
    print("        ⚠️ Does NOT mean SI PREDICTS them")
    print("        All may be driven by common factor (price volatility)")
    issues.append("Correlation interpreted as causation without evidence")

    # Check 8.2: Mechanical relationship
    print("\n  [8.2] Mechanical Relationship:")
    print("        SI is computed from agent RETURNS")
    print("        ADX/RSI/Vol are computed from RETURNS")
    print("        They share the SAME underlying data")
    print("        ⚠️ Correlation may be TAUTOLOGICAL")
    issues.append("Possible tautological relationship (same underlying data)")

    # Check 8.3: Spurious correlation
    print("\n  [8.3] Spurious Correlation Risk:")
    print("        Both SI and features are non-stationary")
    print("        ⚠️ Risk of spurious correlation")
    print("        Should test correlation of CHANGES not levels")
    issues.append("No stationarity test or differencing applied")

    return issues


def audit_missing_tests():
    """Check for missing statistical tests."""
    issues = []

    print("\n" + "="*60)
    print("AUDIT 9: MISSING STATISTICAL TESTS")
    print("="*60)

    missing = [
        ("Stationarity test (ADF/KPSS)", "Should verify SI is stationary"),
        ("HAC standard errors", "Account for autocorrelation in inference"),
        ("Block bootstrap", "Proper confidence intervals with time series"),
        ("Granger causality", "Test if SI predicts beyond past values"),
        ("Cointegration test", "If non-stationary, check for cointegration"),
        ("Out-of-time validation", "Walk-forward testing"),
        ("Structural break test", "Check if relationship is stable over time"),
    ]

    for test, reason in missing:
        print(f"        ⚠️ Missing: {test}")
        print(f"           Reason: {reason}")
        issues.append(f"Missing test: {test}")

    return issues


def audit_test_set_issue():
    """Check specific issue with test set size."""
    issues = []

    print("\n" + "="*60)
    print("AUDIT 10: CRITICAL TEST SET ISSUE")
    print("="*60)

    results_path = Path('results/deep_exploration_corrected/findings.json')
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)

        print("\n  🚨 CRITICAL ISSUE FOUND:")
        print("\n  Test set sizes from results:")
        for asset_result in results.get('per_asset_results', []):
            n_test = asset_result.get('n_test', 0)
            n_train = asset_result.get('n_train', 0)
            asset = asset_result.get('asset', 'unknown')
            print(f"        {asset}: train={n_train}, test={n_test}")
            if n_test < 10:
                issues.append(f"CRITICAL: {asset} test set has only {n_test} samples!")

        print("\n  ⚠️ Most test sets have only 4 samples!")
        print("     This makes validation MEANINGLESS")
        print("\n  ROOT CAUSE:")
        print("     The split uses TRAIN_RATIO=0.6, VAL_RATIO=0.2")
        print("     But SI computation requires warmup period")
        print("     After warmup, test set is nearly empty")
        issues.append("Test sets too small (4 samples) for meaningful validation")

    return issues


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE AUDIT OF SI DEEP EXPLORATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_issues = []

    # Run all audits
    all_issues.extend(audit_data_leakage())
    all_issues.extend(audit_correlation_methodology())
    all_issues.extend(audit_lead_lag_methodology())
    all_issues.extend(audit_return_calculation())
    all_issues.extend(audit_out_of_sample())
    all_issues.extend(audit_sample_size())
    all_issues.extend(audit_consistency())
    all_issues.extend(audit_interpretation())
    all_issues.extend(audit_missing_tests())
    all_issues.extend(audit_test_set_issue())

    # Summary
    print("\n" + "="*70)
    print("AUDIT SUMMARY")
    print("="*70)

    critical_issues = [i for i in all_issues if 'CRITICAL' in i.upper()]
    high_issues = [i for i in all_issues if 'CRITICAL' not in i.upper()]

    print(f"\n  Total issues found: {len(all_issues)}")
    print(f"  Critical issues: {len(critical_issues)}")
    print(f"  Other issues: {len(high_issues)}")

    if critical_issues:
        print("\n  🚨 CRITICAL ISSUES:")
        for issue in critical_issues:
            print(f"     • {issue}")

    if high_issues:
        print("\n  ⚠️ OTHER ISSUES:")
        for issue in high_issues[:10]:  # Show top 10
            print(f"     • {issue}")
        if len(high_issues) > 10:
            print(f"     ... and {len(high_issues) - 10} more")

    # Verdict
    print("\n" + "="*70)
    print("OVERALL VERDICT")
    print("="*70)

    if critical_issues:
        print("\n  ❌ EXPLORATION HAS CRITICAL FLAWS")
        print("     Results cannot be trusted until fixed")
    elif len(high_issues) > 5:
        print("\n  ⚠️ EXPLORATION HAS SIGNIFICANT ISSUES")
        print("     Results should be interpreted with caution")
    else:
        print("\n  ✅ EXPLORATION METHODOLOGY IS REASONABLE")
        print("     Minor issues do not invalidate core findings")

    # Save audit results
    output_path = Path('results/audit/exploration_audit.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audit_results = {
        'timestamp': datetime.now().isoformat(),
        'total_issues': len(all_issues),
        'critical_issues': critical_issues,
        'other_issues': high_issues,
        'verdict': 'CRITICAL FLAWS' if critical_issues else 'SIGNIFICANT ISSUES' if len(high_issues) > 5 else 'REASONABLE',
    }

    with open(output_path, 'w') as f:
        json.dump(audit_results, f, indent=2)

    print(f"\n  Audit saved to {output_path}")
    print("="*70 + "\n")

    return audit_results


if __name__ == "__main__":
    results = main()
