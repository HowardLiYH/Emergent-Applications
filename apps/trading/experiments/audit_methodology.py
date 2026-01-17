#!/usr/bin/env python3
"""
COMPREHENSIVE METHODOLOGY AUDIT

Checks for:
1. Data Quality Issues
2. Statistical Validity Issues
3. Code/Implementation Bugs
4. Generalizability Concerns
5. Look-ahead Bias
6. Multiple Testing Issues
7. Overfitting Risks
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def audit_data_quality():
    """Audit 1: Data Quality Issues"""
    print("\n" + "=" * 70)
    print("AUDIT 1: DATA QUALITY")
    print("=" * 70)

    issues = []

    # Check all data files
    data_dirs = ['data/crypto', 'data/forex', 'data/stocks', 'data/commodities']

    for data_dir in data_dirs:
        if not Path(data_dir).exists():
            issues.append(f"❌ CRITICAL: {data_dir} does not exist")
            continue

        for csv_file in Path(data_dir).glob("*.csv"):
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

            # Check 1: Volume issues (zeros or negative)
            if 'volume' in df.columns:
                zero_vol_pct = (df['volume'] == 0).mean() * 100
                if zero_vol_pct > 10:
                    issues.append(f"⚠️  {csv_file.name}: {zero_vol_pct:.1f}% zero volume - yfinance data quality issue")

            # Check 2: Missing values
            missing_pct = df.isnull().mean().max() * 100
            if missing_pct > 1:
                issues.append(f"⚠️  {csv_file.name}: {missing_pct:.1f}% missing values")

            # Check 3: Data gaps
            if len(df) > 100:
                time_diff = pd.Series(df.index).diff().dropna()
                max_gap = time_diff.max()
                if hasattr(max_gap, 'days') and max_gap.days > 3:
                    issues.append(f"⚠️  {csv_file.name}: Max gap of {max_gap} - potential data quality issue")

            # Check 4: Extreme returns
            if 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                extreme_pct = (abs(returns) > 0.2).mean() * 100
                if extreme_pct > 1:
                    issues.append(f"⚠️  {csv_file.name}: {extreme_pct:.1f}% returns > 20% - verify data integrity")

            # Check 5: Data source
            print(f"   {csv_file.name}: {len(df)} rows, {df.index[0]} to {df.index[-1]}")

    # CRITICAL: Data source verification
    print("\n   DATA SOURCE VERIFICATION:")
    print("   - All data from yfinance (Yahoo Finance)")
    print("   - Crypto data has known volume issues (zeros)")
    print("   - Forex has no real volume data (synthetic)")
    print("   - Stocks/Commodities from US exchanges only")
    issues.append("⚠️  DATA SOURCE: yfinance has known accuracy issues for intraday data")
    issues.append("⚠️  VOLUME DATA: Forex volume is synthetic, crypto volume often zero")

    return issues


def audit_statistical_validity():
    """Audit 2: Statistical Validity Issues"""
    print("\n" + "=" * 70)
    print("AUDIT 2: STATISTICAL VALIDITY")
    print("=" * 70)

    issues = []

    # Load results
    results_dir = Path("results/cross_market_full")
    if not results_dir.exists():
        issues.append("❌ CRITICAL: No cross-market results found")
        return issues

    with open(results_dir / "full_analysis.json") as f:
        results = json.load(f)

    # Check 1: Multiple testing correction
    print("\n   MULTIPLE TESTING CHECK:")
    total_tests = 0
    significant_count = 0

    for market, market_data in results.get('detailed_results', {}).items():
        for asset, asset_data in market_data.items():
            if asset_data.get('status') == 'success':
                corrs = asset_data.get('correlations', [])
                total_tests += len(corrs)
                significant_count += sum(1 for c in corrs if c.get('significant', False))

    # Expected false positives at p<0.05
    expected_fp = total_tests * 0.05
    print(f"   Total tests: {total_tests}")
    print(f"   Significant: {significant_count}")
    print(f"   Expected false positives (p<0.05): {expected_fp:.0f}")

    if significant_count < expected_fp * 2:
        issues.append(f"⚠️  MULTIPLE TESTING: {significant_count} significant out of {total_tests} tests - could be mostly false positives")
    else:
        print(f"   ✅ Significant count ({significant_count}) >> expected false positives ({expected_fp:.0f})")

    # Check 2: FDR correction applied?
    print("\n   FDR CORRECTION CHECK:")
    issues.append("⚠️  FDR CORRECTION: Benjamini-Hochberg not applied to cross-market analysis")
    print("   ❌ FDR correction was NOT applied in cross-market analysis")

    # Check 3: Effect sizes
    print("\n   EFFECT SIZE CHECK:")
    small_effects = 0
    for market, market_data in results.get('detailed_results', {}).items():
        for asset, asset_data in market_data.items():
            if asset_data.get('status') == 'success':
                for corr in asset_data.get('correlations', []):
                    if corr.get('significant') and abs(corr.get('r', 0)) < 0.1:
                        small_effects += 1

    if small_effects > 10:
        issues.append(f"⚠️  EFFECT SIZES: {small_effects} significant correlations have |r| < 0.1 - statistically significant but practically meaningless")
        print(f"   ⚠️ {small_effects} correlations are significant but have tiny effect sizes")

    # Check 4: Sample size for Granger causality
    print("\n   GRANGER CAUSALITY CHECK:")
    prediction_file = Path("results/si_correlations/prediction_results.json")
    if prediction_file.exists():
        with open(prediction_file) as f:
            pred = json.load(f)
        # Granger tests need large samples for reliability
        issues.append("⚠️  GRANGER TESTS: May need 1000+ observations for reliable results")

    return issues


def audit_methodology():
    """Audit 3: Methodology Issues"""
    print("\n" + "=" * 70)
    print("AUDIT 3: METHODOLOGY ISSUES")
    print("=" * 70)

    issues = []

    # Issue 1: SI computation validity
    print("\n   SI COMPUTATION:")
    print("   - SI computed via NichePopulation competition")
    print("   - Rolling window of 168 hours (7 days)")
    print("   - 3 agents per strategy, 3 strategies = 9 agents total")
    issues.append("⚠️  SI SENSITIVITY: SI depends on arbitrary choices (window=168, n_agents=3)")

    # Issue 2: Regime classification
    print("\n   REGIME CLASSIFICATION:")
    print("   - Regimes classified by return percentile thresholds")
    print("   - Thresholds: >70th percentile = bull, <30th = bear, else = sideways")
    issues.append("⚠️  REGIME THRESHOLDS: 30/70 percentile thresholds are arbitrary")

    # Issue 3: Strategy definitions
    print("\n   STRATEGY DEFINITIONS:")
    print("   - 3 simple strategies: Momentum, Mean Reversion, Breakout")
    print("   - All use fixed lookback periods (default)")
    issues.append("⚠️  STRATEGY SIMPLICITY: Only 3 simple strategies - may not capture market complexity")

    # Issue 4: Feature definitions
    print("\n   FEATURE COMPUTATION:")
    print("   - 34 features computed with various lookback windows")
    print("   - Some features use overlapping windows")
    issues.append("⚠️  FEATURE OVERLAP: Rolling windows create autocorrelation in features")

    # Issue 5: Temporal split
    print("\n   DATA SPLITTING:")
    print("   - 70/15/15 temporal split (train/val/test)")
    print("   - No purging or embargo between sets")
    issues.append("⚠️  LOOK-AHEAD LEAKAGE: No purging/embargo between train/val/test splits")

    # Issue 6: Market hours
    print("\n   MARKET HOURS:")
    print("   - Crypto: 24/7")
    print("   - Forex: 24/5 (weekends excluded)")
    print("   - Stocks: Market hours only (~7 hours/day)")
    print("   - Different sample sizes due to market hours")
    issues.append("⚠️  UNEQUAL SAMPLES: Stock data has ~1,900 rows vs Crypto ~9,500 - comparison may be unfair")

    return issues


def audit_implementation():
    """Audit 4: Code/Implementation Bugs"""
    print("\n" + "=" * 70)
    print("AUDIT 4: IMPLEMENTATION AUDIT")
    print("=" * 70)

    issues = []

    # Check 1: Timezone handling
    print("\n   TIMEZONE HANDLING:")
    print("   - Crypto: UTC")
    print("   - Forex: Mixed (converted to UTC)")
    print("   - Stocks: EST (converted to UTC)")
    print("   - Commodities: EST (converted to UTC)")
    issues.append("⚠️  TIMEZONE: Different markets have different trading hours - SI may not be comparable")

    # Check 2: NaN handling
    print("\n   NAN HANDLING CHECK:")
    # Read a sample feature file
    feature_files = list(Path("results/si_correlations").glob("features_*.csv"))
    if feature_files:
        df = pd.read_csv(feature_files[0], index_col=0)
        nan_pct = df.isnull().mean().mean() * 100
        print(f"   Average NaN percentage in features: {nan_pct:.1f}%")
        if nan_pct > 20:
            issues.append(f"⚠️  HIGH NAN RATE: {nan_pct:.1f}% NaN in features - results may be biased")

    # Check 3: Index alignment
    print("\n   INDEX ALIGNMENT:")
    si_files = list(Path("results/si_correlations").glob("si_*.csv"))
    if si_files and feature_files:
        si = pd.read_csv(si_files[0], index_col=0)
        feat = pd.read_csv(feature_files[0], index_col=0)
        overlap = len(set(si.index) & set(feat.index))
        total = max(len(si), len(feat))
        overlap_pct = overlap / total * 100
        print(f"   SI-Feature index overlap: {overlap_pct:.1f}%")
        if overlap_pct < 80:
            issues.append(f"⚠️  LOW OVERLAP: Only {overlap_pct:.1f}% overlap between SI and features")

    return issues


def audit_generalizability():
    """Audit 5: Generalizability Concerns"""
    print("\n" + "=" * 70)
    print("AUDIT 5: GENERALIZABILITY")
    print("=" * 70)

    issues = []

    # Issue 1: Time period
    print("\n   TIME PERIOD:")
    print("   - Data from Dec 2024 to Jan 2026 (~13 months)")
    print("   - Single market regime period")
    issues.append("⚠️  SHORT PERIOD: ~13 months of data may not capture full market cycles")

    # Issue 2: Geographic bias
    print("\n   GEOGRAPHIC COVERAGE:")
    print("   - Stocks: US only (SPY, QQQ, AAPL)")
    print("   - Forex: Major pairs only")
    print("   - No Asian, European, or Emerging market stocks")
    issues.append("⚠️  GEOGRAPHIC BIAS: Only US stocks tested - may not generalize globally")

    # Issue 3: Asset selection bias
    print("\n   ASSET SELECTION:")
    print("   - Crypto: Top 3 by market cap")
    print("   - Stocks: Major ETFs and FAANG")
    print("   - Commodities: Only 2 assets")
    issues.append("⚠️  SELECTION BIAS: Only most liquid assets - SI may not work for smaller assets")

    # Issue 4: Market regime
    print("\n   MARKET REGIME:")
    print("   - Period includes potential bull/bear transitions")
    print("   - No explicit regime stratification")
    issues.append("⚠️  REGIME DEPENDENCY: Results may be regime-specific")

    return issues


def audit_trading_strategy():
    """Audit 6: Trading Strategy Validity"""
    print("\n" + "=" * 70)
    print("AUDIT 6: TRADING STRATEGY AUDIT")
    print("=" * 70)

    issues = []

    # Load strategy results
    strategy_file = Path("results/si_strategy/strategy_results.json")
    if not strategy_file.exists():
        issues.append("⚠️  No strategy results found")
        return issues

    with open(strategy_file) as f:
        results = json.load(f)

    # Issue 1: Transaction costs
    print("\n   TRANSACTION COSTS:")
    print("   - NOT included in backtest")
    print("   - Crypto: ~0.1% per trade")
    print("   - Stocks: ~$0-5 per trade")
    issues.append("❌ CRITICAL: No transaction costs - profits may disappear with realistic costs")

    # Issue 2: Slippage
    print("\n   SLIPPAGE:")
    print("   - NOT modeled")
    print("   - Assumes perfect execution at close prices")
    issues.append("❌ CRITICAL: No slippage modeled - unrealistic execution assumptions")

    # Issue 3: Market impact
    print("\n   MARKET IMPACT:")
    print("   - NOT modeled")
    print("   - Assumes infinite liquidity")
    issues.append("❌ CRITICAL: No market impact - strategy may move prices in practice")

    # Issue 4: Returns analysis
    print("\n   RETURNS ANALYSIS:")
    for symbol, res in results.items():
        if res.get('status') == 'SUCCESS':
            test_ret = res['test'].get('total_return', 0) * 100
            test_sharpe = res['test'].get('sharpe_ratio', 0)
            print(f"   {symbol}: Return={test_ret:+.2f}%, Sharpe={test_sharpe:.2f}")

    avg_return = np.mean([r['test'].get('total_return', 0) for r in results.values() if r.get('status') == 'SUCCESS']) * 100
    if avg_return < 1:
        issues.append(f"⚠️  LOW RETURNS: Average TEST return is only {avg_return:+.2f}% - may be noise")

    # Issue 5: Overfitting check
    print("\n   OVERFITTING CHECK:")
    for symbol, res in results.items():
        if res.get('status') == 'SUCCESS':
            train_sharpe = res['train'].get('sharpe_ratio', 0)
            test_sharpe = res['test'].get('sharpe_ratio', 0)
            degradation = (train_sharpe - test_sharpe) / abs(train_sharpe) if train_sharpe != 0 else 0
            if degradation > 0.5:
                print(f"   ⚠️ {symbol}: Sharpe degradation {degradation:.0%} (train→test)")

    return issues


def main():
    print("=" * 70)
    print("COMPREHENSIVE METHODOLOGY AUDIT")
    print("=" * 70)

    all_issues = []

    # Run all audits
    all_issues.extend(audit_data_quality())
    all_issues.extend(audit_statistical_validity())
    all_issues.extend(audit_methodology())
    all_issues.extend(audit_implementation())
    all_issues.extend(audit_generalizability())
    all_issues.extend(audit_trading_strategy())

    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    critical = [i for i in all_issues if i.startswith("❌")]
    warnings = [i for i in all_issues if i.startswith("⚠️")]

    print(f"\n❌ CRITICAL ISSUES: {len(critical)}")
    for issue in critical:
        print(f"   {issue}")

    print(f"\n⚠️  WARNINGS: {len(warnings)}")
    for issue in warnings:
        print(f"   {issue}")

    # Save audit report
    output_dir = Path("results/audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "methodology_audit.json", "w") as f:
        json.dump({
            'critical_issues': critical,
            'warnings': warnings,
            'total_issues': len(all_issues)
        }, f, indent=2)

    print(f"\nAudit saved to: {output_dir / 'methodology_audit.json'}")

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if len(critical) > 0:
        print("\n🔴 METHODOLOGY HAS CRITICAL ISSUES")
        print("   Results should NOT be used for production trading without addressing:")
        for c in critical:
            print(f"   - {c[2:]}")
    else:
        print("\n🟡 METHODOLOGY HAS WARNINGS BUT NO CRITICAL ISSUES")

    print("\n📋 RECOMMENDED ACTIONS:")
    print("   1. Add transaction costs to strategy backtest")
    print("   2. Apply FDR correction to all p-values")
    print("   3. Add purging/embargo to train/val/test split")
    print("   4. Test on longer time periods (3-5 years)")
    print("   5. Test on more diverse assets (international stocks)")
    print("   6. Validate with walk-forward analysis")


if __name__ == "__main__":
    main()
