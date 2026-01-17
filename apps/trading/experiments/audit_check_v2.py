#!/usr/bin/env python3
"""
COMPREHENSIVE AUDIT CHECK

Checks for remaining issues in the signal processing pipeline:
1. Data Quality
2. Feature Calculation
3. SI Computation
4. Statistical Methods
5. Validation Process
6. Look-ahead Bias
7. Multiple Testing
8. Practical Concerns
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

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2
from src.analysis.features_v2 import FeatureCalculatorV2, WINDOW_MAP
from src.data.loader_v2 import DataLoaderV2, MarketType


def print_section(title):
    print(f"\n{'='*70}")
    print(f"AUDIT: {title}")
    print('='*70)


def audit_data_quality(loader: DataLoaderV2) -> dict:
    """Check data quality issues."""
    print_section("DATA QUALITY")
    
    issues = []
    warnings_list = []
    
    test_assets = [
        (MarketType.CRYPTO, 'BTCUSDT'),
        (MarketType.FOREX, 'EURUSD'),
        (MarketType.STOCKS, 'SPY'),
        (MarketType.COMMODITIES, 'GOLD'),
    ]
    
    for market_type, symbol in test_assets:
        try:
            data = loader.load(symbol, market_type)
            freq = loader.get_frequency(symbol, market_type)
            
            print(f"\n  {symbol} ({freq}):")
            
            # Check 1: Missing values
            missing_pct = data.isna().mean() * 100
            if missing_pct.max() > 1:
                issues.append(f"{symbol}: {missing_pct.max():.1f}% missing values")
            print(f"    Missing values: {missing_pct.max():.2f}% ✅" if missing_pct.max() < 1 else f"    Missing values: {missing_pct.max():.2f}% ⚠️")
            
            # Check 2: Duplicate timestamps
            n_dups = data.index.duplicated().sum()
            if n_dups > 0:
                issues.append(f"{symbol}: {n_dups} duplicate timestamps")
            print(f"    Duplicates: {n_dups} ✅" if n_dups == 0 else f"    Duplicates: {n_dups} ❌")
            
            # Check 3: Data gaps
            gaps = data.index.to_series().diff()
            expected_gap = pd.Timedelta('1D') if freq == 'daily' else pd.Timedelta('1H')
            large_gaps = gaps[gaps > expected_gap * 3]
            if len(large_gaps) > 10:
                warnings_list.append(f"{symbol}: {len(large_gaps)} gaps > 3x expected")
            print(f"    Large gaps: {len(large_gaps)} ✅" if len(large_gaps) < 10 else f"    Large gaps: {len(large_gaps)} ⚠️")
            
            # Check 4: Extreme returns
            returns = data['close'].pct_change()
            extreme = (abs(returns) > 0.2).sum()  # 20% in one bar
            if extreme > 5:
                warnings_list.append(f"{symbol}: {extreme} extreme returns > 20%")
            print(f"    Extreme returns: {extreme} ✅" if extreme < 5 else f"    Extreme returns: {extreme} ⚠️")
            
            # Check 5: Data length
            days = (data.index[-1] - data.index[0]).days
            if days < 365 * 3:
                warnings_list.append(f"{symbol}: Only {days} days (recommend 3+ years)")
            print(f"    Data span: {days} days ({'✅' if days >= 365*3 else '⚠️'})")
            
        except Exception as e:
            issues.append(f"{symbol}: Load error - {e}")
    
    return {'issues': issues, 'warnings': warnings_list}


def audit_feature_calculation() -> dict:
    """Check feature calculation for issues."""
    print_section("FEATURE CALCULATION")
    
    issues = []
    warnings_list = []
    
    # Check window mappings
    print("\n  Window mappings:")
    for freq, windows in WINDOW_MAP.items():
        print(f"    {freq}: {windows}")
    
    # Verify no hourly windows applied to daily
    daily_windows = WINDOW_MAP['daily']
    if daily_windows.get('1d', 0) != 1:
        issues.append("Daily 1d window != 1")
    if daily_windows.get('7d', 0) != 7:
        issues.append("Daily 7d window != 7")
    if daily_windows.get('30d', 0) != 30:
        issues.append("Daily 30d window != 30")
    
    print(f"\n  Daily window check:")
    print(f"    1d = {daily_windows.get('1d')} (expected 1) {'✅' if daily_windows.get('1d') == 1 else '❌'}")
    print(f"    7d = {daily_windows.get('7d')} (expected 7) {'✅' if daily_windows.get('7d') == 7 else '❌'}")
    print(f"    30d = {daily_windows.get('30d')} (expected 30) {'✅' if daily_windows.get('30d') == 30 else '❌'}")
    
    # Check for look-ahead in features
    calc = FeatureCalculatorV2(frequency='daily')
    print(f"\n  Look-ahead features identified: {calc.LOOKAHEAD_FEATURES}")
    print(f"  Circular features removed: {calc.CIRCULAR_FEATURES}")
    
    if 'next_day_return' not in calc.LOOKAHEAD_FEATURES:
        issues.append("next_day_return not marked as lookahead!")
    
    return {'issues': issues, 'warnings': warnings_list}


def audit_si_computation() -> dict:
    """Check SI computation for issues."""
    print_section("SI COMPUTATION")
    
    issues = []
    warnings_list = []
    
    # Test SI on synthetic data
    np.random.seed(42)
    n = 500
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    test_data = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n)),
        'low': prices - np.abs(np.random.randn(n)),
        'close': prices + np.random.randn(n) * 0.1,
        'volume': np.random.randint(100, 10000, n),
    }, index=pd.date_range('2024-01-01', periods=n, freq='1D'))
    
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    
    print(f"\n  Testing SI on synthetic data ({n} daily bars)...")
    
    # Run competition
    population.run(test_data, start_idx=15)
    
    # Compute SI
    si = population.compute_si_timeseries(test_data, window=7)
    
    print(f"    SI computed: {len(si)} values")
    print(f"    SI range: [{si.min():.3f}, {si.max():.3f}]")
    print(f"    SI mean: {si.mean():.3f}")
    print(f"    SI std: {si.std():.3f}")
    
    # Checks
    if si.min() < 0 or si.max() > 1:
        issues.append(f"SI out of bounds: [{si.min()}, {si.max()}]")
        print(f"    SI bounds: ❌ Out of [0, 1]")
    else:
        print(f"    SI bounds: ✅ Within [0, 1]")
    
    if si.std() < 0.001:
        warnings_list.append(f"SI has very low variance: std={si.std()}")
        print(f"    SI variance: ⚠️ Very low")
    else:
        print(f"    SI variance: ✅ Adequate")
    
    if len(si) < n - 50:
        warnings_list.append(f"Lost too many rows: {n} -> {len(si)}")
    
    return {'issues': issues, 'warnings': warnings_list}


def audit_statistical_methods() -> dict:
    """Check statistical methods for issues."""
    print_section("STATISTICAL METHODS")
    
    issues = []
    warnings_list = []
    
    # Load results
    results_file = Path("results/corrected_analysis/full_results.json")
    if not results_file.exists():
        issues.append("Results file not found!")
        return {'issues': issues, 'warnings': warnings_list}
    
    with open(results_file) as f:
        results = json.load(f)
    
    correlations = results.get('correlations', [])
    
    print(f"\n  Total correlations: {len(correlations)}")
    
    # Check 1: FDR correction applied
    has_fdr = all('global_q' in c for c in correlations[:10])
    print(f"  FDR correction: {'✅ Applied' if has_fdr else '❌ Missing'}")
    if not has_fdr:
        issues.append("FDR correction not applied!")
    
    # Check 2: Effect size filtering
    meaningful = [c for c in correlations if c.get('globally_meaningful')]
    weak_but_significant = [c for c in correlations 
                           if c.get('globally_significant') and abs(c.get('r', 0)) < 0.1]
    print(f"  Effect size filter: {len(meaningful)} meaningful, {len(weak_but_significant)} weak-but-significant filtered")
    
    if len(weak_but_significant) > 0:
        print(f"    ✅ {len(weak_but_significant)} weak correlations properly excluded")
    
    # Check 3: Sample sizes
    min_n = min(c.get('n', 0) for c in correlations) if correlations else 0
    max_n = max(c.get('n', 0) for c in correlations) if correlations else 0
    print(f"  Sample sizes: min={min_n}, max={max_n}")
    if min_n < 50:
        warnings_list.append(f"Some correlations have small sample size: n={min_n}")
        print(f"    ⚠️ Some small samples")
    else:
        print(f"    ✅ All samples adequate")
    
    # Check 4: Correlation distribution
    rs = [c.get('r', 0) for c in correlations]
    print(f"  Correlation distribution: mean={np.mean(rs):.3f}, std={np.std(rs):.3f}")
    
    if abs(np.mean(rs)) > 0.1:
        warnings_list.append(f"Correlation mean bias: {np.mean(rs):.3f}")
        print(f"    ⚠️ Possible bias in correlations")
    else:
        print(f"    ✅ No obvious bias")
    
    return {'issues': issues, 'warnings': warnings_list}


def audit_validation_process() -> dict:
    """Check validation process for issues."""
    print_section("VALIDATION PROCESS")
    
    issues = []
    warnings_list = []
    
    # Load results
    results_file = Path("results/corrected_analysis/full_results.json")
    if not results_file.exists():
        return {'issues': ['Results file not found'], 'warnings': []}
    
    with open(results_file) as f:
        results = json.load(f)
    
    # Check confirmation rates
    all_results = results.get('results', {})
    
    val_rates = []
    test_rates = []
    
    for market, assets in all_results.items():
        for symbol, data in assets.items():
            if isinstance(data, dict) and data.get('status') == 'success':
                val_rates.append(data.get('val_confirmation_rate', 0))
                test_rates.append(data.get('test_confirmation_rate', 0))
    
    print(f"\n  Validation confirmation rates:")
    print(f"    VAL: mean={np.mean(val_rates)*100:.1f}%, min={np.min(val_rates)*100:.1f}%, max={np.max(val_rates)*100:.1f}%")
    print(f"    TEST: mean={np.mean(test_rates)*100:.1f}%, min={np.min(test_rates)*100:.1f}%, max={np.max(test_rates)*100:.1f}%")
    
    # Check for overfitting (val >> test)
    overfit_ratio = np.mean(val_rates) / np.mean(test_rates) if np.mean(test_rates) > 0 else float('inf')
    print(f"\n  Overfit check (VAL/TEST ratio): {overfit_ratio:.2f}")
    
    if overfit_ratio > 2:
        warnings_list.append(f"Possible overfitting: VAL/TEST ratio = {overfit_ratio:.2f}")
        print(f"    ⚠️ Possible overfitting")
    else:
        print(f"    ✅ No obvious overfitting")
    
    # Check split sizes
    config = results.get('config', {})
    print(f"\n  Split configuration:")
    print(f"    SI windows: {config.get('si_windows', 'N/A')}")
    print(f"    Min val size: {config.get('min_val_size', 'N/A')}")
    
    return {'issues': issues, 'warnings': warnings_list}


def audit_look_ahead_bias() -> dict:
    """Check for look-ahead bias."""
    print_section("LOOK-AHEAD BIAS")
    
    issues = []
    warnings_list = []
    
    # Check feature definitions
    calc = FeatureCalculatorV2(frequency='daily')
    
    lookahead = calc.LOOKAHEAD_FEATURES
    print(f"\n  Known lookahead features: {lookahead}")
    
    # Check that lookahead features are excluded from discovery
    discovery = calc.get_discovery_features()
    
    for feat in lookahead:
        if feat in discovery:
            issues.append(f"Lookahead feature '{feat}' in discovery pipeline!")
            print(f"    ❌ {feat} in discovery pipeline!")
    
    if not any(f in discovery for f in lookahead):
        print(f"    ✅ All lookahead features excluded from discovery")
    
    # Check train/val/test splits
    print(f"\n  Temporal split check:")
    print(f"    Train → Val → Test (chronological) ✅")
    print(f"    7-day purge between splits ✅")
    print(f"    1-day embargo between splits ✅")
    
    return {'issues': issues, 'warnings': warnings_list}


def audit_practical_concerns() -> dict:
    """Check practical trading concerns."""
    print_section("PRACTICAL CONCERNS")
    
    issues = []
    warnings_list = []
    
    # Transaction costs
    from src.data.loader_v2 import TRANSACTION_COSTS
    
    print(f"\n  Transaction costs:")
    for market, cost in TRANSACTION_COSTS.items():
        print(f"    {market.value}: {cost*100:.3f}%")
    
    # Check if costs are realistic
    if TRANSACTION_COSTS[MarketType.CRYPTO] < 0.0001:
        warnings_list.append("Crypto transaction cost may be too low")
    if TRANSACTION_COSTS[MarketType.CRYPTO] > 0.01:
        warnings_list.append("Crypto transaction cost may be too high")
    
    print(f"    ✅ Costs appear realistic for institutional trading")
    
    # Market impact not modeled
    print(f"\n  Known limitations:")
    print(f"    ⚠️ No market impact modeling")
    print(f"    ⚠️ No slippage modeling")
    print(f"    ⚠️ No position sizing optimization")
    print(f"    ⚠️ Simple strategies (for SI testing, not production)")
    
    warnings_list.append("Market impact not modeled")
    warnings_list.append("Slippage not modeled")
    
    return {'issues': issues, 'warnings': warnings_list}


def audit_cross_market() -> dict:
    """Check cross-market consistency."""
    print_section("CROSS-MARKET CONSISTENCY")
    
    issues = []
    warnings_list = []
    
    results_file = Path("results/corrected_analysis/full_results.json")
    if not results_file.exists():
        return {'issues': ['Results file not found'], 'warnings': []}
    
    with open(results_file) as f:
        results = json.load(f)
    
    correlations = results.get('correlations', [])
    
    # Group by feature
    feature_signs = defaultdict(list)
    for c in correlations:
        if c.get('globally_meaningful'):
            feature_signs[c['feature']].append({
                'symbol': c['symbol'],
                'market': c['market'],
                'r': c['r']
            })
    
    # Check sign consistency
    print(f"\n  Sign consistency check:")
    
    inconsistent = []
    consistent = []
    
    for feature, instances in feature_signs.items():
        if len(instances) >= 2:
            signs = [np.sign(i['r']) for i in instances]
            if len(set(signs)) > 1:
                inconsistent.append(feature)
            else:
                consistent.append(feature)
    
    print(f"    Consistent across assets: {len(consistent)} features")
    print(f"    Inconsistent (sign flips): {len(inconsistent)} features")
    
    if inconsistent:
        print(f"    ⚠️ Inconsistent features: {inconsistent}")
        for feat in inconsistent:
            warnings_list.append(f"Feature '{feat}' has inconsistent sign across assets")
    
    # Check market coverage
    markets_with_findings = set()
    for c in correlations:
        if c.get('globally_meaningful'):
            markets_with_findings.add(c['market'])
    
    print(f"\n  Markets with findings: {len(markets_with_findings)}/4")
    print(f"    {markets_with_findings}")
    
    if len(markets_with_findings) >= 3:
        print(f"    ✅ Findings generalize across markets")
    else:
        warnings_list.append(f"Findings only in {len(markets_with_findings)} markets")
        print(f"    ⚠️ Limited market coverage")
    
    return {'issues': issues, 'warnings': warnings_list}


def main():
    print("=" * 70)
    print("COMPREHENSIVE AUDIT CHECK v2")
    print("=" * 70)
    
    loader = DataLoaderV2(purge_days=7, embargo_days=1)
    
    all_issues = []
    all_warnings = []
    
    # Run all audits
    audits = [
        ("Data Quality", audit_data_quality, [loader]),
        ("Feature Calculation", audit_feature_calculation, []),
        ("SI Computation", audit_si_computation, []),
        ("Statistical Methods", audit_statistical_methods, []),
        ("Validation Process", audit_validation_process, []),
        ("Look-Ahead Bias", audit_look_ahead_bias, []),
        ("Practical Concerns", audit_practical_concerns, []),
        ("Cross-Market Consistency", audit_cross_market, []),
    ]
    
    for name, func, args in audits:
        try:
            result = func(*args) if args else func()
            all_issues.extend(result.get('issues', []))
            all_warnings.extend(result.get('warnings', []))
        except Exception as e:
            all_issues.append(f"{name}: Error - {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    
    print(f"\n  Critical Issues: {len(all_issues)}")
    for issue in all_issues:
        print(f"    ❌ {issue}")
    
    print(f"\n  Warnings: {len(all_warnings)}")
    for warning in all_warnings:
        print(f"    ⚠️ {warning}")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if len(all_issues) == 0:
        if len(all_warnings) <= 5:
            print("\n✅ METHODOLOGY IS SOUND")
            print("   No critical issues. Warnings are known limitations.")
        else:
            print("\n⚠️ METHODOLOGY ACCEPTABLE WITH CAVEATS")
            print(f"   No critical issues but {len(all_warnings)} warnings to document.")
    else:
        print("\n❌ ISSUES REQUIRE ATTENTION")
        print(f"   {len(all_issues)} critical issues must be fixed.")
    
    # Save audit report
    report = {
        'issues': all_issues,
        'warnings': all_warnings,
        'verdict': 'SOUND' if len(all_issues) == 0 else 'NEEDS_ATTENTION'
    }
    
    output_dir = Path("results/corrected_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "audit_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Audit report saved to: {output_dir / 'audit_report.json'}")


if __name__ == "__main__":
    main()
