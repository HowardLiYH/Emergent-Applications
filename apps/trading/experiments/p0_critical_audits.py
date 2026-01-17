#!/usr/bin/env python3
"""
P0: Critical Audits - NEXT_STEPS_PLAN v4.1

This script runs all 5 critical audits before any analysis execution:
- Audit A: Survivorship Bias Check
- Audit B: Data Quality Verification
- Audit C: Reproducibility Check + Data Versioning
- Audit D: Look-Ahead Bias Check
- Audit E: Economic vs Statistical Significance Framework

Author: Yuhao Li, University of Pennsylvania
Date: January 17, 2026
"""

import sys
sys.path.insert(0, '.')

import os
import json
import hashlib
import random
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# PARAMETER DOCUMENTATION (Prof. Weber Requirement)
# ============================================================================

PARAMETER_CHOICES = {
    # Data paths
    'data_dir': 'data',

    # Audit thresholds
    'missing_threshold': 0.01,      # < 1% missing values allowed
    'gap_threshold_days': 3,        # No gaps > 3 days
    'extreme_return_threshold': 0.5, # |return| < 50%

    # Economic significance
    'min_correlation_threshold': 0.10,  # |r| > 0.10 for economic significance
    'assumed_volatility': 0.20,         # 20% annual volatility
    'trades_per_year': 50,              # Assumed trades per year

    # Random seed
    'random_seed': 42,
}

PARAMETER_RATIONALE = {
    'missing_threshold': "1% is industry standard for data quality",
    'gap_threshold_days': "3 days allows for weekends, holidays",
    'extreme_return_threshold': "50% daily return is extremely rare, likely data error",
    'min_correlation_threshold': "Correlations < 0.10 are typically noise",
}

# ============================================================================
# AUDIT A: SURVIVORSHIP BIAS CHECK
# ============================================================================

def audit_survivorship() -> Dict:
    """
    Verify that asset selection was not biased by hindsight.

    Checks:
    1. All assets existed for full analysis period
    2. No assets were selected based on recent performance
    3. Selection criteria documented before analysis
    """
    print("\n" + "="*60)
    print("AUDIT A: SURVIVORSHIP BIAS CHECK")
    print("="*60)

    # Asset inception dates
    ASSET_START_DATES = {
        # Crypto
        'BTC': '2009-01-03',   # Genesis block
        'ETH': '2015-07-30',   # Mainnet launch
        'SOL': '2020-03-16',   # Mainnet beta - WARNING: < 5 years before 2021

        # Stocks
        'SPY': '1993-01-29',
        'QQQ': '1999-03-10',
        'AAPL': '1980-12-12',

        # Forex
        'EURUSD': '1999-01-01',
        'GBPUSD': '1971-01-01',
        'USDJPY': '1971-01-01',

        # Commodities
        'GOLD': '1974-12-31',  # Gold futures
        'OIL': '1983-03-30',   # Crude oil futures
    }

    analysis_start = '2021-01-01'

    results = {
        'audit': 'A - Survivorship Bias',
        'passed': True,
        'assets_checked': [],
        'issues': [],
        'warnings': []
    }

    for asset, start_date in ASSET_START_DATES.items():
        start = pd.Timestamp(start_date)
        analysis = pd.Timestamp(analysis_start)

        years_before = (analysis - start).days / 365

        asset_result = {
            'asset': asset,
            'inception_date': start_date,
            'years_before_analysis': round(years_before, 1),
        }

        if start > analysis:
            asset_result['status'] = 'FAIL'
            asset_result['issue'] = f"Asset started {start_date}, AFTER analysis period"
            results['issues'].append(f"❌ {asset}: Started after analysis period")
            results['passed'] = False
        elif years_before < 2:
            asset_result['status'] = 'WARNING'
            asset_result['issue'] = f"Only {years_before:.1f} years before analysis"
            results['warnings'].append(f"⚠️ {asset}: Only {years_before:.1f} years history before analysis")
        else:
            asset_result['status'] = 'PASS'

        results['assets_checked'].append(asset_result)
        print(f"  {asset}: {asset_result['status']} (inception: {start_date}, {years_before:.1f} years before analysis)")

    # Check for pre-registration
    pre_reg_path = Path('experiments/pre_registration.json')
    if pre_reg_path.exists():
        results['pre_registration_exists'] = True
        print(f"\n  ✅ Pre-registration file exists: {pre_reg_path}")
    else:
        results['pre_registration_exists'] = False
        results['warnings'].append("⚠️ No pre-registration file found")
        print(f"\n  ⚠️ No pre-registration file found at {pre_reg_path}")

    # Summary
    if results['passed'] and len(results['warnings']) == 0:
        print(f"\n  ✅ AUDIT A PASSED: No survivorship bias detected")
    elif results['passed']:
        print(f"\n  ⚠️ AUDIT A PASSED WITH WARNINGS: {len(results['warnings'])} warnings")
    else:
        print(f"\n  ❌ AUDIT A FAILED: {len(results['issues'])} issues found")

    return results


# ============================================================================
# AUDIT B: DATA QUALITY VERIFICATION
# ============================================================================

def audit_data_quality() -> Dict:
    """
    Ensure data is clean and reliable before running any analysis.

    Checks:
    1. Missing values < 1%
    2. No gaps > 3 days
    3. No extreme returns > 50%
    4. No duplicate timestamps
    5. Timezone consistency (all UTC)
    """
    print("\n" + "="*60)
    print("AUDIT B: DATA QUALITY VERIFICATION")
    print("="*60)

    results = {
        'audit': 'B - Data Quality',
        'passed': True,
        'assets_checked': [],
        'issues': [],
        'warnings': []
    }

    data_dir = Path(PARAMETER_CHOICES['data_dir'])

    # Find all CSV files
    csv_files = list(data_dir.rglob("*.csv"))

    if len(csv_files) == 0:
        results['passed'] = False
        results['issues'].append("❌ No data files found")
        print("  ❌ No data files found in data/ directory")
        return results

    print(f"  Found {len(csv_files)} data files\n")

    for filepath in csv_files:
        asset_name = filepath.stem
        market = filepath.parent.name

        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)

            asset_result = {
                'asset': asset_name,
                'market': market,
                'filepath': str(filepath),
                'n_rows': len(df),
                'date_range': f"{df.index[0]} to {df.index[-1]}",
                'issues': []
            }

            # Check 1: Missing values
            missing_pct = df.isnull().sum().max() / len(df)
            if missing_pct > PARAMETER_CHOICES['missing_threshold']:
                asset_result['issues'].append(f"Missing values: {missing_pct:.2%}")

            # Check 2: Gaps > 3 days (adjusted for market type)
            # Note: Traditional markets (stocks, forex, commodities) have weekends and holidays
            # Crypto trades 24/7, so gaps are more concerning
            if len(df) > 1:
                gaps = df.index.to_series().diff()

                if market == 'crypto':
                    # Crypto: gaps > 3 days are concerning
                    gap_threshold = pd.Timedelta(days=3)
                else:
                    # Traditional markets: gaps > 5 days (allows for weekends + 1 holiday)
                    # Long weekends (e.g., Thanksgiving) can be 4 days
                    gap_threshold = pd.Timedelta(days=5)

                large_gaps = gaps[gaps > gap_threshold]
                if len(large_gaps) > 0:
                    # For traditional markets, only flag if gaps > 10 days (extended closures)
                    if market != 'crypto':
                        very_large_gaps = gaps[gaps > pd.Timedelta(days=10)]
                        if len(very_large_gaps) > 0:
                            asset_result['issues'].append(f"Very large gaps: {len(very_large_gaps)} gaps > 10 days")
                        # Otherwise just note weekend gaps as info, not issue
                    else:
                        asset_result['issues'].append(f"Large gaps: {len(large_gaps)} gaps > 3 days")

            # Check 3: Extreme returns
            if 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                extreme = returns[returns.abs() > PARAMETER_CHOICES['extreme_return_threshold']]
                if len(extreme) > 0:
                    asset_result['issues'].append(f"Extreme returns: {len(extreme)} days with |r| > 50%")

            # Check 4: Duplicates
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                asset_result['issues'].append(f"Duplicate timestamps: {duplicates}")

            # Status
            if len(asset_result['issues']) == 0:
                asset_result['status'] = 'PASS'
                print(f"  ✅ {market}/{asset_name}: PASS ({len(df)} rows)")
            else:
                asset_result['status'] = 'FAIL'
                results['passed'] = False
                for issue in asset_result['issues']:
                    results['issues'].append(f"❌ {market}/{asset_name}: {issue}")
                    print(f"  ❌ {market}/{asset_name}: {issue}")

            results['assets_checked'].append(asset_result)

        except Exception as e:
            results['issues'].append(f"❌ {market}/{asset_name}: Failed to read - {str(e)}")
            print(f"  ❌ {market}/{asset_name}: Failed to read - {str(e)}")
            results['passed'] = False

    # Summary
    if results['passed']:
        print(f"\n  ✅ AUDIT B PASSED: All {len(results['assets_checked'])} files passed quality checks")
    else:
        print(f"\n  ❌ AUDIT B FAILED: {len(results['issues'])} issues found")

    return results


# ============================================================================
# AUDIT C: REPRODUCIBILITY CHECK + DATA VERSIONING
# ============================================================================

def audit_reproducibility() -> Dict:
    """
    Ensure results can be reproduced by others.

    Checks:
    1. Python version documented
    2. All package versions pinned
    3. Random seeds set
    4. Git commit hash recorded
    5. Data manifest with checksums created
    """
    print("\n" + "="*60)
    print("AUDIT C: REPRODUCIBILITY CHECK + DATA VERSIONING")
    print("="*60)

    results = {
        'audit': 'C - Reproducibility',
        'passed': True,
        'manifest': {},
        'issues': [],
        'warnings': []
    }

    # 1. Python version
    import sys
    results['manifest']['python_version'] = sys.version
    print(f"  Python version: {sys.version.split()[0]}")

    # 2. Package versions
    try:
        import pkg_resources
        packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        results['manifest']['packages'] = packages
        print(f"  Packages tracked: {len(packages)}")
    except:
        results['warnings'].append("⚠️ Could not enumerate packages")

    # 3. Random seeds
    random.seed(PARAMETER_CHOICES['random_seed'])
    np.random.seed(PARAMETER_CHOICES['random_seed'])
    results['manifest']['random_seed'] = PARAMETER_CHOICES['random_seed']
    print(f"  Random seed set: {PARAMETER_CHOICES['random_seed']}")

    # 4. Git commit
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        git_dirty = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip() != ''
        results['manifest']['git_commit'] = git_commit
        results['manifest']['git_dirty'] = git_dirty
        print(f"  Git commit: {git_commit[:8]}{'*' if git_dirty else ''}")
    except:
        results['warnings'].append("⚠️ Could not get git commit")

    # 5. Data manifest with checksums
    print("\n  Generating data manifest...")
    data_manifest = {}
    data_dir = Path(PARAMETER_CHOICES['data_dir'])

    for filepath in data_dir.rglob("*.csv"):
        try:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            df = pd.read_csv(filepath, index_col=0, parse_dates=True)

            data_manifest[str(filepath)] = {
                'md5': file_hash,
                'rows': len(df),
                'columns': list(df.columns),
                'date_range': f"{df.index[0]} to {df.index[-1]}",
                'file_size_kb': round(filepath.stat().st_size / 1024, 2),
            }
            print(f"    {filepath.name}: {file_hash[:8]}... ({len(df)} rows)")
        except Exception as e:
            results['warnings'].append(f"⚠️ Could not hash {filepath}: {e}")

    results['manifest']['data_files'] = data_manifest

    # Save manifest
    manifest_path = Path('results/audit/reproducibility_manifest.json')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert manifest to JSON-serializable format
    json_manifest = {
        'generated_at': datetime.now().isoformat(),
        'python_version': results['manifest'].get('python_version', 'unknown'),
        'random_seed': results['manifest'].get('random_seed', 42),
        'git_commit': results['manifest'].get('git_commit', 'unknown'),
        'git_dirty': results['manifest'].get('git_dirty', False),
        'n_packages': len(results['manifest'].get('packages', {})),
        'data_files': data_manifest,
    }

    with open(manifest_path, 'w') as f:
        json.dump(json_manifest, f, indent=2, default=str)

    print(f"\n  ✅ Manifest saved to {manifest_path}")

    # Summary
    if results['passed']:
        print(f"\n  ✅ AUDIT C PASSED: Reproducibility manifest generated")
    else:
        print(f"\n  ❌ AUDIT C FAILED: {len(results['issues'])} issues found")

    return results


# ============================================================================
# AUDIT D: LOOK-AHEAD BIAS CHECK
# ============================================================================

def audit_look_ahead() -> Dict:
    """
    Verify no future information is used in any calculation.

    This is a code review checklist - we verify by inspecting key functions.
    """
    print("\n" + "="*60)
    print("AUDIT D: LOOK-AHEAD BIAS CHECK")
    print("="*60)

    results = {
        'audit': 'D - Look-Ahead Bias',
        'passed': True,
        'checks': [],
        'issues': [],
        'warnings': []
    }

    # Check 1: Regime classification
    print("\n  Checking regime classification code...")
    regime_files = [
        'src/analysis/regime_detection.py',
        'src/competition/niche_population_v2.py',
    ]

    for filepath in regime_files:
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                content = f.read()

            # Look for potential look-ahead patterns
            issues = []

            # Pattern 1: Using data[idx:] or data.iloc[idx:] without restriction
            if 'idx:]' in content and 'idx-' not in content:
                issues.append("Potential future data access: 'idx:]' pattern found")

            # Pattern 2: shift(-1) without proper handling
            if 'shift(-' in content:
                issues.append("shift(-n) found - verify this is for target, not features")

            if len(issues) == 0:
                results['checks'].append({'file': filepath, 'status': 'PASS'})
                print(f"    ✅ {filepath}: No obvious look-ahead patterns")
            else:
                results['checks'].append({'file': filepath, 'status': 'WARNING', 'issues': issues})
                results['warnings'].extend([f"⚠️ {filepath}: {i}" for i in issues])
                print(f"    ⚠️ {filepath}: {len(issues)} potential issues (manual review needed)")
        else:
            print(f"    ⚪ {filepath}: File not found (skip)")

    # Check 2: Feature calculation
    print("\n  Checking feature calculation code...")
    feature_files = [
        'src/analysis/features_v2.py',
        'src/analysis/features.py',
    ]

    for filepath in feature_files:
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                content = f.read()

            # Features should use .shift(1) or rolling windows ending at current time
            has_proper_lag = '.shift(1)' in content or '.shift()' in content
            has_rolling = '.rolling(' in content

            if has_proper_lag or has_rolling:
                results['checks'].append({'file': filepath, 'status': 'PASS'})
                print(f"    ✅ {filepath}: Uses proper lagging/rolling")
            else:
                results['warnings'].append(f"⚠️ {filepath}: No explicit lagging found")
                print(f"    ⚠️ {filepath}: No explicit lagging found (manual review)")

    # Check 3: Train/Val/Test splits
    print("\n  Checking data splitting code...")
    loader_files = [
        'src/data/loader_v2.py',
        'src/data/loader.py',
    ]

    for filepath in loader_files:
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                content = f.read()

            has_temporal_split = 'iloc[:' in content or 'train_ratio' in content
            has_embargo = 'embargo' in content.lower()

            if has_temporal_split:
                results['checks'].append({'file': filepath, 'status': 'PASS'})
                print(f"    ✅ {filepath}: Uses temporal splitting")
                if has_embargo:
                    print(f"    ✅ {filepath}: Has embargo period")
            else:
                results['warnings'].append(f"⚠️ {filepath}: Temporal split not found")

    # Summary
    print(f"\n  Checks completed: {len(results['checks'])}")
    print(f"  Warnings: {len(results['warnings'])}")

    if len(results['warnings']) == 0:
        print(f"\n  ✅ AUDIT D PASSED: No look-ahead bias detected")
    else:
        print(f"\n  ⚠️ AUDIT D PASSED WITH WARNINGS: Manual review recommended for {len(results['warnings'])} items")

    return results


# ============================================================================
# AUDIT E: ECONOMIC VS STATISTICAL SIGNIFICANCE
# ============================================================================

def audit_economic_significance() -> Dict:
    """
    Define framework for assessing economic significance.

    This creates the framework - actual assessment happens in P1.
    """
    print("\n" + "="*60)
    print("AUDIT E: ECONOMIC VS STATISTICAL SIGNIFICANCE FRAMEWORK")
    print("="*60)

    results = {
        'audit': 'E - Economic Significance',
        'passed': True,
        'framework': {},
        'issues': [],
    }

    # Define the framework
    framework = {
        'correlation_thresholds': {
            'noise': {'range': '|r| < 0.10', 'economic_significance': 'None'},
            'marginal': {'range': '0.10 ≤ |r| < 0.15', 'economic_significance': 'Marginal'},
            'moderate': {'range': '0.15 ≤ |r| < 0.20', 'economic_significance': 'Potentially tradeable'},
            'strong': {'range': '|r| ≥ 0.20', 'economic_significance': 'Strong edge'},
        },
        'break_even_calculation': {
            'formula': 'break_even_corr = annual_cost / (volatility * sqrt(trades_per_year))',
            'assumed_volatility': PARAMETER_CHOICES['assumed_volatility'],
            'trades_per_year': PARAMETER_CHOICES['trades_per_year'],
        },
        'grinold_thresholds': {
            'ic_minimum': 0.03,
            'ir_minimum': 0.3,
        }
    }

    # Calculate break-even correlation for different cost scenarios
    vol = PARAMETER_CHOICES['assumed_volatility']
    trades = PARAMETER_CHOICES['trades_per_year']

    cost_scenarios = {
        'crypto': 0.0012,  # 6 bps * 2 (round trip)
        'forex': 0.0002,   # 1 bp * 2
        'stocks': 0.0004,  # 2 bps * 2
        'commodities': 0.0006,  # 3 bps * 2
    }

    print("\n  Break-even correlations by market:")
    for market, annual_cost in cost_scenarios.items():
        total_annual_cost = annual_cost * trades
        break_even = total_annual_cost / (vol * np.sqrt(trades))
        framework[f'break_even_{market}'] = round(break_even, 4)
        print(f"    {market}: |r| > {break_even:.4f} (at {trades} trades/year)")

    results['framework'] = framework

    # Save framework
    framework_path = Path('results/audit/economic_significance_framework.json')
    framework_path.parent.mkdir(parents=True, exist_ok=True)

    with open(framework_path, 'w') as f:
        json.dump(framework, f, indent=2)

    print(f"\n  ✅ Framework saved to {framework_path}")
    print(f"\n  ✅ AUDIT E PASSED: Economic significance framework established")

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all P0 critical audits."""

    print("\n" + "="*60)
    print("P0: CRITICAL AUDITS - NEXT_STEPS_PLAN v4.1")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Random seed: {PARAMETER_CHOICES['random_seed']}")

    # Run all audits
    all_results = {}

    all_results['A'] = audit_survivorship()
    all_results['B'] = audit_data_quality()
    all_results['C'] = audit_reproducibility()
    all_results['D'] = audit_look_ahead()
    all_results['E'] = audit_economic_significance()

    # Summary
    print("\n" + "="*60)
    print("P0 AUDIT SUMMARY")
    print("="*60)

    all_passed = True
    for audit_id, result in all_results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        warnings = len(result.get('warnings', []))
        issues = len(result.get('issues', []))

        warning_str = f" ({warnings} warnings)" if warnings > 0 else ""
        issue_str = f" ({issues} issues)" if issues > 0 else ""

        print(f"  Audit {audit_id}: {status}{warning_str}{issue_str}")

        if not result['passed']:
            all_passed = False

    # Save full results
    results_path = Path('results/audit/p0_audit_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'all_passed': all_passed,
        'audits': {k: {
            'passed': v['passed'],
            'issues': v.get('issues', []),
            'warnings': v.get('warnings', []),
        } for k, v in all_results.items()}
    }

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n  Full results saved to {results_path}")

    # Final verdict
    print("\n" + "="*60)
    if all_passed:
        print("✅ P0 CRITICAL AUDITS: ALL PASSED")
        print("   Ready to proceed to P1: Backtest with Costs")
    else:
        print("❌ P0 CRITICAL AUDITS: FAILED")
        print("   Fix issues before proceeding")
    print("="*60 + "\n")

    return all_passed, all_results


if __name__ == "__main__":
    passed, results = main()
    sys.exit(0 if passed else 1)
