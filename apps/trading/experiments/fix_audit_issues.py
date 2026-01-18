#!/usr/bin/env python3
"""
FIX AUDIT ISSUES

Issues found:
1. EURUSD: 30 OHLC inconsistencies
2. GBPUSD: 30 OHLC inconsistencies
3. No random seed setting

This script diagnoses and fixes these issues.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================
# ISSUE 1 & 2: OHLC INCONSISTENCIES
# ============================================================

def diagnose_ohlc_issues():
    """Diagnose OHLC inconsistencies in Forex data."""
    print("\n" + "="*70)
    print("  DIAGNOSING OHLC ISSUES")
    print("="*70)

    from src.data.loader_v2 import DataLoaderV2, MarketType
    loader = DataLoaderV2()

    for symbol in ['EURUSD', 'GBPUSD']:
        print(f"\n  {symbol}:")
        data = loader.load(symbol, MarketType.FOREX)

        # Check each type of inconsistency
        high_lt_low = data['high'] < data['low']
        high_lt_open = data['high'] < data['open']
        high_lt_close = data['high'] < data['close']
        low_gt_open = data['low'] > data['open']
        low_gt_close = data['low'] > data['close']

        issues = {
            'high < low': high_lt_low.sum(),
            'high < open': high_lt_open.sum(),
            'high < close': high_lt_close.sum(),
            'low > open': low_gt_open.sum(),
            'low > close': low_gt_close.sum(),
        }

        for issue_type, count in issues.items():
            if count > 0:
                print(f"    {issue_type}: {count} rows")

        # Show sample of problematic rows
        any_issue = high_lt_low | high_lt_open | high_lt_close | low_gt_open | low_gt_close
        problem_rows = data[any_issue]
        if len(problem_rows) > 0:
            print(f"\n    Sample problematic rows:")
            print(problem_rows[['open', 'high', 'low', 'close']].head(3).to_string())

            # Check if it's a precision issue
            print(f"\n    Checking if precision issue...")
            max_diff = (data['high'] - data['low']).abs().max()
            min_diff = (data['high'] - data['low']).abs().min()
            print(f"    High-Low range: min={min_diff:.6f}, max={max_diff:.6f}")


def fix_ohlc_issues():
    """Fix OHLC inconsistencies in the data loader."""
    print("\n" + "="*70)
    print("  FIXING OHLC ISSUES")
    print("="*70)

    # The fix: Ensure OHLC consistency in the data loader
    # High = max(open, high, low, close)
    # Low = min(open, high, low, close)

    loader_path = Path("src/data/loader_v2.py")

    if not loader_path.exists():
        print("  ❌ loader_v2.py not found")
        return False

    with open(loader_path, 'r') as f:
        content = f.read()

    # Check if fix already applied
    if 'ensure_ohlc_consistency' in content:
        print("  ✅ OHLC fix already applied")
        return True

    # Add the fix function after imports
    fix_code = '''

def ensure_ohlc_consistency(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure OHLC data is consistent:
    - High >= max(open, close)
    - Low <= min(open, close)
    - High >= Low
    """
    df = data.copy()

    # Fix High: should be max of all
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)

    # Fix Low: should be min of all
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

    return df

'''

    # Find where to insert (after imports, before class definition)
    import_end = content.rfind('import ')
    import_end = content.find('\n', import_end) + 1

    # Find next blank line after imports
    while content[import_end:import_end+2] != '\n\n':
        import_end = content.find('\n', import_end) + 1

    new_content = content[:import_end] + fix_code + content[import_end:]

    # Also add call to ensure_ohlc_consistency in the load method
    # Find the load method and add the fix before return
    if 'return data' in new_content:
        new_content = new_content.replace(
            'return data',
            'data = ensure_ohlc_consistency(data)\n        return data'
        )

    with open(loader_path, 'w') as f:
        f.write(new_content)

    print("  ✅ Added ensure_ohlc_consistency function to loader_v2.py")
    return True


# ============================================================
# ISSUE 3: RANDOM SEED
# ============================================================

def add_random_seed():
    """Add random seed setting to ensure reproducibility."""
    print("\n" + "="*70)
    print("  ADDING RANDOM SEED")
    print("="*70)

    v2_path = Path("experiments/test_all_applications_v2.py")

    if not v2_path.exists():
        print("  ❌ test_all_applications_v2.py not found")
        return False

    with open(v2_path, 'r') as f:
        content = f.read()

    # Check if already has seed
    if 'np.random.seed' in content or 'RANDOM_SEED' in content:
        print("  ✅ Random seed already set")
        return True

    # Add seed after imports
    seed_code = '''
# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
'''

    # Find where to insert (after the last import)
    insert_pos = content.find('# ============================================================')
    if insert_pos == -1:
        insert_pos = content.find('\n\n', content.rfind('import '))

    new_content = content[:insert_pos] + seed_code + '\n' + content[insert_pos:]

    with open(v2_path, 'w') as f:
        f.write(new_content)

    print("  ✅ Added RANDOM_SEED = 42 to test_all_applications_v2.py")
    return True


# ============================================================
# VERIFY FIXES
# ============================================================

def verify_fixes():
    """Verify that fixes were applied correctly."""
    print("\n" + "="*70)
    print("  VERIFYING FIXES")
    print("="*70)

    # Reload and check OHLC
    from importlib import reload
    import src.data.loader_v2 as loader_module
    reload(loader_module)

    from src.data.loader_v2 import DataLoaderV2, MarketType
    loader = DataLoaderV2()

    all_fixed = True

    for symbol in ['EURUSD', 'GBPUSD']:
        data = loader.load(symbol, MarketType.FOREX)

        # Check OHLC consistency
        high_lt_low = (data['high'] < data['low']).sum()
        high_lt_open = (data['high'] < data['open']).sum()
        high_lt_close = (data['high'] < data['close']).sum()
        low_gt_open = (data['low'] > data['open']).sum()
        low_gt_close = (data['low'] > data['close']).sum()

        total_issues = high_lt_low + high_lt_open + high_lt_close + low_gt_open + low_gt_close

        if total_issues == 0:
            print(f"  ✅ {symbol}: OHLC fixed (0 issues)")
        else:
            print(f"  ❌ {symbol}: Still has {total_issues} OHLC issues")
            all_fixed = False

    # Check random seed
    v2_path = Path("experiments/test_all_applications_v2.py")
    with open(v2_path, 'r') as f:
        content = f.read()

    if 'RANDOM_SEED' in content:
        print("  ✅ Random seed: Set")
    else:
        print("  ❌ Random seed: Not set")
        all_fixed = False

    return all_fixed


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*70)
    print("  FIXING AUDIT ISSUES")
    print("="*70)
    print(f"  Time: {datetime.now().isoformat()}")

    # Step 1: Diagnose
    diagnose_ohlc_issues()

    # Step 2: Fix OHLC
    fix_ohlc_issues()

    # Step 3: Add random seed
    add_random_seed()

    # Step 4: Verify
    all_fixed = verify_fixes()

    print("\n" + "="*70)
    if all_fixed:
        print("  ✅ ALL ISSUES FIXED")
    else:
        print("  ⚠️ SOME ISSUES REMAIN")
    print("="*70 + "\n")

    return all_fixed

if __name__ == "__main__":
    main()
