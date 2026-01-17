"""
Data validation - MUST RUN before any analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path


class DataValidator:
    """Validate data quality before analysis."""

    def __init__(self, strict: bool = True):
        self.strict = strict  # If True, raises errors. If False, just warns.

    def validate(self, df: pd.DataFrame, symbol: str = "unknown") -> Dict:
        """
        Run all validation checks.

        Returns dict with:
            - valid: bool (True if all checks pass)
            - issues: list of issues found
            - stats: data statistics
        """
        issues = []
        warnings = []

        print(f"\n{'='*60}")
        print(f"VALIDATING: {symbol}")
        print('='*60)

        # 1. Check for missing values
        print("\n1. Checking missing values...")
        missing_pct = df.isna().mean() * 100
        for col, pct in missing_pct.items():
            if pct > 5:
                issues.append(f"Column '{col}' has {pct:.1f}% missing values")
            elif pct > 0:
                warnings.append(f"Column '{col}' has {pct:.1f}% missing values")

        if missing_pct.sum() == 0:
            print("   ✅ No missing values")
        else:
            print(f"   ⚠️  Missing values found: {missing_pct[missing_pct > 0].to_dict()}")

        # 2. Check for duplicates
        print("\n2. Checking for duplicate timestamps...")
        n_dups = df.index.duplicated().sum()
        if n_dups > 0:
            issues.append(f"Found {n_dups} duplicate timestamps")
            print(f"   ❌ Found {n_dups} duplicates")
        else:
            print("   ✅ No duplicates")

        # 3. Check for gaps
        print("\n3. Checking for data gaps...")
        if len(df) > 1:
            expected_freq = pd.Timedelta('1h')
            actual_gaps = df.index.to_series().diff()
            large_gaps = actual_gaps[actual_gaps > expected_freq * 2]

            if len(large_gaps) > 0:
                max_gap = large_gaps.max()
                warnings.append(f"Found {len(large_gaps)} gaps > 2 hours (max: {max_gap})")
                print(f"   ⚠️  Found {len(large_gaps)} gaps (max: {max_gap})")
            else:
                print("   ✅ No significant gaps")

        # 4. Check for extreme returns
        print("\n4. Checking for extreme returns...")
        returns = df['close'].pct_change()
        extreme_threshold = 0.5  # 50% in 1 hour
        extreme = (abs(returns) > extreme_threshold).sum()

        if extreme > 0:
            warnings.append(f"Found {extreme} extreme returns (>{extreme_threshold*100}% in 1h)")
            print(f"   ⚠️  Found {extreme} extreme returns")

            # Show the extremes
            extreme_rows = returns[abs(returns) > extreme_threshold]
            for ts, ret in extreme_rows.items():
                print(f"      {ts}: {ret*100:+.1f}%")
        else:
            print("   ✅ No extreme returns")

        # 5. Check data range
        print("\n5. Checking data range...")
        days = (df.index.max() - df.index.min()).days

        if days < 180:
            issues.append(f"Only {days} days of data (recommend 365+)")
            print(f"   ❌ Only {days} days (recommend 365+)")
        elif days < 365:
            warnings.append(f"Only {days} days of data (365+ preferred)")
            print(f"   ⚠️  {days} days (365+ preferred)")
        else:
            print(f"   ✅ {days} days of data")

        # 6. Check for negative prices
        print("\n6. Checking for invalid prices...")
        neg_prices = (df[['open', 'high', 'low', 'close']] <= 0).sum().sum()
        if neg_prices > 0:
            issues.append(f"Found {neg_prices} non-positive prices")
            print(f"   ❌ Found {neg_prices} non-positive prices")
        else:
            print("   ✅ All prices valid")

        # 7. Check OHLC consistency
        print("\n7. Checking OHLC consistency...")
        ohlc_issues = (
            (df['high'] < df['low']).sum() +
            (df['high'] < df['open']).sum() +
            (df['high'] < df['close']).sum() +
            (df['low'] > df['open']).sum() +
            (df['low'] > df['close']).sum()
        )
        if ohlc_issues > 0:
            issues.append(f"Found {ohlc_issues} OHLC inconsistencies")
            print(f"   ❌ Found {ohlc_issues} OHLC inconsistencies")
        else:
            print("   ✅ OHLC consistent")

        # Summary
        print(f"\n{'='*60}")
        valid = len(issues) == 0

        if valid and len(warnings) == 0:
            print("✅ VALIDATION PASSED - Data is clean!")
        elif valid:
            print(f"⚠️  VALIDATION PASSED with {len(warnings)} warnings")
        else:
            print(f"❌ VALIDATION FAILED - {len(issues)} critical issues")
            for issue in issues:
                print(f"   - {issue}")

        # Stats
        stats = {
            'n_rows': len(df),
            'date_start': str(df.index.min()),
            'date_end': str(df.index.max()),
            'days': days,
            'price_min': float(df['close'].min()),
            'price_max': float(df['close'].max()),
            'return_mean': float(returns.mean()),
            'return_std': float(returns.std()),
        }

        result = {
            'valid': valid,
            'issues': issues,
            'warnings': warnings,
            'stats': stats,
        }

        # Raise error if strict mode and issues found
        if self.strict and not valid:
            raise ValueError(
                f"Data validation failed for {symbol}:\n" +
                "\n".join(f"  - {i}" for i in issues)
            )

        return result


def validate_all_markets(markets: Dict[str, List[str]]) -> Dict:
    """
    Validate data for multiple markets and assets.

    Args:
        markets: Dict like {'crypto': ['BTCUSDT', 'ETHUSDT'], 'forex': ['EURUSD']}

    Returns:
        Dict with validation results for each asset
    """
    from .loader import MultiMarketLoader, MarketType

    results = {}
    validator = DataValidator(strict=False)

    for market_name, assets in markets.items():
        market_type = MarketType(market_name)
        loader = MultiMarketLoader(market_type)

        for asset in assets:
            try:
                df = loader.load(asset)
                result = validator.validate(df, f"{market_name}/{asset}")
                results[f"{market_name}/{asset}"] = result
            except FileNotFoundError as e:
                results[f"{market_name}/{asset}"] = {
                    'valid': False,
                    'issues': [str(e)],
                    'warnings': [],
                    'stats': None
                }

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    valid_count = sum(1 for r in results.values() if r['valid'])
    print(f"\nPassed: {valid_count}/{len(results)}")

    for asset, result in results.items():
        status = "✅" if result['valid'] else "❌"
        print(f"  {status} {asset}")

    return results
