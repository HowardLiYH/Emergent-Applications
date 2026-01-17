#!/usr/bin/env python3
"""
Run SI dynamics pipeline: How should we use SI?
Pipeline 3: 9 SI dynamics features

Usage:
    python experiments/run_dynamics.py
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import json
from pathlib import Path


def compute_si_variants(si: pd.Series) -> pd.DataFrame:
    """Compute different SI variants."""
    variants = pd.DataFrame(index=si.index)

    variants['si_raw'] = si
    variants['si_1h'] = si.rolling(1).mean()
    variants['si_4h'] = si.rolling(4).mean()
    variants['si_1d'] = si.rolling(24).mean()
    variants['si_1w'] = si.rolling(168).mean()

    # Derivatives
    variants['dsi_dt'] = si.diff()
    variants['si_acceleration'] = variants['dsi_dt'].diff()

    # Stability
    variants['si_std'] = si.rolling(24).std()

    # Percentile
    variants['si_percentile'] = si.rolling(720).rank(pct=True)

    return variants


def main():
    print("=" * 60)
    print("PHASE 6: SI DYNAMICS PIPELINE")
    print("How should we use SI?")
    print("=" * 60)

    output_dir = Path("results/si_correlations")

    # 1. Load metadata
    print("\n1. Loading metadata...")
    try:
        with open(output_dir / "metadata.json") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("❌ metadata.json not found. Run run_backtest.py first!")
        return False

    symbols = metadata['symbols']
    all_results = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {symbol}")
        print('='*60)

        # 2. Load data
        try:
            si = pd.read_csv(output_dir / f"si_{symbol}_train.csv", index_col=0, parse_dates=True).squeeze()
            features = pd.read_csv(output_dir / f"features_{symbol}_train.csv", index_col=0, parse_dates=True)
        except FileNotFoundError as e:
            print(f"   ❌ {e}")
            continue

        # Use next_day_return as profit proxy
        if 'next_day_return' in features.columns:
            profit = features['next_day_return']
        else:
            profit = features.iloc[:, 0]

        # 3. Compute SI variants
        print("\n2. Computing SI variants...")
        variants = compute_si_variants(si)
        print(f"   Created {len(variants.columns)} variants")

        # 4. Compare variants
        print("\n3. Comparing variants with profit...")
        variant_results = {}

        for col in variants.columns:
            mask = ~(variants[col].isna() | profit.isna())
            if mask.sum() < 100:
                continue

            r, p = spearmanr(variants[col][mask], profit[mask])
            variant_results[col] = {'r': float(r), 'p': float(p)}

        # Sort by absolute correlation
        sorted_variants = sorted(variant_results.items(), key=lambda x: abs(x[1]['r']), reverse=True)

        print("\n   VARIANT PERFORMANCE:")
        for name, res in sorted_variants:
            print(f"   {name:20s} r={res['r']:+.3f}  p={res['p']:.4f}")

        best_variant = sorted_variants[0][0] if sorted_variants else None
        print(f"\n   BEST VARIANT: {best_variant}")

        # 5. Momentum analysis
        print("\n4. Momentum analysis (does dSI/dt matter?)...")
        dsi = variants['dsi_dt'].dropna()
        profit_aligned = profit.loc[dsi.index].dropna()
        common_idx = dsi.index.intersection(profit_aligned.index)
        dsi = dsi.loc[common_idx]
        profit_aligned = profit_aligned.loc[common_idx]

        rising_si = dsi > 0
        profit_rising = profit_aligned[rising_si].mean() if rising_si.sum() > 0 else np.nan
        profit_falling = profit_aligned[~rising_si].mean() if (~rising_si).sum() > 0 else np.nan

        print(f"   Profit when SI rising: {profit_rising:.4f}" if not np.isnan(profit_rising) else "   Profit when SI rising: N/A")
        print(f"   Profit when SI falling: {profit_falling:.4f}" if not np.isnan(profit_falling) else "   Profit when SI falling: N/A")

        momentum_helps = profit_rising > profit_falling if not (np.isnan(profit_rising) or np.isnan(profit_falling)) else None

        # 6. Stability analysis
        print("\n5. Stability analysis (does SI volatility matter?)...")
        si_std = variants['si_std'].dropna()
        profit_aligned = profit.loc[si_std.index].dropna()
        common_idx = si_std.index.intersection(profit_aligned.index)
        si_std = si_std.loc[common_idx]
        profit_aligned = profit_aligned.loc[common_idx]

        if len(si_std) > 0:
            stable = si_std < si_std.median()
            profit_stable = profit_aligned[stable].mean() if stable.sum() > 0 else np.nan
            profit_volatile = profit_aligned[~stable].mean() if (~stable).sum() > 0 else np.nan

            print(f"   Profit when SI stable: {profit_stable:.4f}" if not np.isnan(profit_stable) else "   Profit when SI stable: N/A")
            print(f"   Profit when SI volatile: {profit_volatile:.4f}" if not np.isnan(profit_volatile) else "   Profit when SI volatile: N/A")

            stability_helps = profit_stable > profit_volatile if not (np.isnan(profit_stable) or np.isnan(profit_volatile)) else None
        else:
            stability_helps = None

        # Save results
        all_results[symbol] = {
            'best_variant': best_variant,
            'variant_correlations': variant_results,
            'momentum_effect': {
                'profit_when_rising': float(profit_rising) if not np.isnan(profit_rising) else None,
                'profit_when_falling': float(profit_falling) if not np.isnan(profit_falling) else None,
                'momentum_helps': bool(momentum_helps) if momentum_helps is not None else None
            },
            'stability_effect': {
                'stability_helps': bool(stability_helps) if stability_helps is not None else None
            }
        }

    # Save all results
    with open(output_dir / "dynamics_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("SI DYNAMICS COMPLETE!")
    for symbol, res in all_results.items():
        print(f"\n{symbol}:")
        print(f"  Best SI variant: {res['best_variant']}")
        print(f"  Momentum helps: {res['momentum_effect']['momentum_helps']}")
        print(f"  Stability helps: {res['stability_effect']['stability_helps']}")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
