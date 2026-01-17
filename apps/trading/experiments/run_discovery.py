#!/usr/bin/env python3
"""
Run discovery pipeline: What does SI correlate with?
Pipeline 1: 46 discovery features (no lookahead, no circular)

Usage:
    python experiments/run_discovery.py
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import json
from pathlib import Path
from src.analysis.correlations import CorrelationAnalyzer


def main():
    print("=" * 60)
    print("PHASE 4: DISCOVERY PIPELINE")
    print("What does SI correlate with?")
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
    print(f"   Found {len(symbols)} assets: {symbols}")

    all_results = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {symbol}")
        print('='*60)

        # 2. Load SI and features
        print("\n2. Loading SI and features...")
        try:
            si = pd.read_csv(output_dir / f"si_{symbol}_train.csv", index_col=0, parse_dates=True).squeeze()
            features = pd.read_csv(output_dir / f"features_{symbol}_train.csv", index_col=0, parse_dates=True)
        except FileNotFoundError as e:
            print(f"   ❌ {e}")
            continue

        discovery_features = metadata['stats'][symbol]['discovery_features']
        print(f"   SI: {len(si)} rows")
        print(f"   Features: {len(discovery_features)} discovery features")

        # 3. Run correlation analysis
        print("\n3. Running correlation analysis...")
        analyzer = CorrelationAnalyzer()  # Auto block size
        results = analyzer.run_discovery(si, features, discovery_features)

        # 4. Report results
        print("\n4. Results:")
        print("-" * 60)

        significant = results[results['significant']]
        print(f"\n   SIGNIFICANT CORRELATIONS ({len(significant)}):")
        if len(significant) > 0:
            print(significant[['feature', 'r', 'p_fdr', 'ci_low', 'ci_high']].to_string(index=False))
        else:
            print("   None found")

        print(f"\n   TOP 10 CORRELATIONS (by |r|):")
        print(results.head(10)[['feature', 'r', 'p_fdr', 'significant']].to_string(index=False))

        # 5. Save results
        results.to_csv(output_dir / f"discovery_{symbol}.csv", index=False)

        all_results[symbol] = {
            'n_features_tested': len(results),
            'n_significant': len(significant),
            'top_correlate': results.iloc[0]['feature'] if len(results) > 0 else None,
            'top_r': float(results.iloc[0]['r']) if len(results) > 0 else None,
            'significant_features': significant['feature'].tolist() if len(significant) > 0 else [],
        }

    # 6. Cross-asset summary
    print("\n" + "=" * 60)
    print("CROSS-ASSET SUMMARY")
    print("=" * 60)

    # Find features significant in multiple assets
    all_significant = []
    for symbol, res in all_results.items():
        all_significant.extend(res.get('significant_features', []))

    from collections import Counter
    feature_counts = Counter(all_significant)

    print("\nFeatures significant in multiple assets:")
    for feature, count in feature_counts.most_common(10):
        if count >= 2:
            print(f"   {feature}: {count} assets")

    # Save summary
    with open(output_dir / "discovery_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("DISCOVERY COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    # Decision point
    total_significant = sum(res['n_significant'] for res in all_results.values())
    if total_significant >= 3:
        print("\n✅ SUCCESS: Found 3+ significant correlations!")
        print("   NEXT: python experiments/run_prediction.py")
    else:
        print("\n⚠️  CAUTION: Found <3 significant correlations")
        print("   CONSIDER: Check data quality, try different SI variants")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
