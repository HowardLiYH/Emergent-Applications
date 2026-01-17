#!/usr/bin/env python3
"""
Final holdout TEST set validation.
This is the ultimate test - features must confirm here to be trusted.
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr

from src.data.loader import DataLoader
from src.analysis.features import FeatureCalculator
from src.agents.strategies import DEFAULT_STRATEGIES
from src.competition.niche_population import NichePopulation


def main():
    print("=" * 60)
    print("FINAL TEST SET VALIDATION")
    print("⚠️  This is the holdout test - no re-running allowed!")
    print("=" * 60)

    output_dir = Path("results/si_correlations")

    # 1. Load confirmed features from validation
    print("\n1. Loading validation results...")
    with open(output_dir / "validation_results.json") as f:
        validation = json.load(f)

    # Get features confirmed in ≥2 assets
    from collections import Counter
    all_confirmed = []
    for symbol, res in validation.items():
        all_confirmed.extend(res.get('confirmed', []))

    confirmed_counts = Counter(all_confirmed)
    candidates = [f for f, c in confirmed_counts.items() if c >= 2]

    print(f"   Candidates (confirmed in ≥2 assets): {candidates}")

    if len(candidates) == 0:
        print("   ❌ No candidates to test!")
        return False

    # 2. Load metadata
    with open(output_dir / "metadata.json") as f:
        metadata = json.load(f)

    symbols = metadata.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])

    all_results = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"TESTING: {symbol}")
        print("=" * 60)

        # 3. Load data
        loader = DataLoader()
        try:
            data = loader.load(symbol)
        except FileNotFoundError:
            print(f"   Data not found for {symbol}")
            continue

        # 4. Split and get TEST set
        _, _, test = loader.temporal_split(data)

        if len(test) < 500:
            print(f"   TEST set too small: {len(test)} rows")
            continue

        print(f"   TEST set: {len(test)} rows")

        # 5. Run competition on TEST
        print("\n2. Running competition on TEST set...")
        population = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=3)
        population.run(test, start_idx=200)
        si = population.compute_si_timeseries(test, window=168)

        print(f"   SI computed: mean={si.mean():.3f}, std={si.std():.3f}")

        # 6. Compute features on TEST
        print("\n3. Computing features on TEST...")
        calc = FeatureCalculator()
        features = calc.compute_all(test)

        # 7. Align
        common_idx = si.index.intersection(features.index)
        si = si.loc[common_idx]
        features = features.loc[common_idx]
        print(f"   Aligned {len(common_idx)} rows")

        # 8. Load train discovery for comparison
        train_disc_file = output_dir / f"discovery_{symbol}.csv"
        if train_disc_file.exists():
            train_results = pd.read_csv(train_disc_file)
        else:
            print(f"   No train discovery results for {symbol}")
            continue

        # 9. Test candidates
        print("\n4. Testing candidates on TEST set...")
        results = []
        confirmed = []

        for feature in candidates:
            if feature not in features.columns:
                continue

            # Get train correlation
            train_row = train_results[train_results['feature'] == feature]
            if len(train_row) == 0:
                continue
            train_r = train_row['r'].values[0]

            # Compute test correlation
            mask = ~(si.isna() | features[feature].isna())
            if mask.sum() < 50:
                continue

            test_r, test_p = spearmanr(si[mask], features[feature][mask])

            # Confirm: same direction AND p < 0.05
            same_direction = (test_r * train_r) > 0
            significant = test_p < 0.05

            status = "✅ CONFIRMED" if (same_direction and significant) else "❌ FAILED"
            print(f"   {feature}: test_r={test_r:.3f} (train: {train_r:.3f}) {status}")

            results.append({
                'feature': feature,
                'train_r': float(train_r),
                'test_r': float(test_r),
                'test_p': float(test_p),
                'same_direction': bool(same_direction),
                'significant': bool(significant),
                'confirmed': bool(same_direction and significant)
            })

            if same_direction and significant:
                confirmed.append(feature)

        all_results[symbol] = {
            'candidates_tested': candidates,
            'n_confirmed': len(confirmed),
            'confirmed': confirmed,
            'confirmation_rate': len(confirmed) / len(candidates) if candidates else 0,
            'details': results
        }

    # 10. Save results
    print("\n" + "=" * 60)
    print("SAVING TEST RESULTS")
    print("=" * 60)

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # 11. Final Summary
    print("\n" + "=" * 60)
    print("🎯 FINAL TEST RESULTS")
    print("=" * 60)

    total_confirmed = []
    for symbol, res in all_results.items():
        print(f"\n{symbol}:")
        print(f"  Candidates tested: {len(res['candidates_tested'])}")
        print(f"  Confirmed: {res['n_confirmed']}")
        print(f"  Confirmation rate: {res['confirmation_rate']:.0%}")
        print(f"  Confirmed features: {res['confirmed']}")
        total_confirmed.extend(res['confirmed'])

    # Features confirmed across all three sets (train → val → test)
    final_counts = Counter(total_confirmed)
    universal = [f for f, c in final_counts.items() if c >= 2]

    print("\n" + "=" * 60)
    print("🏆 UNIVERSALLY CONFIRMED FEATURES")
    print("(Confirmed in train → val → test across ≥2 assets)")
    print("=" * 60)

    if universal:
        for f in universal:
            print(f"  ✅ {f} (confirmed in {final_counts[f]} assets)")
        print(f"\n🎉 SUCCESS: {len(universal)} features passed all validation stages!")
    else:
        print("  ⚠️  No features universally confirmed across all stages")

    print(f"\nResults saved to: {output_dir / 'test_results.json'}")

    return len(universal) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
