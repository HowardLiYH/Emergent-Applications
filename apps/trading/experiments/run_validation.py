#!/usr/bin/env python3
"""
Phase 7: Validate discovery findings on validation set.
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
    print("PHASE 7: VALIDATION")
    print("Validating discovery findings on VAL set")
    print("=" * 60)

    output_dir = Path("results/si_correlations")

    # 1. Load candidates from discovery
    print("\n1. Loading discovery results...")
    discovery_files = list(output_dir.glob("discovery_*.csv"))

    with open(output_dir / "metadata.json") as f:
        metadata = json.load(f)

    symbols = metadata.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])

    # Get significant features from discovery
    all_significant = set()
    for symbol in symbols:
        disc_file = output_dir / f"discovery_{symbol}.csv"
        if disc_file.exists():
            df = pd.read_csv(disc_file)
            sig = df[df['significant'] == True]['feature'].tolist()
            all_significant.update(sig)

    candidates = list(all_significant)
    print(f"   Candidates to validate: {candidates}")

    if len(candidates) == 0:
        print("   No candidates to validate!")
        return False

    all_results = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"VALIDATING: {symbol}")
        print("=" * 60)

        # 2. Load data
        loader = DataLoader()
        try:
            data = loader.load(symbol)
        except FileNotFoundError:
            print(f"   Data not found for {symbol}")
            continue

        # 3. Split and get VAL set
        _, val, _ = loader.temporal_split(data)

        if len(val) < 500:
            print(f"   VAL set too small: {len(val)} rows")
            continue

        # 4. Run competition on VAL
        print(f"\n2. Running competition on VAL set ({len(val)} rows)...")
        population = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=3)
        population.run(val, start_idx=200)
        si = population.compute_si_timeseries(val, window=168)

        print(f"   SI computed: mean={si.mean():.3f}, std={si.std():.3f}")

        # 5. Compute features on VAL
        print("\n3. Computing features on VAL...")
        calc = FeatureCalculator()
        features = calc.compute_all(val)

        # 6. Align
        common_idx = si.index.intersection(features.index)
        si = si.loc[common_idx]
        features = features.loc[common_idx]
        print(f"   Aligned {len(common_idx)} rows")

        # 7. Load train discovery results for this symbol
        train_disc_file = output_dir / f"discovery_{symbol}.csv"
        if train_disc_file.exists():
            train_results = pd.read_csv(train_disc_file)
        else:
            print(f"   No train discovery results for {symbol}")
            continue

        # 8. Validate candidates
        print("\n4. Validating candidates...")
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

            # Compute val correlation
            mask = ~(si.isna() | features[feature].isna())
            if mask.sum() < 50:
                continue

            val_r, val_p = spearmanr(si[mask], features[feature][mask])

            # Confirm: same direction AND p < 0.05
            same_direction = (val_r * train_r) > 0
            significant = val_p < 0.05

            status = "✅ CONFIRMED" if (same_direction and significant) else "❌ NOT CONFIRMED"
            print(f"   {feature}: r={val_r:.3f} (train: {train_r:.3f}) {status}")

            results.append({
                'feature': feature,
                'train_r': float(train_r),
                'val_r': float(val_r),
                'val_p': float(val_p),
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

    # 9. Save results
    print("\n" + "=" * 60)
    print("SAVING VALIDATION RESULTS")
    print("=" * 60)

    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # 10. Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    total_confirmed = []
    for symbol, res in all_results.items():
        print(f"\n{symbol}:")
        print(f"  Candidates tested: {len(res['candidates_tested'])}")
        print(f"  Confirmed: {res['n_confirmed']}")
        print(f"  Confirmation rate: {res['confirmation_rate']:.0%}")
        print(f"  Confirmed features: {res['confirmed']}")
        total_confirmed.extend(res['confirmed'])

    # Features confirmed in multiple assets
    from collections import Counter
    confirmed_counts = Counter(total_confirmed)
    cross_asset = [f for f, c in confirmed_counts.items() if c >= 2]

    print("\n" + "=" * 60)
    print("CROSS-ASSET CONFIRMED FEATURES")
    print("=" * 60)
    if cross_asset:
        print(f"Features confirmed in ≥2 assets: {cross_asset}")
    else:
        print("No features confirmed in multiple assets")

    print("\n✅ VALIDATION COMPLETE!")
    print(f"Results saved to: {output_dir / 'validation_results.json'}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
