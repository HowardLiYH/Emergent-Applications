#!/usr/bin/env python3
"""
Run backtest and compute SI for multiple assets.
ALWAYS multi-asset - NEVER single-coin!

Usage:
    python experiments/run_backtest.py
"""
import sys
sys.path.insert(0, '.')

from src.data.loader import DataLoader, MultiMarketLoader, MarketType
from src.analysis.features import FeatureCalculator
from src.agents.strategies import DEFAULT_STRATEGIES
from src.competition.niche_population import NichePopulation
from src.utils.reproducibility import set_all_seeds, create_manifest, save_manifest
import pandas as pd
import json
from pathlib import Path


def run_single_asset(symbol: str, data_dir: str = "data/crypto") -> dict:
    """Run backtest for a single asset and return results."""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {symbol}")
    print('='*60)

    # 1. Load data
    print("\n1. Loading data...")
    loader = DataLoader(data_dir)
    try:
        data = loader.load(symbol)
    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        return None

    print(f"   Loaded {len(data)} rows")

    # 2. Split data
    print("\n2. Splitting data...")
    train, val, test = loader.temporal_split(data)

    # 3. Run competition on TRAIN only
    print("\n3. Running competition on TRAIN set...")
    population = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=3)
    history = population.run(train, start_idx=200)
    print(f"   Completed {len(history)} competition rounds")

    # 4. Compute SI timeseries
    print("\n4. Computing SI timeseries...")
    si = population.compute_si_timeseries(train, window=168)
    print(f"   SI mean: {si.mean():.3f}, std: {si.std():.3f}")

    # 5. Compute features
    print("\n5. Computing features...")
    calc = FeatureCalculator()
    features = calc.compute_all(train)
    print(f"   Computed {len(features.columns)} features")

    # 6. Align SI and features
    print("\n6. Aligning data...")
    common_idx = si.index.intersection(features.index)
    si_aligned = si.loc[common_idx]
    features_aligned = features.loc[common_idx]
    print(f"   Aligned {len(common_idx)} rows")

    return {
        'symbol': symbol,
        'si': si_aligned,
        'features': features_aligned,
        'discovery_features': calc.get_discovery_features(),
        'prediction_features': calc.get_prediction_features(),
        'train_data': train,
        'val_data': val,
        'test_data': test,
        'population': population,
    }


def main():
    print("=" * 60)
    print("PHASE 3: BACKTEST & SI COMPUTATION")
    print("⚠️  MULTI-ASSET MODE - Never single-coin!")
    print("=" * 60)

    # Set seeds for reproducibility
    set_all_seeds(42)

    # Multi-asset: BTC, ETH, SOL minimum
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

    results = {}
    for symbol in symbols:
        result = run_single_asset(symbol)
        if result is not None:
            results[symbol] = result

    if len(results) == 0:
        print("\n❌ NO DATA AVAILABLE - Please download data first!")
        print("   Run: python experiments/download_data.py")
        return False

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    output_dir = Path("results/si_correlations")
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol, result in results.items():
        result['si'].to_csv(output_dir / f"si_{symbol}_train.csv")
        result['features'].to_csv(output_dir / f"features_{symbol}_train.csv")

    # Save metadata
    metadata = {
        'symbols': list(results.keys()),
        'n_assets': len(results),
        'stats': {
            symbol: {
                'n_rows': len(result['si']),
                'si_mean': float(result['si'].mean()),
                'si_std': float(result['si'].std()),
                'n_features': len(result['features'].columns),
                'discovery_features': result['discovery_features'],
                'prediction_features': result['prediction_features'],
            }
            for symbol, result in results.items()
        }
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save manifest for reproducibility
    manifest = create_manifest(config={'symbols': symbols, 'seed': 42})
    save_manifest(manifest, str(output_dir / "manifest.json"))

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE!")
    print(f"Processed {len(results)} assets: {list(results.keys())}")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
