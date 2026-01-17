#!/usr/bin/env python3
"""
CORRECTED ANALYSIS - Frequency-Aware Parameters

This script fixes the critical issue where hourly parameters were applied to daily data.
All components now use frequency-aware windows/lookbacks.

Fixes:
1. ✅ Feature windows: 1d/7d/30d correctly mapped for each frequency
2. ✅ SI window: 7 for daily, 168 for hourly
3. ✅ Validation threshold: 30 for daily, 300 for hourly
4. ✅ Regime classification: Frequency-aware lookback
5. ✅ Strategy lookbacks: Frequency-aware
6. ✅ FDR correction across all tests
7. ✅ Realistic transaction costs
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2
from src.analysis.features_v2 import FeatureCalculatorV2
from src.data.loader_v2 import DataLoaderV2, MarketType, TRANSACTION_COSTS


# ============================================================
# FREQUENCY-AWARE CONFIGURATION
# ============================================================

# Minimum validation size by frequency
MIN_VAL_SIZE = {
    'hourly': 300,   # 300 hours ~12 days
    'daily': 30,     # 30 days
}

# SI windows by frequency
SI_WINDOWS = {
    'hourly': 168,   # 7 days
    'daily': 7,      # 7 days
}

# Start index by frequency
START_IDX = {
    'hourly': 200,
    'daily': 15,     # Need ~2 weeks warm-up for regime classification
}

# Markets and assets
MARKETS = {
    MarketType.CRYPTO: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    MarketType.FOREX: ['EURUSD', 'GBPUSD', 'USDJPY'],
    MarketType.STOCKS: ['SPY', 'QQQ', 'AAPL'],
    MarketType.COMMODITIES: ['GOLD', 'OIL'],
}


def apply_fdr_correction(pvalues: list) -> list:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return []

    sorted_idx = np.argsort(pvalues)
    sorted_pvals = np.array(pvalues)[sorted_idx]

    bh_adjusted = np.zeros(n)
    for i in range(n):
        bh_adjusted[i] = sorted_pvals[i] * n / (i + 1)

    for i in range(n-2, -1, -1):
        bh_adjusted[i] = min(bh_adjusted[i], bh_adjusted[i+1])

    adjusted = np.zeros(n)
    for i, idx in enumerate(sorted_idx):
        adjusted[idx] = min(bh_adjusted[i], 1.0)

    return adjusted.tolist()


def analyze_asset(
    symbol: str,
    market_type: MarketType,
    loader: DataLoaderV2,
) -> dict:
    """Analyze a single asset with frequency-aware parameters."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {symbol} ({market_type.value})")
    print("=" * 60)

    try:
        # 1. Load data and detect frequency
        data = loader.load(symbol, market_type)
        frequency = loader.get_frequency(symbol, market_type)

        si_window = SI_WINDOWS[frequency]
        min_val = MIN_VAL_SIZE[frequency]
        start_idx = START_IDX[frequency]

        print(f"   Loaded {len(data)} rows ({frequency})")
        print(f"   Period: {data.index[0]} to {data.index[-1]}")
        print(f"   SI window: {si_window} bars ({frequency})")

        # 2. Split with purging
        train, val, test = loader.temporal_split_with_purging(
            data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            frequency=frequency
        )

        if len(train) < 100:
            print(f"   ❌ Insufficient training data: {len(train)}")
            return {'status': 'insufficient_data'}

        # 3. Get frequency-aware strategies
        strategies = get_default_strategies(frequency)
        print(f"   Using {len(strategies)} {frequency} strategies")

        # 4. Run competition on TRAIN
        print(f"\n   Running competition on TRAIN...")
        population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency=frequency)

        actual_start = min(start_idx, len(train) // 4)
        population.run(train, start_idx=actual_start)

        si_train = population.compute_si_timeseries(train, window=si_window)
        print(f"   SI computed: mean={si_train.mean():.3f}, std={si_train.std():.3f}")

        # 5. Compute features on TRAIN with correct frequency
        print(f"   Computing features ({frequency} windows)...")
        calc = FeatureCalculatorV2(frequency=frequency)
        features = calc.compute_all(train)

        # 6. Align
        common_idx = si_train.index.intersection(features.index)
        si_aligned = si_train.loc[common_idx]
        features_aligned = features.loc[common_idx]
        print(f"   Aligned {len(common_idx)} rows")

        if len(common_idx) < 50:
            print(f"   ❌ Insufficient aligned data: {len(common_idx)}")
            return {'status': 'insufficient_aligned'}

        # 7. Compute correlations
        discovery_features = calc.get_discovery_features()
        correlations = []
        pvalues = []

        for col in discovery_features:
            if col not in features_aligned.columns:
                continue
            mask = ~(si_aligned.isna() | features_aligned[col].isna())
            if mask.sum() < 30:
                continue

            r, p = spearmanr(si_aligned[mask], features_aligned[col][mask])
            if np.isnan(r):
                continue

            correlations.append({
                'feature': col,
                'r': float(r),
                'p': float(p),
                'n': int(mask.sum())
            })
            pvalues.append(p)

        # 8. Apply FDR correction
        if pvalues:
            q_values = apply_fdr_correction(pvalues)
            for i, corr in enumerate(correlations):
                corr['q'] = q_values[i]
                corr['significant_fdr'] = q_values[i] < 0.05
                corr['meaningful'] = abs(corr['r']) > 0.1 and q_values[i] < 0.05

        # 9. Count meaningful correlations
        meaningful = [c for c in correlations if c.get('meaningful', False)]
        print(f"   Meaningful correlations (FDR q<0.05, |r|>0.1): {len(meaningful)}")

        # 10. Validate on VAL set (with correct threshold)
        val_results = validate_on_set(
            val, meaningful, si_window, 3, frequency, min_val
        )
        print(f"   Validation confirmation: {val_results.get('confirmation_rate', 0)*100:.1f}%")

        # 11. Test on TEST set
        test_results = validate_on_set(
            test, meaningful, si_window, 3, frequency, min_val
        )
        print(f"   Test confirmation: {test_results.get('confirmation_rate', 0)*100:.1f}%")

        return {
            'status': 'success',
            'symbol': symbol,
            'market': market_type.value,
            'frequency': frequency,
            'n_train': len(train),
            'n_val': len(val),
            'n_test': len(test),
            'si_mean': float(si_train.mean()),
            'si_std': float(si_train.std()),
            'si_window': si_window,
            'n_correlations': len(correlations),
            'n_meaningful': len(meaningful),
            'meaningful_features': [c['feature'] for c in meaningful],
            'correlations': correlations,
            'val_confirmation_rate': val_results.get('confirmation_rate', 0),
            'val_confirmed': val_results.get('confirmed', 0),
            'test_confirmation_rate': test_results.get('confirmation_rate', 0),
            'test_confirmed': test_results.get('confirmed', 0),
        }

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


def validate_on_set(data, candidates, si_window, n_agents, frequency, min_size):
    """Validate discovered correlations on validation/test set."""
    if len(candidates) == 0:
        return {'confirmation_rate': 0, 'confirmed': 0, 'tested': 0, 'reason': 'no_candidates'}

    if len(data) < min_size:
        return {
            'confirmation_rate': 0,
            'confirmed': 0,
            'tested': len(candidates),
            'reason': f'insufficient_data_{len(data)}_vs_{min_size}'
        }

    try:
        strategies = get_default_strategies(frequency)
        population = NichePopulationV2(strategies, n_agents_per_strategy=n_agents, frequency=frequency)

        start_idx = START_IDX[frequency]
        start_idx = min(start_idx, len(data) // 4)
        population.run(data, start_idx=start_idx)

        si = population.compute_si_timeseries(data, window=si_window)

        calc = FeatureCalculatorV2(frequency=frequency)
        features = calc.compute_all(data)

        common_idx = si.index.intersection(features.index)

        if len(common_idx) < 20:
            return {
                'confirmation_rate': 0,
                'confirmed': 0,
                'tested': len(candidates),
                'reason': f'insufficient_aligned_{len(common_idx)}'
            }

        si = si.loc[common_idx]
        features = features.loc[common_idx]

        confirmed = 0
        tested = 0
        details = []

        for cand in candidates:
            feat = cand['feature']
            train_r = cand['r']

            if feat not in features.columns:
                continue

            mask = ~(si.isna() | features[feat].isna())
            if mask.sum() < 15:
                continue

            tested += 1
            val_r, val_p = spearmanr(si[mask], features[feat][mask])

            # Confirm if same direction and p < 0.1
            same_direction = (val_r * train_r) > 0
            significant = val_p < 0.1

            if same_direction and significant:
                confirmed += 1

            details.append({
                'feature': feat,
                'train_r': train_r,
                'val_r': float(val_r),
                'val_p': float(val_p),
                'confirmed': same_direction and significant
            })

        return {
            'confirmation_rate': confirmed / tested if tested > 0 else 0,
            'confirmed': confirmed,
            'tested': tested,
            'details': details
        }
    except Exception as e:
        return {
            'confirmation_rate': 0,
            'confirmed': 0,
            'tested': len(candidates),
            'reason': f'error_{str(e)}'
        }


def main():
    print("=" * 70)
    print("CORRECTED ANALYSIS - FREQUENCY-AWARE PARAMETERS")
    print("=" * 70)

    print("\n📋 FIXES IMPLEMENTED:")
    print("   1. ✅ Feature windows: 1d=1, 7d=7, 30d=30 for daily")
    print("   2. ✅ SI window: 7 for daily (not 168)")
    print("   3. ✅ Validation threshold: 30 for daily (not 300)")
    print("   4. ✅ Regime classification: 7-day lookback for daily")
    print("   5. ✅ Strategy lookbacks: 1-7 days for daily (not 24-168)")
    print("   6. ✅ FDR correction across all tests")
    print("   7. ✅ Realistic transaction costs per market")

    loader = DataLoaderV2(purge_days=7, embargo_days=1)

    all_results = {}
    all_correlations = []

    # ============================================================
    # MAIN ANALYSIS
    # ============================================================
    for market_type, symbols in MARKETS.items():
        print(f"\n{'#'*70}")
        print(f"# MARKET: {market_type.value.upper()}")
        print(f"# Transaction cost: {TRANSACTION_COSTS[market_type]*100:.3f}%")
        print(f"{'#'*70}")

        market_results = {}

        for symbol in symbols:
            result = analyze_asset(symbol, market_type, loader)
            market_results[symbol] = result

            if result.get('status') == 'success':
                for corr in result.get('correlations', []):
                    corr['symbol'] = symbol
                    corr['market'] = market_type.value
                    all_correlations.append(corr)

        all_results[market_type.value] = market_results

    # ============================================================
    # CROSS-MARKET FDR CORRECTION
    # ============================================================
    print("\n" + "=" * 70)
    print("CROSS-MARKET FDR CORRECTION")
    print("=" * 70)

    if all_correlations:
        all_pvals = [c['p'] for c in all_correlations]
        all_q = apply_fdr_correction(all_pvals)

        for i, corr in enumerate(all_correlations):
            corr['global_q'] = all_q[i]
            corr['globally_significant'] = all_q[i] < 0.05
            corr['globally_meaningful'] = all_q[i] < 0.05 and abs(corr['r']) > 0.1

        globally_meaningful = [c for c in all_correlations if c.get('globally_meaningful')]
        print(f"\nTotal tests: {len(all_correlations)}")
        print(f"Globally meaningful (FDR q<0.05, |r|>0.1): {len(globally_meaningful)}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\n| Market | Assets | Meaningful | Val Confirm | Test Confirm |")
    print("|--------|--------|------------|-------------|--------------|")

    for market_type, symbols in MARKETS.items():
        market_data = all_results.get(market_type.value, {})
        n_success = sum(1 for s in symbols if market_data.get(s, {}).get('status') == 'success')
        total_meaningful = sum(
            market_data.get(s, {}).get('n_meaningful', 0)
            for s in symbols
        )

        val_rates = [
            market_data.get(s, {}).get('val_confirmation_rate', 0)
            for s in symbols
            if market_data.get(s, {}).get('status') == 'success'
        ]
        avg_val = np.mean(val_rates) if val_rates else 0

        test_rates = [
            market_data.get(s, {}).get('test_confirmation_rate', 0)
            for s in symbols
            if market_data.get(s, {}).get('status') == 'success'
        ]
        avg_test = np.mean(test_rates) if test_rates else 0

        print(f"| {market_type.value:<6} | {n_success}/{len(symbols)}    | {total_meaningful:>10} | {avg_val:>10.1%} | {avg_test:>12.1%} |")

    # Cross-asset features
    if all_correlations:
        print("\n" + "=" * 70)
        print("FEATURES MEANINGFUL IN 2+ ASSETS (after global FDR)")
        print("=" * 70)

        feature_counts = defaultdict(list)
        for c in all_correlations:
            if c.get('globally_meaningful'):
                feature_counts[c['feature']].append({
                    'symbol': c['symbol'],
                    'market': c['market'],
                    'r': c['r']
                })

        cross_asset = {f: v for f, v in feature_counts.items() if len(v) >= 2}

        print(f"\n{len(cross_asset)} features are meaningful in 2+ assets:\n")
        for feature, instances in sorted(cross_asset.items(), key=lambda x: -len(x[1])):
            avg_r = np.mean([i['r'] for i in instances])
            markets = set(i['market'] for i in instances)
            signs = ['+' if i['r'] > 0 else '-' for i in instances]
            sign_consistent = len(set(signs)) == 1
            consistency = "✅" if sign_consistent else "⚠️"
            print(f"  {consistency} {feature}: {len(instances)} assets, {len(markets)} markets, avg r={avg_r:+.3f}")

    # Save results
    output_dir = Path("results/corrected_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "full_results.json", "w") as f:
        json.dump({
            'results': all_results,
            'correlations': all_correlations,
            'config': {
                'si_windows': SI_WINDOWS,
                'min_val_size': MIN_VAL_SIZE,
                'start_idx': START_IDX,
            },
            'transaction_costs': {k.value: v for k, v in TRANSACTION_COSTS.items()}
        }, f, indent=2, default=float)

    print(f"\n✅ Results saved to: {output_dir / 'full_results.json'}")

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Count confirmed across val+test
    total_val_confirmed = sum(
        r.get('val_confirmed', 0)
        for market in all_results.values()
        for r in market.values()
        if isinstance(r, dict)
    )
    total_test_confirmed = sum(
        r.get('test_confirmed', 0)
        for market in all_results.values()
        for r in market.values()
        if isinstance(r, dict)
    )

    print(f"\nTotal confirmations in VAL: {total_val_confirmed}")
    print(f"Total confirmations in TEST: {total_test_confirmed}")

    if total_val_confirmed >= 5 and total_test_confirmed >= 3:
        print("\n✅ SUCCESS: SI correlations replicate across train/val/test!")
    elif total_val_confirmed >= 3 or total_test_confirmed >= 2:
        print("\n⚠️  PARTIAL SUCCESS: Some correlations replicate")
    else:
        print("\n❌ CORRELATIONS DO NOT REPLICATE")
        print("   This suggests overfitting in training or fundamental issues")


if __name__ == "__main__":
    main()
