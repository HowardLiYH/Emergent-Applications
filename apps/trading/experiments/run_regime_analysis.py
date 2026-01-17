#!/usr/bin/env python3
"""
Regime-Conditioned Analysis: Check correlations WITHIN each market regime.

A correlation can flip sign in different regimes, making aggregate correlation meaningless.
This analysis checks SI correlations separately in:
- Trending regime (strong directional movement)
- Mean-reverting regime (range-bound)
- Volatile regime (high uncertainty)
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr
from typing import Literal

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2 as NichePopulation
from src.analysis.features_v2 import FeatureCalculatorV2 as FeatureCalculator

# Regime classification thresholds
def classify_regime(data: pd.DataFrame, idx: int, lookback: int = 7) -> str:
    """
    Classify market regime at a given index.

    Returns: 'trending', 'mean_reverting', or 'volatile'
    """
    if idx < lookback:
        return 'unknown'

    window = data.iloc[idx-lookback:idx]
    returns = window['close'].pct_change().dropna()

    if len(returns) < 3:
        return 'unknown'

    # Volatility
    volatility = returns.std()

    # Trend strength (absolute cumulative return / volatility)
    cum_return = (window['close'].iloc[-1] / window['close'].iloc[0]) - 1
    trend_strength = abs(cum_return) / (volatility + 1e-10)

    # Classification
    if volatility > returns.std() * 2:  # High volatility
        return 'volatile'
    elif trend_strength > 1.5:  # Strong trend
        return 'trending'
    else:  # Mean-reverting
        return 'mean_reverting'


def add_regime_column(data: pd.DataFrame, lookback: int = 7) -> pd.DataFrame:
    """Add regime classification to each row."""
    regimes = []
    for i in range(len(data)):
        regime = classify_regime(data, i, lookback)
        regimes.append(regime)

    data = data.copy()
    data['regime'] = regimes
    return data


def analyze_regime_correlations(si: pd.Series, features: pd.DataFrame, regimes: pd.Series) -> dict:
    """
    Compute correlations within each regime.
    """
    results = {}

    # Align all data
    common_idx = si.index.intersection(features.index).intersection(regimes.index)
    si = si.loc[common_idx]
    features = features.loc[common_idx]
    regimes = regimes.loc[common_idx]

    # Get unique regimes (excluding 'unknown')
    unique_regimes = [r for r in regimes.unique() if r != 'unknown']

    for regime in unique_regimes:
        mask = regimes == regime
        n_samples = mask.sum()

        if n_samples < 30:  # Need minimum samples
            continue

        regime_results = []

        for col in features.columns:
            si_regime = si[mask]
            feat_regime = features.loc[mask, col]

            # Drop NaN
            valid = ~(si_regime.isna() | feat_regime.isna())
            if valid.sum() < 20:
                continue

            r, p = spearmanr(si_regime[valid], feat_regime[valid])

            if not np.isnan(r):
                regime_results.append({
                    'feature': col,
                    'r': float(r),
                    'p': float(p),
                    'n': int(valid.sum()),
                    'significant': p < 0.05
                })

        results[regime] = {
            'n_samples': int(n_samples),
            'correlations': regime_results
        }

    return results


def check_sign_consistency(regime_results: dict) -> dict:
    """
    Check if correlations have consistent signs across regimes.
    """
    # Collect all features
    feature_signs = {}

    for regime, data in regime_results.items():
        for corr in data['correlations']:
            feat = corr['feature']
            if feat not in feature_signs:
                feature_signs[feat] = {}
            feature_signs[feat][regime] = np.sign(corr['r'])

    # Check consistency
    consistency = {}
    for feat, signs in feature_signs.items():
        unique_signs = set(signs.values())
        consistent = len(unique_signs) == 1 or 0 in unique_signs

        consistency[feat] = {
            'signs': {k: int(v) for k, v in signs.items()},
            'consistent': consistent,
            'flips': not consistent
        }

    return consistency


def main():
    print("="*70)
    print("REGIME-CONDITIONED ANALYSIS")
    print("="*70)

    # Load data
    from src.data.loader_v2 import DataLoaderV2, MarketType

    loader = DataLoaderV2()

    # Test on multiple assets
    assets = [
        (MarketType.CRYPTO, 'BTCUSDT'),
        (MarketType.FOREX, 'EURUSD'),
        (MarketType.STOCKS, 'SPY'),
        (MarketType.COMMODITIES, 'GOLD'),
    ]

    all_results = {}
    sign_flips_summary = []

    for market_type, symbol in assets:
        print(f"\n{'='*60}")
        print(f"Analyzing: {market_type.value}/{symbol}")
        print("="*60)

        try:
            # Load data
            data = loader.load(symbol, market_type)
            frequency = 'daily'  # All resampled to daily

            # Add regimes
            data = add_regime_column(data, lookback=7)

            # Check regime distribution
            regime_counts = data['regime'].value_counts()
            print(f"\nRegime distribution:")
            for regime, count in regime_counts.items():
                pct = count / len(data) * 100
                print(f"  {regime}: {count} ({pct:.1f}%)")

            # Run competition and get SI
            strategies = get_default_strategies(frequency=frequency)
            population = NichePopulation(strategies, n_agents_per_strategy=3, frequency=frequency)
            population.run(data, start_idx=30)

            si_window = 7  # 7 days for daily data
            si = population.compute_si_timeseries(data, window=si_window)

            # Compute features
            calc = FeatureCalculator(frequency=frequency)
            features = calc.compute_all(data)

            # Get regime column as series
            regimes = data['regime']

            # Analyze correlations within each regime
            regime_results = analyze_regime_correlations(si, features, regimes)

            # Check sign consistency
            consistency = check_sign_consistency(regime_results)

            # Count flips
            n_flips = sum(1 for v in consistency.values() if v['flips'])
            n_features = len(consistency)

            print(f"\nSign consistency: {n_features - n_flips}/{n_features} features consistent")

            if n_flips > 0:
                print(f"Features with sign flips:")
                for feat, info in consistency.items():
                    if info['flips']:
                        print(f"  - {feat}: {info['signs']}")
                        sign_flips_summary.append({
                            'asset': f"{market_type}/{symbol}",
                            'feature': feat,
                            'signs': info['signs']
                        })

            # Store results
            all_results[f"{market_type.value}/{symbol}"] = {
                'regime_distribution': {k: int(v) for k, v in regime_counts.items()},
                'regime_correlations': regime_results,
                'sign_consistency': consistency,
                'n_sign_flips': n_flips,
                'n_features': n_features
            }

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            all_results[f"{market_type.value}/{symbol}"] = {'error': str(e)}

    # Summary
    print("\n" + "="*70)
    print("REGIME ANALYSIS SUMMARY")
    print("="*70)

    total_flips = sum(r.get('n_sign_flips', 0) for r in all_results.values() if 'n_sign_flips' in r)
    total_features = sum(r.get('n_features', 0) for r in all_results.values() if 'n_features' in r)

    print(f"\nTotal sign flips across all assets: {total_flips}")
    print(f"Total feature-asset pairs tested: {total_features}")

    if total_flips == 0:
        print("\n✅ NO SIGN FLIPS: Correlations are consistent across regimes")
    elif total_flips < total_features * 0.1:
        print(f"\n⚠️ MINOR INCONSISTENCY: {total_flips}/{total_features} ({total_flips/total_features*100:.1f}%) flips")
    else:
        print(f"\n❌ SIGNIFICANT INCONSISTENCY: {total_flips}/{total_features} ({total_flips/total_features*100:.1f}%) flips")

    # Save results
    output_dir = Path("results/regime_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "regime_results.json", 'w') as f:
        json.dump({
            'results': all_results,
            'sign_flips_summary': sign_flips_summary,
            'total_flips': total_flips,
            'total_features': total_features
        }, f, indent=2, default=str)

    print(f"\n✅ Results saved to {output_dir}/regime_results.json")


if __name__ == "__main__":
    main()
