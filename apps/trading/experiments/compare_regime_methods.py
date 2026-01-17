#!/usr/bin/env python3
"""
Compare Regime Detection Methods

Compares:
1. Rule-based (current approach)
2. HMM 2-state (industry standard)
3. HMM 3-state
4. GMM 2-state (data-driven)

Metrics:
- Regime distribution
- Regime stability (avg duration)
- SI correlation sign flips within regimes
- Agreement between methods
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr
from collections import defaultdict

from src.data.loader_v2 import DataLoaderV2, MarketType
from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2 as NichePopulation
from src.analysis.features_v2 import FeatureCalculatorV2 as FeatureCalculator
from src.analysis.regime_detection import get_detector


def compute_regime_stats(regimes: pd.Series) -> dict:
    """Compute statistics about regime assignments."""
    # Distribution
    counts = regimes.value_counts(normalize=True).to_dict()

    # Average regime duration
    regime_changes = (regimes != regimes.shift()).cumsum()
    durations = regimes.groupby(regime_changes).size()
    avg_duration = durations.mean()

    # Number of transitions
    n_transitions = (regimes != regimes.shift()).sum() - 1

    return {
        'distribution': {str(k): float(v) for k, v in counts.items()},
        'avg_duration': float(avg_duration),
        'n_transitions': int(n_transitions),
        'n_samples': len(regimes)
    }


def compute_sign_flips(si: pd.Series, features: pd.DataFrame, regimes: pd.Series) -> dict:
    """Compute SI correlation sign flips across regimes."""
    common_idx = si.index.intersection(features.index).intersection(regimes.index)
    si = si.loc[common_idx]
    features = features.loc[common_idx]
    regimes = regimes.loc[common_idx]

    unique_regimes = [r for r in regimes.unique() if not pd.isna(r)]

    feature_signs = defaultdict(dict)

    for regime in unique_regimes:
        mask = regimes == regime
        if mask.sum() < 30:
            continue

        for col in features.columns:
            si_r = si[mask]
            feat_r = features.loc[mask, col]

            valid = ~(si_r.isna() | feat_r.isna())
            if valid.sum() < 20:
                continue

            try:
                r, p = spearmanr(si_r[valid], feat_r[valid])
                if not np.isnan(r) and p < 0.1:  # Significant at 10%
                    feature_signs[col][int(regime)] = int(np.sign(r))
            except:
                pass

    # Count flips
    n_flips = 0
    n_total = 0
    flipped_features = []

    for feat, signs in feature_signs.items():
        if len(signs) >= 2:
            n_total += 1
            unique_signs = set(signs.values())
            if len(unique_signs) > 1 and 0 not in unique_signs:
                n_flips += 1
                flipped_features.append(feat)

    return {
        'n_flips': n_flips,
        'n_total': n_total,
        'flip_rate': n_flips / n_total if n_total > 0 else 0,
        'flipped_features': flipped_features
    }


def compare_regime_agreement(regimes_dict: dict) -> dict:
    """Compare agreement between different regime methods."""
    methods = list(regimes_dict.keys())
    agreement = {}

    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            r1 = regimes_dict[m1]
            r2 = regimes_dict[m2]

            common_idx = r1.index.intersection(r2.index)
            r1 = r1.loc[common_idx]
            r2 = r2.loc[common_idx]

            # For 2-state methods, compute direct agreement
            # For different n_states, bin into low/high
            if r1.nunique() == r2.nunique():
                agree = (r1 == r2).mean()
            else:
                # Bin into binary: 0 vs rest
                r1_bin = (r1 > 0).astype(int)
                r2_bin = (r2 > 0).astype(int)
                agree = (r1_bin == r2_bin).mean()

            agreement[f'{m1}_vs_{m2}'] = float(agree)

    return agreement


def analyze_asset(symbol: str, market_type: MarketType, loader: DataLoaderV2) -> dict:
    """Run full comparison on a single asset."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {market_type.value}/{symbol}")
    print("="*60)

    # Load data
    data = loader.load(symbol, market_type)
    frequency = 'daily'

    # Methods to compare
    methods = {
        'rule_2': get_detector('rule', n_regimes=2),
        'hmm_2': get_detector('hmm', n_regimes=2),
        'hmm_3': get_detector('hmm', n_regimes=3),
        'gmm_2': get_detector('gmm', n_regimes=2),
    }

    # Fit and predict regimes
    regimes_dict = {}
    regime_stats = {}

    for name, detector in methods.items():
        try:
            detector.fit(data)
            regimes = detector.predict(data)
            regimes_dict[name] = regimes
            regime_stats[name] = compute_regime_stats(regimes)

            print(f"\n{name}:")
            print(f"  Distribution: {regime_stats[name]['distribution']}")
            print(f"  Avg duration: {regime_stats[name]['avg_duration']:.1f} days")
            print(f"  Transitions: {regime_stats[name]['n_transitions']}")
        except Exception as e:
            print(f"  {name}: Error - {e}")
            continue

    # Run SI competition
    strategies = get_default_strategies(frequency=frequency)
    population = NichePopulation(strategies, n_agents_per_strategy=3, frequency=frequency)
    population.run(data, start_idx=30)
    si = population.compute_si_timeseries(data, window=7)

    # Compute features
    calc = FeatureCalculator(frequency=frequency)
    features = calc.compute_all(data)

    # Compute sign flips for each method
    flip_results = {}
    for name, regimes in regimes_dict.items():
        flip_results[name] = compute_sign_flips(si, features, regimes)
        print(f"\n{name} sign flips: {flip_results[name]['n_flips']}/{flip_results[name]['n_total']} ({flip_results[name]['flip_rate']*100:.1f}%)")

    # Compare agreement
    agreement = compare_regime_agreement(regimes_dict)

    print(f"\nMethod agreement:")
    for pair, agree in agreement.items():
        print(f"  {pair}: {agree*100:.1f}%")

    return {
        'regime_stats': regime_stats,
        'flip_results': flip_results,
        'agreement': agreement
    }


def main():
    print("="*70)
    print("REGIME DETECTION METHOD COMPARISON")
    print("="*70)

    loader = DataLoaderV2()

    assets = [
        (MarketType.CRYPTO, 'BTCUSDT'),
        (MarketType.FOREX, 'EURUSD'),
        (MarketType.STOCKS, 'SPY'),
        (MarketType.COMMODITIES, 'GOLD'),
    ]

    all_results = {}

    for market_type, symbol in assets:
        try:
            result = analyze_asset(symbol, market_type, loader)
            all_results[f"{market_type.value}/{symbol}"] = result
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate summary
    print("\n" + "="*70)
    print("SUMMARY: SIGN FLIP RATES BY METHOD")
    print("="*70)

    method_flips = defaultdict(list)
    for asset, result in all_results.items():
        for method, flips in result.get('flip_results', {}).items():
            method_flips[method].append(flips['flip_rate'])

    print("\n| Method | Avg Flip Rate | Interpretation |")
    print("|--------|---------------|----------------|")

    for method in ['rule_2', 'hmm_2', 'hmm_3', 'gmm_2']:
        if method in method_flips:
            avg_rate = np.mean(method_flips[method]) * 100
            if avg_rate < 15:
                interp = "✅ Low - regimes well-defined"
            elif avg_rate < 25:
                interp = "⚠️ Medium - some regime dependency"
            else:
                interp = "❌ High - regimes not capturing structure"
            print(f"| {method:8} | {avg_rate:11.1f}% | {interp} |")

    # Agreement summary
    print("\n" + "="*70)
    print("SUMMARY: METHOD AGREEMENT")
    print("="*70)

    agreement_agg = defaultdict(list)
    for asset, result in all_results.items():
        for pair, agree in result.get('agreement', {}).items():
            agreement_agg[pair].append(agree)

    print("\n| Comparison | Avg Agreement |")
    print("|------------|---------------|")
    for pair in sorted(agreement_agg.keys()):
        avg_agree = np.mean(agreement_agg[pair]) * 100
        print(f"| {pair:20} | {avg_agree:11.1f}% |")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    # Find best method (lowest flip rate)
    best_method = min(method_flips.keys(), key=lambda m: np.mean(method_flips[m]))
    best_rate = np.mean(method_flips[best_method]) * 100

    print(f"\n✅ Best method: {best_method} ({best_rate:.1f}% flip rate)")

    if 'hmm' in best_method:
        print("   → HMM provides probabilistic regime detection")
        print("   → Regimes are more stable (fewer whipsaws)")
        print("   → Recommended for production use")
    elif 'gmm' in best_method:
        print("   → GMM captures data-driven clusters")
        print("   → Good for multivariate regime detection")
    else:
        print("   → Rule-based is simpler but may miss nuances")

    # Save results
    output_dir = Path("results/regime_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'method_flip_rates': {m: float(np.mean(rates)) for m, rates in method_flips.items()},
        'method_agreement': {p: float(np.mean(rates)) for p, rates in agreement_agg.items()},
        'best_method': best_method,
        'best_flip_rate': float(best_rate),
        'detailed_results': all_results
    }

    with open(output_dir / "comparison_results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✅ Results saved to {output_dir}/comparison_results.json")


if __name__ == "__main__":
    main()
