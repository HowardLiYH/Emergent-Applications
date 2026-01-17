#!/usr/bin/env python3
"""
Phase 9: Cross-market validation.
Does SI work beyond crypto to other market types?

NOTE: This requires data for forex, stocks, commodities.
Currently only crypto data is available.
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter
from scipy.stats import spearmanr

from src.data.loader import MultiMarketLoader, MarketType
from src.data.validation import DataValidator
from src.analysis.features import FeatureCalculator
from src.agents.strategies import DEFAULT_STRATEGIES
from src.competition.niche_population import NichePopulation
from src.analysis.correlations import CorrelationAnalyzer


# Markets to test
MARKETS = {
    MarketType.CRYPTO: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    MarketType.FOREX: ['EURUSD', 'GBPUSD', 'USDJPY'],
    MarketType.STOCKS: ['SPY', 'QQQ', 'AAPL'],
    MarketType.COMMODITIES: ['GOLD', 'OIL'],
}


def run_single_asset(market_type: MarketType, symbol: str) -> dict:
    """Run full pipeline on a single asset."""
    print(f"\n{'='*60}")
    print(f"Processing: {market_type.value}/{symbol}")
    print('='*60)

    try:
        # 1. Load data
        loader = MultiMarketLoader(market_type)
        data = loader.load(symbol)
        print(f"   Loaded {len(data)} rows")

        # 2. Validate
        validator = DataValidator(strict=False)
        validation = validator.validate(data, f"{market_type.value}/{symbol}")

        if not validation['valid']:
            return {
                'symbol': symbol,
                'market': market_type.value,
                'status': 'INVALID_DATA',
                'issues': validation['issues']
            }

        # 3. Split
        train, val, test = loader.temporal_split(data)

        if len(train) < 1000:
            return {
                'symbol': symbol,
                'market': market_type.value,
                'status': 'INSUFFICIENT_DATA',
                'n_rows': len(train)
            }

        # 4. Run competition
        print("   Running competition...")
        population = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=3)
        population.run(train, start_idx=200)
        si = population.compute_si_timeseries(train, window=168)
        print(f"   SI: mean={si.mean():.3f}, std={si.std():.3f}")

        # 5. Compute features
        print("   Computing features...")
        calc = FeatureCalculator()
        features = calc.compute_all(train)

        # 6. Align
        common_idx = si.index.intersection(features.index)
        si = si.loc[common_idx]
        features = features.loc[common_idx]
        print(f"   Aligned {len(common_idx)} rows")

        # 7. Run discovery (subset for speed)
        print("   Running discovery...")
        analyzer = CorrelationAnalyzer(block_size=24)
        discovery_features = calc.get_discovery_features()[:20]  # Subset for speed
        results = analyzer.run_discovery(si, features, discovery_features)

        # 8. Summary
        significant = results[results['significant']]

        return {
            'symbol': symbol,
            'market': market_type.value,
            'status': 'SUCCESS',
            'n_rows': len(common_idx),
            'si_mean': float(si.mean()),
            'si_std': float(si.std()),
            'n_significant': len(significant),
            'top_feature': results.iloc[0]['feature'] if len(results) > 0 else None,
            'top_r': float(results.iloc[0]['r']) if len(results) > 0 else None,
            'significant_features': significant['feature'].tolist(),
        }

    except Exception as e:
        return {
            'symbol': symbol,
            'market': market_type.value,
            'status': 'ERROR',
            'error': str(e)
        }


def main():
    print("="*60)
    print("PHASE 9: CROSS-MARKET VALIDATION")
    print("="*60)

    all_results = {}

    for market_type, symbols in MARKETS.items():
        print(f"\n{'='*60}")
        print(f"MARKET: {market_type.value.upper()}")
        print("="*60)

        for symbol in symbols:
            result = run_single_asset(market_type, symbol)
            all_results[f"{market_type.value}/{symbol}"] = result

    # Summary
    print("\n" + "="*60)
    print("CROSS-MARKET SUMMARY")
    print("="*60)

    # Group by market
    market_summary = {}
    for key, result in all_results.items():
        market = result['market']
        if market not in market_summary:
            market_summary[market] = {'success': 0, 'total': 0, 'significant': 0}

        market_summary[market]['total'] += 1
        if result['status'] == 'SUCCESS':
            market_summary[market]['success'] += 1
            market_summary[market]['significant'] += result.get('n_significant', 0)

    print("\nBy Market:")
    for market, stats in market_summary.items():
        print(f"  {market}: {stats['success']}/{stats['total']} assets, "
              f"{stats['significant']} total significant correlations")

    # Find common features
    print("\n" + "="*60)
    print("COMMON FEATURES ACROSS ASSETS")
    print("="*60)

    all_significant = []
    for key, result in all_results.items():
        if result['status'] == 'SUCCESS':
            all_significant.extend(result.get('significant_features', []))

    feature_counts = Counter(all_significant)

    print("\nFeatures significant in multiple assets:")
    for feature, count in feature_counts.most_common(10):
        if count >= 2:
            print(f"  {feature}: {count} assets")

    # Save results
    output_dir = Path("results/si_correlations")
    with open(output_dir / "cross_market_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("CROSS-MARKET CONCLUSION")
    print("="*60)

    # Check if SI works across market types
    markets_with_findings = sum(1 for m, s in market_summary.items() if s['significant'] > 0)

    if markets_with_findings >= 3:
        print("✅ SI FINDINGS GENERALIZE ACROSS MARKETS!")
        print(f"   Found significant correlations in {markets_with_findings}/{len(market_summary)} market types")
    elif markets_with_findings >= 2:
        print("⚠️  SI PARTIALLY GENERALIZES")
        print(f"   Found significant correlations in {markets_with_findings}/{len(market_summary)} market types")
    else:
        print("⚠️  SI MAY BE MARKET-SPECIFIC")
        print(f"   Only found correlations in {markets_with_findings}/{len(market_summary)} market types")

    # Common feature check
    universal_features = [f for f, c in feature_counts.items() if c >= 3]
    if len(universal_features) > 0:
        print(f"\n✅ UNIVERSAL SI CORRELATES: {universal_features}")
    else:
        partially_universal = [f for f, c in feature_counts.items() if c >= 2]
        if partially_universal:
            print(f"\n⚠️  Features significant in 2+ assets: {partially_universal}")
        else:
            print("\n⚠️  No features significant across multiple assets")

    print(f"\n✅ CROSS-MARKET VALIDATION COMPLETE!")
    print(f"Results saved to: {output_dir / 'cross_market_results.json'}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
