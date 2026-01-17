#!/usr/bin/env python3
"""
THOROUGH Cross-Market SI Analysis

Computes SI and correlations for ALL markets:
- Crypto (BTC, ETH, SOL)
- Forex (EUR/USD, GBP/USD, USD/JPY)
- Stocks (SPY, QQQ, AAPL)
- Commodities (GOLD, OIL)

Shows full correlation tables for each market.
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr
from collections import defaultdict

from src.agents.strategies import DEFAULT_STRATEGIES
from src.competition.niche_population import NichePopulation
from src.analysis.features import FeatureCalculator


def load_data(filepath: str) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]

    # Normalize timezone: convert to UTC and remove timezone info
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)

    return df


def analyze_single_asset(name: str, data: pd.DataFrame) -> dict:
    """
    Full SI analysis for a single asset.
    Returns SI stats and all feature correlations.
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING: {name}")
    print(f"{'='*60}")
    print(f"   Data: {len(data)} rows, {data.index[0]} to {data.index[-1]}")

    if len(data) < 500:
        print(f"   ❌ Insufficient data (need 500+, have {len(data)})")
        return {'status': 'insufficient_data', 'rows': len(data)}

    # 1. Run competition
    print(f"\n1. Running NichePopulation competition...")
    population = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=3)

    try:
        population.run(data, start_idx=200)
        si = population.compute_si_timeseries(data, window=168)
    except Exception as e:
        print(f"   ❌ Competition failed: {e}")
        return {'status': 'competition_failed', 'error': str(e)}

    print(f"   SI computed: n={len(si)}, mean={si.mean():.3f}, std={si.std():.3f}")
    print(f"   SI range: [{si.min():.3f}, {si.max():.3f}]")

    # 2. Compute features
    print(f"\n2. Computing market features...")
    calc = FeatureCalculator()
    features = calc.compute_all(data)
    print(f"   Computed {len(features.columns)} features")

    # 3. Align SI with features
    common_idx = si.index.intersection(features.index)
    si_aligned = si.loc[common_idx]
    features_aligned = features.loc[common_idx]
    print(f"   Aligned {len(common_idx)} rows")

    if len(common_idx) < 100:
        print(f"   ❌ Too few aligned rows")
        return {'status': 'insufficient_aligned', 'aligned_rows': len(common_idx)}

    # 4. Compute ALL correlations
    print(f"\n3. Computing correlations with SI...")
    correlations = []

    for col in features_aligned.columns:
        mask = ~(si_aligned.isna() | features_aligned[col].isna())
        if mask.sum() < 50:
            continue

        r, p = spearmanr(si_aligned[mask], features_aligned[col][mask])

        correlations.append({
            'feature': col,
            'r': float(r),
            'p': float(p),
            'n': int(mask.sum()),
            'significant': p < 0.05
        })

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x['r']), reverse=True)

    # Print top correlations
    print(f"\n   TOP 10 SI CORRELATIONS:")
    print(f"   {'Feature':<30} {'r':>8} {'p':>10} {'Sig':>5}")
    print(f"   {'-'*55}")
    for c in correlations[:10]:
        sig = "✅" if c['significant'] else ""
        print(f"   {c['feature']:<30} {c['r']:>8.3f} {c['p']:>10.4f} {sig:>5}")

    # Count significant
    n_significant = sum(1 for c in correlations if c['significant'])
    print(f"\n   Total significant (p<0.05): {n_significant}/{len(correlations)}")

    return {
        'status': 'success',
        'rows': len(data),
        'aligned_rows': len(common_idx),
        'si_mean': float(si.mean()),
        'si_std': float(si.std()),
        'si_min': float(si.min()),
        'si_max': float(si.max()),
        'n_features': len(correlations),
        'n_significant': n_significant,
        'correlations': correlations
    }


def main():
    print("=" * 70)
    print("THOROUGH CROSS-MARKET SI ANALYSIS")
    print("Analyzing SI correlations across Crypto, Forex, Stocks, Commodities")
    print("=" * 70)

    # Define all assets by market
    markets = {
        'CRYPTO': [
            ('BTCUSDT', 'data/crypto/BTCUSDT_1h.csv'),
            ('ETHUSDT', 'data/crypto/ETHUSDT_1h.csv'),
            ('SOLUSDT', 'data/crypto/SOLUSDT_1h.csv'),
        ],
        'FOREX': [
            ('EURUSD', 'data/forex/EURUSD_1h.csv'),
            ('GBPUSD', 'data/forex/GBPUSD_1h.csv'),
            ('USDJPY', 'data/forex/USDJPY_1h.csv'),
        ],
        'STOCKS': [
            ('SPY', 'data/stocks/SPY_1h.csv'),
            ('QQQ', 'data/stocks/QQQ_1h.csv'),
            ('AAPL', 'data/stocks/AAPL_1h.csv'),
        ],
        'COMMODITIES': [
            ('GOLD', 'data/commodities/GOLD_1h.csv'),
            ('OIL', 'data/commodities/OIL_1h.csv'),
        ],
    }

    all_results = {}
    market_summaries = {}

    for market_name, assets in markets.items():
        print(f"\n{'#'*70}")
        print(f"# MARKET: {market_name}")
        print(f"{'#'*70}")

        market_results = {}
        market_correlations = defaultdict(list)

        for asset_name, filepath in assets:
            # Check if file exists
            if not Path(filepath).exists():
                print(f"\n⚠️  {asset_name}: Data file not found at {filepath}")
                market_results[asset_name] = {'status': 'file_not_found'}
                continue

            # Load data
            try:
                data = load_data(filepath)
            except Exception as e:
                print(f"\n⚠️  {asset_name}: Failed to load data: {e}")
                market_results[asset_name] = {'status': 'load_failed', 'error': str(e)}
                continue

            # Analyze
            result = analyze_single_asset(asset_name, data)
            market_results[asset_name] = result

            # Collect correlations for cross-asset comparison
            if result.get('status') == 'success':
                for corr in result.get('correlations', []):
                    market_correlations[corr['feature']].append({
                        'asset': asset_name,
                        'r': corr['r'],
                        'p': corr['p'],
                        'significant': corr['significant']
                    })

        all_results[market_name] = market_results

        # Market summary
        successful = [r for r in market_results.values() if r.get('status') == 'success']
        if successful:
            avg_si = np.mean([r['si_mean'] for r in successful])
            total_sig = sum(r['n_significant'] for r in successful)

            # Find features significant in multiple assets
            cross_asset_features = []
            for feature, corrs in market_correlations.items():
                n_sig = sum(1 for c in corrs if c['significant'])
                if n_sig >= 2:
                    avg_r = np.mean([c['r'] for c in corrs])
                    cross_asset_features.append({
                        'feature': feature,
                        'n_assets': n_sig,
                        'avg_r': avg_r
                    })

            cross_asset_features.sort(key=lambda x: x['n_assets'], reverse=True)

            market_summaries[market_name] = {
                'n_assets_analyzed': len(successful),
                'avg_si_level': float(avg_si),
                'total_significant_correlations': total_sig,
                'cross_asset_features': cross_asset_features[:10]
            }
        else:
            market_summaries[market_name] = {
                'n_assets_analyzed': 0,
                'error': 'No assets successfully analyzed'
            }

    # Final Summary
    print("\n" + "=" * 70)
    print("CROSS-MARKET SUMMARY")
    print("=" * 70)

    print("\n| Market       | Assets | Avg SI | Total Sig. Correlations | Cross-Asset Features |")
    print("|--------------|--------|--------|-------------------------|---------------------|")

    for market, summary in market_summaries.items():
        n_assets = summary.get('n_assets_analyzed', 0)
        if n_assets > 0:
            avg_si = summary.get('avg_si_level', 0)
            total_sig = summary.get('total_significant_correlations', 0)
            n_cross = len(summary.get('cross_asset_features', []))
            print(f"| {market:<12} | {n_assets:>6} | {avg_si:>6.3f} | {total_sig:>23} | {n_cross:>19} |")
        else:
            print(f"| {market:<12} | {n_assets:>6} | {'N/A':>6} | {'N/A':>23} | {'N/A':>19} |")

    # Detailed cross-asset features per market
    print("\n" + "=" * 70)
    print("FEATURES SIGNIFICANT IN 2+ ASSETS (per market)")
    print("=" * 70)

    for market, summary in market_summaries.items():
        cross_features = summary.get('cross_asset_features', [])
        if cross_features:
            print(f"\n{market}:")
            for f in cross_features[:5]:
                print(f"   {f['feature']:<30} (avg r={f['avg_r']:+.3f}, {f['n_assets']} assets)")
        else:
            print(f"\n{market}: No cross-asset significant features")

    # Save results
    output_dir = Path("results/cross_market_full")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "full_analysis.json", "w") as f:
        json.dump({
            'market_summaries': market_summaries,
            'detailed_results': all_results
        }, f, indent=2, default=float)

    print(f"\n\nResults saved to: {output_dir / 'full_analysis.json'}")

    # Final conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Compare SI effectiveness across markets
    for market, summary in market_summaries.items():
        n_cross = len(summary.get('cross_asset_features', []))
        if n_cross >= 2:
            print(f"✅ {market}: SI shows consistent correlations ({n_cross} cross-asset features)")
        elif n_cross == 1:
            print(f"⚠️  {market}: SI shows weak correlations (only 1 cross-asset feature)")
        else:
            print(f"❌ {market}: SI shows no consistent correlations")


if __name__ == "__main__":
    main()
