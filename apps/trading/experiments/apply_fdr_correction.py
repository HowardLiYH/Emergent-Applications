#!/usr/bin/env python3
"""
Apply FDR correction to cross-market results.
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
from pathlib import Path

def main():
    # Load results
    with open('results/cross_market_full/full_analysis.json') as f:
        results = json.load(f)

    # Collect all p-values
    all_pvalues = []
    all_info = []

    for market, market_data in results.get('detailed_results', {}).items():
        for asset, asset_data in market_data.items():
            if asset_data.get('status') == 'success':
                for corr in asset_data.get('correlations', []):
                    all_pvalues.append(corr['p'])
                    all_info.append({
                        'market': market,
                        'asset': asset,
                        'feature': corr['feature'],
                        'r': corr['r']
                    })

    n = len(all_pvalues)

    # Apply Benjamini-Hochberg FDR correction
    sorted_idx = np.argsort(all_pvalues)
    sorted_pvals = np.array(all_pvalues)[sorted_idx]

    # BH adjusted p-values
    bh_adjusted = np.zeros(n)
    for i in range(n):
        bh_adjusted[i] = sorted_pvals[i] * n / (i + 1)

    # Make monotonic
    for i in range(n-2, -1, -1):
        bh_adjusted[i] = min(bh_adjusted[i], bh_adjusted[i+1])

    # Map back to original order
    adjusted_original = np.zeros(n)
    for i, idx in enumerate(sorted_idx):
        adjusted_original[idx] = bh_adjusted[i]

    # Count significant after FDR
    fdr_significant = np.sum(adjusted_original < 0.05)
    original_significant = np.sum(np.array(all_pvalues) < 0.05)

    print('FDR CORRECTION RESULTS')
    print('=' * 50)
    print(f'Total tests: {n}')
    print(f'Original significant (p < 0.05): {original_significant}')
    print(f'After FDR correction (q < 0.05): {fdr_significant}')
    print(f'Reduction: {(1 - fdr_significant/original_significant)*100:.1f}%')

    # Filter by effect size too
    effect_meaningful = sum(1 for i in range(n) if adjusted_original[i] < 0.05 and abs(all_info[i]['r']) > 0.1)
    print(f'After FDR + |r| > 0.1 filter: {effect_meaningful}')

    # Show remaining significant features
    print('\n' + '=' * 50)
    print('FEATURES SURVIVING FDR + EFFECT SIZE FILTER')
    print('=' * 50)

    surviving = []
    for i in range(n):
        if adjusted_original[i] < 0.05 and abs(all_info[i]['r']) > 0.1:
            surviving.append({
                **all_info[i],
                'q_value': float(adjusted_original[i])
            })

    # Group by market
    by_market = {}
    for s in surviving:
        market = s['market']
        if market not in by_market:
            by_market[market] = []
        by_market[market].append(s)

    for market, features in by_market.items():
        print(f'\n{market}:')
        # Unique features
        unique_features = {}
        for f in features:
            feat = f['feature']
            if feat not in unique_features:
                unique_features[feat] = []
            unique_features[feat].append(f)

        for feat, instances in sorted(unique_features.items(), key=lambda x: -len(x[1])):
            avg_r = np.mean([i['r'] for i in instances])
            print(f'  {feat}: {len(instances)} assets, avg r={avg_r:+.3f}')


if __name__ == "__main__":
    main()
