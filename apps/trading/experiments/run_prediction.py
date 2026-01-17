#!/usr/bin/env python3
"""
Run prediction pipeline: Does SI predict future outcomes?
Pipeline 2: 2 prediction features (next_day_return, next_day_volatility)

Usage:
    python experiments/run_prediction.py
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import json
from pathlib import Path


def lagged_correlation(si: pd.Series, target: pd.Series, lag: int) -> float:
    """Correlation between SI(t) and target(t+lag)."""
    si_shifted = si.shift(lag)
    mask = ~(si_shifted.isna() | target.isna())
    if mask.sum() < 50:
        return np.nan
    r, _ = spearmanr(si_shifted[mask], target[mask])
    return r


def signal_decay(si: pd.Series, target: pd.Series, max_lag: int = 168) -> dict:
    """Analyze how SI's predictive power decays over time."""
    lags = list(range(1, max_lag + 1, 6))  # Every 6 hours
    rs = [lagged_correlation(si, target, lag) for lag in lags]

    # Find optimal lag
    valid_rs = [(l, r) for l, r in zip(lags, rs) if not np.isnan(r)]
    if len(valid_rs) == 0:
        return {
            'lags': lags,
            'correlations': rs,
            'optimal_lag': None,
            'peak_r': None,
            'half_life': None
        }

    best_idx = np.argmax([abs(r) for _, r in valid_rs])
    optimal_lag = valid_rs[best_idx][0]
    peak_r = valid_rs[best_idx][1]

    # Estimate half-life
    half_r = abs(peak_r) / 2
    half_life = None
    for l, r in valid_rs[best_idx:]:
        if abs(r) < half_r:
            half_life = l - optimal_lag
            break

    return {
        'lags': lags,
        'correlations': rs,
        'optimal_lag': optimal_lag,
        'peak_r': peak_r,
        'half_life': half_life
    }


def main():
    print("=" * 60)
    print("PHASE 5: PREDICTION PIPELINE")
    print("Does SI predict future outcomes?")
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
    prediction_targets = ['next_day_return', 'next_day_volatility']

    all_results = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {symbol}")
        print('='*60)

        # 2. Load data
        try:
            si = pd.read_csv(output_dir / f"si_{symbol}_train.csv", index_col=0, parse_dates=True).squeeze()
            features = pd.read_csv(output_dir / f"features_{symbol}_train.csv", index_col=0, parse_dates=True)
        except FileNotFoundError as e:
            print(f"   ❌ {e}")
            continue

        results = {}

        for target in prediction_targets:
            if target not in features.columns:
                print(f"   ⚠️  {target} not found in features")
                continue

            print(f"\n2. Analyzing {target}...")

            # Signal decay
            decay = signal_decay(si, features[target])
            print(f"   Optimal lag: {decay['optimal_lag']} hours")
            print(f"   Peak correlation: {decay['peak_r']:.3f}" if decay['peak_r'] else "   Peak correlation: N/A")
            print(f"   Half-life: {decay['half_life']} hours" if decay['half_life'] else "   Half-life: N/A")

            # Granger causality
            print(f"   Running Granger causality...")
            df = pd.concat([si, features[target]], axis=1).dropna()
            df.columns = ['si', 'target']

            granger_significant = False
            min_p = None

            if len(df) > 100:
                try:
                    granger = grangercausalitytests(df[['target', 'si']], maxlag=24, verbose=False)
                    p_values = [granger[lag][0]['ssr_ftest'][1] for lag in range(1, 25)]
                    min_p = min(p_values)
                    best_lag = p_values.index(min_p) + 1

                    print(f"   Granger p-value: {min_p:.4f} at lag {best_lag}")
                    granger_significant = min_p < 0.05
                except Exception as e:
                    print(f"   Granger test failed: {e}")

            results[target] = {
                'signal_decay': {
                    'optimal_lag': decay['optimal_lag'],
                    'peak_r': float(decay['peak_r']) if decay['peak_r'] else None,
                    'half_life': decay['half_life']
                },
                'granger_p': float(min_p) if min_p else None,
                'granger_significant': bool(granger_significant)
            }

        all_results[symbol] = results

    # 3. Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    with open(output_dir / "prediction_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE COMPLETE!")
    print("=" * 60)

    # Summary
    for symbol, results in all_results.items():
        print(f"\n{symbol}:")
        for target, res in results.items():
            granger = "✅ SI Granger-causes" if res.get('granger_significant') else "❌ No causality"
            print(f"  {target}: {granger}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
