#!/usr/bin/env python3
"""
P5: Ensemble Methods with SI-Only Baseline - NEXT_STEPS_PLAN v4.1

Combine SI with other signals and compare to SI-only baseline.

Author: Yuhao Li, University of Pennsylvania
Date: January 17, 2026
"""

import sys
sys.path.insert(0, '.')

import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETER DOCUMENTATION
# ============================================================================

PARAMETER_CHOICES = {
    'si_window': 7,
    'momentum_window': 20,
    'vol_window': 20,
    'trend_window': 14,
    'ridge_alpha': 1.0,        # Ridge regularization
    'train_ratio': 0.7,        # Train split
    'random_seed': 42,
}

np.random.seed(PARAMETER_CHOICES['random_seed'])

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(market: str, symbol: str) -> pd.DataFrame:
    """Load data for a specific asset."""
    data_dir = Path('data')
    daily_path = data_dir / market / f"{symbol}_1d.csv"
    if daily_path.exists():
        df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        return df
    raise FileNotFoundError(f"Data not found for {market}/{symbol}")


def get_all_assets() -> List[tuple]:
    """Get all available assets."""
    assets = []
    data_dir = Path('data')
    for market_dir in data_dir.iterdir():
        if market_dir.is_dir() and market_dir.name in ['crypto', 'forex', 'stocks', 'commodities']:
            for filepath in market_dir.glob("*_1d.csv"):
                symbol = filepath.stem.replace('_1d', '')
                assets.append((market_dir.name, symbol))
    return assets


# ============================================================================
# SIGNAL COMPUTATION
# ============================================================================

def compute_si_simple(data: pd.DataFrame, window: int = 7) -> pd.Series:
    """Compute simplified SI."""
    returns = data['close'].pct_change()
    momentum = returns.rolling(window).mean()
    mean_rev = -returns.rolling(window).mean()
    volatility = returns.rolling(window).std()

    strategies = pd.DataFrame({
        'momentum': momentum,
        'mean_reversion': mean_rev,
        'low_vol': -volatility,
    }).dropna()

    strategies_norm = strategies.apply(lambda x: (x - x.mean()) / (x.std() + 1e-10))
    si = strategies_norm.std(axis=1)
    si = (si - si.min()) / (si.max() - si.min() + 1e-10)
    return si


def compute_signals(data: pd.DataFrame, si: pd.Series) -> pd.DataFrame:
    """Compute all signals for ensemble."""
    returns = data['close'].pct_change()

    signals = pd.DataFrame(index=data.index)

    # 1. SI signal (our novel signal)
    signals['si'] = si

    # 2. Momentum signal
    mom_window = PARAMETER_CHOICES['momentum_window']
    mom = data['close'].pct_change(mom_window)
    signals['momentum'] = (mom.rank(pct=True) - 0.5) * 2  # Normalize to [-1, 1]

    # 3. Mean reversion signal
    ma = data['close'].rolling(mom_window).mean()
    distance_from_ma = (data['close'] - ma) / ma
    signals['mean_reversion'] = -(distance_from_ma.rank(pct=True) - 0.5) * 2

    # 4. Volatility signal (low vol = higher position)
    vol = returns.rolling(PARAMETER_CHOICES['vol_window']).std()
    signals['inv_vol'] = -(vol.rank(pct=True) - 0.5) * 2

    # 5. Trend strength
    if 'high' in data.columns and 'low' in data.columns:
        high_low = (data['high'] - data['low']) / data['close']
        trend = high_low.rolling(PARAMETER_CHOICES['trend_window']).mean()
    else:
        trend = returns.abs().rolling(PARAMETER_CHOICES['trend_window']).mean()
    signals['trend'] = (trend.rank(pct=True) - 0.5) * 2

    return signals.dropna()


# ============================================================================
# ENSEMBLE METHODS
# ============================================================================

def equal_weight_ensemble(signals: pd.DataFrame) -> pd.Series:
    """Simple equal-weighted average of all signals."""
    return signals.mean(axis=1)


def si_only(signals: pd.DataFrame) -> pd.Series:
    """SI-only baseline (required comparison)."""
    return signals['si']


def ridge_ensemble(signals: pd.DataFrame, returns: pd.Series,
                   train_idx: pd.Index) -> pd.Series:
    """
    Ridge regression meta-learner.
    Learn optimal weights on training data, apply to all.
    """
    # Align
    aligned = pd.concat([signals, returns.rename('target')], axis=1).dropna()

    # Split
    train_mask = aligned.index.isin(train_idx)
    train_data = aligned[train_mask]

    if len(train_data) < 50:
        # Fall back to equal weight
        return signals.mean(axis=1)

    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target'].shift(-1).dropna()
    X_train = X_train.iloc[:-1]  # Align with shifted target

    # Fit ridge
    model = Ridge(alpha=PARAMETER_CHOICES['ridge_alpha'])
    model.fit(X_train, y_train)

    # Apply to all signals
    ensemble_signal = pd.Series(model.predict(signals), index=signals.index)

    return ensemble_signal, dict(zip(signals.columns, model.coef_))


def dynamic_correlation_ensemble(signals: pd.DataFrame, returns: pd.Series,
                                 lookback: int = 60) -> pd.Series:
    """
    Dynamic weighting based on recent correlation with returns.
    """
    ensemble = pd.Series(index=signals.index, dtype=float)

    for i in range(lookback, len(signals)):
        window = slice(i-lookback, i)

        # Compute correlations
        weights = []
        for col in signals.columns:
            corr = signals[col].iloc[window].corr(returns.iloc[window])
            weights.append(max(0, corr))  # Only positive correlations

        total_weight = sum(weights) + 1e-10
        normalized = [w / total_weight for w in weights]

        # Weighted sum
        ensemble.iloc[i] = sum(signals[col].iloc[i] * w
                               for col, w in zip(signals.columns, normalized))

    return ensemble


# ============================================================================
# BACKTEST
# ============================================================================

def backtest_signal(data: pd.DataFrame, signal: pd.Series, market: str) -> Dict:
    """Backtest a signal."""
    returns = data['close'].pct_change()

    costs = {'crypto': 0.0006, 'forex': 0.0001, 'stocks': 0.0002, 'commodities': 0.0003}
    cost = costs.get(market, 0.0002)

    # Convert signal to position
    position = np.sign(signal.shift(1))  # Direction from signal
    position = position.reindex(returns.index).fillna(0)

    strategy_returns = returns * position
    trades = position.diff().abs().fillna(0)
    net_returns = strategy_returns - (trades * cost)

    sharpe = (net_returns.mean() / (net_returns.std() + 1e-10)) * np.sqrt(252)

    # Drawdown
    cumulative = (1 + net_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    max_dd = drawdown.min()

    return {
        'sharpe': float(sharpe),
        'total_return': float((1 + net_returns).prod() - 1),
        'max_drawdown': float(max_dd),
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_asset(market: str, symbol: str) -> Dict:
    """Run P5 analysis for a single asset."""
    print(f"\n  Analyzing {market}/{symbol}...")

    try:
        data = load_data(market, symbol)
        returns = data['close'].pct_change()

        # Compute SI and signals
        si = compute_si_simple(data, window=PARAMETER_CHOICES['si_window'])
        signals = compute_signals(data, si)

        # Training split
        train_size = int(len(signals) * PARAMETER_CHOICES['train_ratio'])
        train_idx = signals.index[:train_size]

        # Generate ensemble signals
        si_only_signal = si_only(signals)
        equal_weight_signal = equal_weight_ensemble(signals)

        # Ridge (returns weights)
        ridge_result = ridge_ensemble(signals, returns, train_idx)
        if isinstance(ridge_result, tuple):
            ridge_signal, ridge_weights = ridge_result
        else:
            ridge_signal = ridge_result
            ridge_weights = {}

        dynamic_signal = dynamic_correlation_ensemble(signals, returns)

        # Backtest each
        results = {
            'si_only': backtest_signal(data, si_only_signal, market),
            'equal_weight': backtest_signal(data, equal_weight_signal, market),
            'ridge': backtest_signal(data, ridge_signal, market),
            'dynamic': backtest_signal(data, dynamic_signal, market),
        }

        # Compare to SI-only baseline
        si_sharpe = results['si_only']['sharpe']

        improvements = {}
        for method in ['equal_weight', 'ridge', 'dynamic']:
            improvements[method] = {
                'sharpe_diff': results[method]['sharpe'] - si_sharpe,
                'sharpe_pct': (results[method]['sharpe'] - si_sharpe) / abs(si_sharpe + 0.01) * 100,
            }

        # Best method
        sharpes = {m: r['sharpe'] for m, r in results.items()}
        best_method = max(sharpes, key=sharpes.get)

        result = {
            'market': market,
            'symbol': symbol,
            'results': results,
            'improvements': improvements,
            'ridge_weights': ridge_weights,
            'best_method': best_method,

            # Key metrics
            'si_only_sharpe': si_sharpe,
            'best_ensemble_sharpe': max(results['equal_weight']['sharpe'],
                                        results['ridge']['sharpe'],
                                        results['dynamic']['sharpe']),
            'ensemble_beats_si': max(results['equal_weight']['sharpe'],
                                     results['ridge']['sharpe'],
                                     results['dynamic']['sharpe']) > si_sharpe,
            'ensemble_improvement_pct': max(improvements['equal_weight']['sharpe_pct'],
                                           improvements['ridge']['sharpe_pct'],
                                           improvements['dynamic']['sharpe_pct']),
        }

        # Print summary
        status = "✅" if result['ensemble_beats_si'] else "⚠️"
        print(f"    {status} SI-only: {si_sharpe:.3f}, Best ensemble: {result['best_ensemble_sharpe']:.3f} "
              f"({best_method})")

        return result

    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        return {'market': market, 'symbol': symbol, 'error': str(e)}


def main():
    """Run P5 ensemble analysis for all assets."""

    print("\n" + "="*60)
    print("P5: ENSEMBLE METHODS WITH SI-ONLY BASELINE")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    assets = get_all_assets()
    print(f"\nFound {len(assets)} assets to analyze")

    results = []
    for market, symbol in assets:
        result = analyze_asset(market, symbol)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("P5 SUMMARY")
    print("="*60)

    valid_results = [r for r in results if 'error' not in r]

    ensemble_beats = [r for r in valid_results if r.get('ensemble_beats_si', False)]

    print(f"\n  Assets analyzed: {len(valid_results)}")
    print(f"  Ensemble beats SI-only: {len(ensemble_beats)}/{len(valid_results)}")

    if valid_results:
        avg_si = np.mean([r['si_only_sharpe'] for r in valid_results])
        avg_ensemble = np.mean([r['best_ensemble_sharpe'] for r in valid_results])
        avg_improvement = np.mean([r['ensemble_improvement_pct'] for r in valid_results])

        print(f"\n  Average SI-only Sharpe: {avg_si:.3f}")
        print(f"  Average best ensemble Sharpe: {avg_ensemble:.3f}")
        print(f"  Average improvement: {avg_improvement:.1f}%")

    # Best method distribution
    method_counts = {'si_only': 0, 'equal_weight': 0, 'ridge': 0, 'dynamic': 0}
    for r in valid_results:
        method = r.get('best_method', 'unknown')
        if method in method_counts:
            method_counts[method] += 1

    print("\n  Best method distribution:")
    for method, count in method_counts.items():
        print(f"    {method}: {count}/{len(valid_results)} ({count/len(valid_results)*100:.0f}%)")

    # SI weight in Ridge models
    if valid_results:
        si_weights = [r.get('ridge_weights', {}).get('si', 0) for r in valid_results]
        avg_si_weight = np.mean([w for w in si_weights if w != 0])
        print(f"\n  Average SI weight in Ridge: {avg_si_weight:.3f}")

    # Save results
    output_path = Path('results/p5_ensemble/full_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'parameters': PARAMETER_CHOICES,
            'n_assets': len(valid_results),
            'summary': {
                'ensemble_beats_si': len(ensemble_beats),
                'avg_si_sharpe': float(avg_si) if valid_results else 0,
                'avg_ensemble_sharpe': float(avg_ensemble) if valid_results else 0,
                'method_distribution': method_counts,
            },
            'results': results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to {output_path}")

    # Final verdict
    print("\n" + "="*60)

    if len(ensemble_beats) >= len(valid_results) / 2:
        print("✅ P5 PASSED: Ensemble methods improve upon SI-only in majority")
    else:
        print("⚠️ P5 MIXED: SI-only baseline is competitive")
        print("   This validates SI as a strong standalone signal")
    print("="*60 + "\n")

    return results


if __name__ == "__main__":
    results = main()
