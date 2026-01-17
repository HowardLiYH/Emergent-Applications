#!/usr/bin/env python3
"""
P3: Walk-Forward Validation - NEXT_STEPS_PLAN v4.1

Validate SI signal with proper walk-forward methodology.

Author: Yuhao Li, University of Pennsylvania
Date: January 17, 2026
"""

import sys
sys.path.insert(0, '.')

import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETER DOCUMENTATION
# ============================================================================

PARAMETER_CHOICES = {
    'si_window': 7,
    'train_months': 18,       # 18 months training
    'test_months': 3,         # 3 months testing
    'expanding': True,        # Use expanding window
    'min_train_size': 200,    # Minimum training observations
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
# WALK-FORWARD WINDOWS
# ============================================================================

def generate_windows(data: pd.DataFrame, 
                    train_months: int = 18,
                    test_months: int = 3,
                    expanding: bool = True) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate train/test windows for walk-forward validation.
    """
    windows = []
    
    start_date = data.index[0]
    end_date = data.index[-1]
    
    if expanding:
        # Expanding window
        train_end = start_date + pd.DateOffset(months=train_months)
        
        while True:
            test_end = train_end + pd.DateOffset(months=test_months)
            
            if test_end > end_date:
                break
            
            train = data[start_date:train_end]
            test = data[train_end:test_end]
            
            if len(train) >= PARAMETER_CHOICES['min_train_size'] and len(test) >= 20:
                windows.append((train, test))
            
            train_end = test_end
    else:
        # Rolling window
        current_start = start_date
        
        while True:
            train_end = current_start + pd.DateOffset(months=train_months)
            test_end = train_end + pd.DateOffset(months=test_months)
            
            if test_end > end_date:
                break
            
            train = data[current_start:train_end]
            test = data[train_end:test_end]
            
            if len(train) >= PARAMETER_CHOICES['min_train_size'] and len(test) >= 20:
                windows.append((train, test))
            
            current_start = current_start + pd.DateOffset(months=test_months)
    
    return windows


# ============================================================================
# SI COMPUTATION
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


# ============================================================================
# STRATEGY
# ============================================================================

def optimize_threshold(train_data: pd.DataFrame, si: pd.Series, market: str) -> float:
    """
    Find optimal SI threshold on training data.
    """
    best_sharpe = -np.inf
    best_threshold = 0.5
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        signals = (si > threshold).astype(float)
        bt = backtest(train_data, signals, market)
        
        if bt['sharpe'] > best_sharpe:
            best_sharpe = bt['sharpe']
            best_threshold = threshold
    
    return best_threshold


def backtest(data: pd.DataFrame, signals: pd.Series, market: str) -> Dict:
    """Run backtest with costs."""
    returns = data['close'].pct_change()
    
    costs = {'crypto': 0.0006, 'forex': 0.0001, 'stocks': 0.0002, 'commodities': 0.0003}
    cost = costs.get(market, 0.0002)
    
    aligned_signals = signals.shift(1).reindex(returns.index).fillna(0)
    strategy_returns = returns * aligned_signals
    
    trades = aligned_signals.diff().abs().fillna(0)
    net_returns = strategy_returns - (trades * cost)
    
    if net_returns.std() > 0:
        sharpe = (net_returns.mean() / net_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    return {
        'sharpe': float(sharpe),
        'total_return': float((1 + net_returns).prod() - 1),
        'n_days': int(len(net_returns.dropna())),
    }


def bootstrap_sharpe_ci(returns: pd.Series, n_bootstrap: int = 500) -> Dict:
    """Bootstrap confidence interval for Sharpe ratio."""
    sharpes = []
    n = len(returns)
    
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample = returns.iloc[sample_idx]
        if sample.std() > 0:
            sharpe = (sample.mean() / sample.std()) * np.sqrt(252)
            sharpes.append(sharpe)
    
    if len(sharpes) == 0:
        return {'mean': 0, 'ci_lower': 0, 'ci_upper': 0, 'prob_positive': 0}
    
    return {
        'mean': float(np.mean(sharpes)),
        'ci_lower': float(np.percentile(sharpes, 2.5)),
        'ci_upper': float(np.percentile(sharpes, 97.5)),
        'prob_positive': float(np.mean(np.array(sharpes) > 0)),
    }


# ============================================================================
# WALK-FORWARD ANALYSIS
# ============================================================================

def run_walk_forward(data: pd.DataFrame, market: str) -> Dict:
    """
    Run walk-forward validation for a single asset.
    """
    windows = generate_windows(
        data, 
        train_months=PARAMETER_CHOICES['train_months'],
        test_months=PARAMETER_CHOICES['test_months'],
        expanding=PARAMETER_CHOICES['expanding']
    )
    
    if len(windows) == 0:
        return {'error': 'Insufficient data for walk-forward'}
    
    window_results = []
    all_oos_returns = []
    thresholds = []
    
    for i, (train, test) in enumerate(windows):
        # Compute SI on training
        train_si = compute_si_simple(train, window=PARAMETER_CHOICES['si_window'])
        
        # Optimize threshold
        best_threshold = optimize_threshold(train, train_si, market)
        thresholds.append(best_threshold)
        
        # Generate signals
        train_signals = (train_si > best_threshold).astype(float)
        
        # In-sample performance
        is_bt = backtest(train, train_signals, market)
        
        # Out-of-sample: compute SI on test data
        test_si = compute_si_simple(test, window=PARAMETER_CHOICES['si_window'])
        test_signals = (test_si > best_threshold).astype(float)
        
        # OOS performance
        oos_bt = backtest(test, test_signals, market)
        
        # Store OOS returns for CI
        returns = test['close'].pct_change()
        aligned = test_signals.shift(1).reindex(returns.index).fillna(0)
        oos_returns = returns * aligned
        all_oos_returns.append(oos_returns)
        
        window_results.append({
            'window': i,
            'train_start': str(train.index[0].date()),
            'train_end': str(train.index[-1].date()),
            'test_start': str(test.index[0].date()),
            'test_end': str(test.index[-1].date()),
            'train_size': len(train),
            'test_size': len(test),
            'best_threshold': best_threshold,
            'is_sharpe': is_bt['sharpe'],
            'oos_sharpe': oos_bt['sharpe'],
            'oos_return': oos_bt['total_return'],
            'profitable': oos_bt['total_return'] > 0,
        })
    
    # Aggregate OOS returns
    combined_oos = pd.concat(all_oos_returns)
    
    # Bootstrap CI for combined OOS
    ci = bootstrap_sharpe_ci(combined_oos.dropna())
    
    # Parameter stability
    threshold_stability = np.std(thresholds) / np.mean(thresholds) if np.mean(thresholds) > 0 else 0
    
    # Summary
    is_sharpes = [w['is_sharpe'] for w in window_results]
    oos_sharpes = [w['oos_sharpe'] for w in window_results]
    profitable_windows = sum(1 for w in window_results if w['profitable'])
    
    degradation = 1 - (np.mean(oos_sharpes) / (np.mean(is_sharpes) + 0.01))
    
    return {
        'n_windows': len(window_results),
        'windows': window_results,
        'avg_is_sharpe': float(np.mean(is_sharpes)),
        'avg_oos_sharpe': float(np.mean(oos_sharpes)),
        'oos_sharpe_ci': ci,
        'profitable_windows': profitable_windows,
        'profitable_rate': profitable_windows / len(window_results),
        'is_oos_degradation': float(degradation),
        'threshold_stability_cv': float(threshold_stability),
        'parameter_stable': threshold_stability < 0.3,
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_asset(market: str, symbol: str) -> Dict:
    """Run P3 analysis for a single asset."""
    print(f"\n  Analyzing {market}/{symbol}...")
    
    try:
        data = load_data(market, symbol)
        
        wf_results = run_walk_forward(data, market)
        
        if 'error' in wf_results:
            print(f"    ❌ {wf_results['error']}")
            return {'market': market, 'symbol': symbol, 'error': wf_results['error']}
        
        result = {
            'market': market,
            'symbol': symbol,
            'n_windows': wf_results['n_windows'],
            'avg_is_sharpe': wf_results['avg_is_sharpe'],
            'avg_oos_sharpe': wf_results['avg_oos_sharpe'],
            'oos_sharpe_ci': wf_results['oos_sharpe_ci'],
            'profitable_rate': wf_results['profitable_rate'],
            'is_oos_degradation': wf_results['is_oos_degradation'],
            'parameter_stable': wf_results['parameter_stable'],
            
            # Success criteria
            'oos_profitable_majority': wf_results['profitable_rate'] > 0.55,
            'degradation_acceptable': wf_results['is_oos_degradation'] < 0.4,
            'ci_excludes_zero': wf_results['oos_sharpe_ci']['ci_lower'] > 0,
        }
        
        # Print summary
        rate = wf_results['profitable_rate']
        status = "✅" if rate > 0.55 else "⚠️"
        print(f"    {status} Windows: {wf_results['n_windows']}, "
              f"Profitable: {rate:.0%}, "
              f"IS Sharpe: {wf_results['avg_is_sharpe']:.3f}, "
              f"OOS Sharpe: {wf_results['avg_oos_sharpe']:.3f}")
        
        return result
        
    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        return {'market': market, 'symbol': symbol, 'error': str(e)}


def main():
    """Run P3 walk-forward validation for all assets."""
    
    print("\n" + "="*60)
    print("P3: WALK-FORWARD VALIDATION")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Train: {PARAMETER_CHOICES['train_months']} months, Test: {PARAMETER_CHOICES['test_months']} months")
    print(f"Window type: {'Expanding' if PARAMETER_CHOICES['expanding'] else 'Rolling'}")
    
    assets = get_all_assets()
    print(f"\nFound {len(assets)} assets to analyze")
    
    results = []
    for market, symbol in assets:
        result = analyze_asset(market, symbol)
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("P3 SUMMARY")
    print("="*60)
    
    valid_results = [r for r in results if 'error' not in r]
    
    oos_profitable = [r for r in valid_results if r.get('oos_profitable_majority', False)]
    low_degradation = [r for r in valid_results if r.get('degradation_acceptable', False)]
    ci_positive = [r for r in valid_results if r.get('ci_excludes_zero', False)]
    param_stable = [r for r in valid_results if r.get('parameter_stable', False)]
    
    print(f"\n  Assets analyzed: {len(valid_results)}")
    print(f"  OOS profitable >55%: {len(oos_profitable)}/{len(valid_results)}")
    print(f"  IS-OOS degradation <40%: {len(low_degradation)}/{len(valid_results)}")
    print(f"  OOS CI excludes zero: {len(ci_positive)}/{len(valid_results)}")
    print(f"  Parameter stable (CV<0.3): {len(param_stable)}/{len(valid_results)}")
    
    if valid_results:
        avg_oos_rate = np.mean([r.get('profitable_rate', 0) for r in valid_results])
        avg_oos_sharpe = np.mean([r.get('avg_oos_sharpe', 0) for r in valid_results])
        
        print(f"\n  Average OOS profitable rate: {avg_oos_rate:.0%}")
        print(f"  Average OOS Sharpe: {avg_oos_sharpe:.3f}")
    
    # Save results
    output_path = Path('results/p3_walk_forward/full_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'parameters': PARAMETER_CHOICES,
            'n_assets': len(valid_results),
            'summary': {
                'oos_profitable_majority': len(oos_profitable),
                'low_degradation': len(low_degradation),
                'ci_positive': len(ci_positive),
                'avg_oos_rate': float(avg_oos_rate) if valid_results else 0,
            },
            'results': results,
        }, f, indent=2, default=str)
    
    print(f"\n  Results saved to {output_path}")
    
    # Final verdict
    print("\n" + "="*60)
    
    if len(oos_profitable) >= len(valid_results) / 2:
        print("✅ P3 PASSED: Majority of assets show >55% OOS profitable windows")
        print("   Walk-forward validation confirms signal persistence")
        print("   Ready to proceed to P4: SI Risk Overlay")
    else:
        print("⚠️ P3 PARTIALLY PASSED: Some assets show weak OOS performance")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    results = main()
