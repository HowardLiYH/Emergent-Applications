#!/usr/bin/env python3
"""
P4: SI Risk Overlay - NEXT_STEPS_PLAN v4.1

Use SI as a position sizing overlay, not just a directional signal.
Compare SI sizing vs inverse volatility vs constant sizing.

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

warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETER DOCUMENTATION
# ============================================================================

PARAMETER_CHOICES = {
    'si_window': 7,
    'vol_window': 20,
    'target_vol': 0.15,        # 15% annualized target volatility
    'max_leverage': 2.0,       # Maximum position size
    'base_direction': 'long',  # Base strategy direction
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
# POSITION SIZING METHODS
# ============================================================================

def constant_sizing(data: pd.DataFrame) -> pd.Series:
    """Constant position size = 1.0"""
    return pd.Series(1.0, index=data.index)


def inverse_vol_sizing(data: pd.DataFrame, 
                       vol_window: int = 20,
                       target_vol: float = 0.15) -> pd.Series:
    """
    Inverse volatility sizing (risk parity).
    Position = target_vol / realized_vol
    """
    returns = data['close'].pct_change()
    rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
    
    # Target vol / realized vol
    sizing = target_vol / rolling_vol.clip(lower=0.05)  # Floor vol at 5%
    
    # Cap at max leverage
    sizing = sizing.clip(upper=PARAMETER_CHOICES['max_leverage'])
    
    return sizing


def si_sizing(data: pd.DataFrame, si: pd.Series) -> pd.Series:
    """
    SI-based position sizing.
    High SI = high position, Low SI = low position
    """
    # Normalize SI to [0.5, 2.0] range
    sizing = 0.5 + si * 1.5
    
    # Cap at max leverage
    sizing = sizing.clip(upper=PARAMETER_CHOICES['max_leverage'])
    
    return sizing


def hybrid_sizing(data: pd.DataFrame, si: pd.Series,
                  vol_window: int = 20, target_vol: float = 0.15) -> pd.Series:
    """
    Hybrid: Inverse vol × SI
    Uses vol for risk control and SI for conviction
    """
    inv_vol = inverse_vol_sizing(data, vol_window, target_vol)
    si_component = 0.5 + si * 1.0  # [0.5, 1.5]
    
    sizing = inv_vol * si_component
    sizing = sizing.clip(upper=PARAMETER_CHOICES['max_leverage'])
    
    return sizing


# ============================================================================
# BACKTEST
# ============================================================================

def backtest_sizing(data: pd.DataFrame, sizing: pd.Series, market: str) -> Dict:
    """Backtest with given position sizing."""
    returns = data['close'].pct_change()
    
    # Costs
    costs = {'crypto': 0.0006, 'forex': 0.0001, 'stocks': 0.0002, 'commodities': 0.0003}
    cost = costs.get(market, 0.0002)
    
    # Strategy: always long with varying position size
    aligned_sizing = sizing.shift(1).reindex(returns.index).fillna(1.0)
    strategy_returns = returns * aligned_sizing
    
    # Costs based on position changes
    position_changes = aligned_sizing.diff().abs().fillna(0)
    net_returns = strategy_returns - (position_changes * cost)
    
    # Metrics
    ann_return = net_returns.mean() * 252
    ann_vol = net_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Drawdown
    cumulative = (1 + net_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    max_dd = drawdown.min()
    
    # Calmar ratio
    calmar = -ann_return / max_dd if max_dd < 0 else 0
    
    # Average position size
    avg_size = aligned_sizing.mean()
    
    return {
        'sharpe': float(sharpe),
        'annual_return': float(ann_return),
        'annual_vol': float(ann_vol),
        'max_drawdown': float(max_dd),
        'calmar': float(calmar),
        'avg_position_size': float(avg_size),
        'total_return': float((1 + net_returns).prod() - 1),
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_asset(market: str, symbol: str) -> Dict:
    """Run P4 analysis for a single asset."""
    print(f"\n  Analyzing {market}/{symbol}...")
    
    try:
        data = load_data(market, symbol)
        si = compute_si_simple(data, window=PARAMETER_CHOICES['si_window'])
        
        # Generate sizing for each method
        sizing_methods = {
            'constant': constant_sizing(data),
            'inverse_vol': inverse_vol_sizing(data, PARAMETER_CHOICES['vol_window'], PARAMETER_CHOICES['target_vol']),
            'si_only': si_sizing(data, si),
            'hybrid': hybrid_sizing(data, si, PARAMETER_CHOICES['vol_window'], PARAMETER_CHOICES['target_vol']),
        }
        
        # Backtest each
        results = {}
        for method, sizing in sizing_methods.items():
            bt = backtest_sizing(data, sizing, market)
            results[method] = bt
        
        # Compare to constant (baseline)
        baseline = results['constant']
        
        improvements = {}
        for method in ['inverse_vol', 'si_only', 'hybrid']:
            improvements[method] = {
                'sharpe_improvement': results[method]['sharpe'] - baseline['sharpe'],
                'dd_improvement': baseline['max_drawdown'] - results[method]['max_drawdown'],  # More negative is worse
                'calmar_improvement': results[method]['calmar'] - baseline['calmar'],
            }
        
        # Best method
        sharpe_by_method = {m: r['sharpe'] for m, r in results.items()}
        best_method = max(sharpe_by_method, key=sharpe_by_method.get)
        
        result = {
            'market': market,
            'symbol': symbol,
            'results': results,
            'improvements': improvements,
            'best_method': best_method,
            
            # Key comparisons
            'si_beats_constant': results['si_only']['sharpe'] > results['constant']['sharpe'],
            'si_beats_invol': results['si_only']['sharpe'] > results['inverse_vol']['sharpe'],
            'hybrid_best': best_method == 'hybrid',
            'dd_reduction_pct': improvements['si_only']['dd_improvement'] / abs(baseline['max_drawdown']) * 100 if baseline['max_drawdown'] < 0 else 0,
        }
        
        # Print summary
        status = "✅" if result['si_beats_constant'] else "⚠️"
        print(f"    {status} Constant: {baseline['sharpe']:.3f}, SI: {results['si_only']['sharpe']:.3f}, "
              f"InvVol: {results['inverse_vol']['sharpe']:.3f}, Hybrid: {results['hybrid']['sharpe']:.3f}")
        
        return result
        
    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        return {'market': market, 'symbol': symbol, 'error': str(e)}


def main():
    """Run P4 risk overlay analysis for all assets."""
    
    print("\n" + "="*60)
    print("P4: SI RISK OVERLAY")
    print("Comparing SI sizing vs Inverse Vol vs Constant")
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
    print("P4 SUMMARY")
    print("="*60)
    
    valid_results = [r for r in results if 'error' not in r]
    
    si_beats_const = [r for r in valid_results if r.get('si_beats_constant', False)]
    si_beats_invol = [r for r in valid_results if r.get('si_beats_invol', False)]
    hybrid_best = [r for r in valid_results if r.get('hybrid_best', False)]
    
    print(f"\n  Assets analyzed: {len(valid_results)}")
    print(f"  SI sizing beats Constant: {len(si_beats_const)}/{len(valid_results)}")
    print(f"  SI sizing beats Inverse Vol: {len(si_beats_invol)}/{len(valid_results)}")
    print(f"  Hybrid is best method: {len(hybrid_best)}/{len(valid_results)}")
    
    if valid_results:
        avg_dd_reduction = np.mean([r.get('dd_reduction_pct', 0) for r in valid_results])
        print(f"\n  Average DD reduction with SI: {avg_dd_reduction:.1f}%")
        
        # Method comparison table
        print("\n  Average Sharpe by Method:")
        for method in ['constant', 'inverse_vol', 'si_only', 'hybrid']:
            sharpes = [r['results'][method]['sharpe'] for r in valid_results]
            print(f"    {method:15s}: {np.mean(sharpes):.3f}")
    
    # Best method distribution
    method_counts = {'constant': 0, 'inverse_vol': 0, 'si_only': 0, 'hybrid': 0}
    for r in valid_results:
        method = r.get('best_method', 'unknown')
        if method in method_counts:
            method_counts[method] += 1
    
    print("\n  Best method distribution:")
    for method, count in method_counts.items():
        print(f"    {method}: {count}/{len(valid_results)} ({count/len(valid_results)*100:.0f}%)")
    
    # Save results
    output_path = Path('results/p4_risk_overlay/full_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'parameters': PARAMETER_CHOICES,
            'n_assets': len(valid_results),
            'summary': {
                'si_beats_constant': len(si_beats_const),
                'si_beats_invol': len(si_beats_invol),
                'hybrid_best': len(hybrid_best),
                'method_distribution': method_counts,
            },
            'results': results,
        }, f, indent=2, default=str)
    
    print(f"\n  Results saved to {output_path}")
    
    # Final verdict
    print("\n" + "="*60)
    
    if len(si_beats_const) >= len(valid_results) / 2:
        print("✅ P4 PASSED: SI sizing beats constant sizing in majority")
        print("   SI provides value as a risk overlay signal")
        print("   Ready to proceed to P5: Ensemble")
    else:
        print("⚠️ P4 PARTIALLY PASSED: SI sizing shows mixed results")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    results = main()
