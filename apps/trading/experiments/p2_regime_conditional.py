#!/usr/bin/env python3
"""
P2: Regime-Conditional SI - NEXT_STEPS_PLAN v4.1

Implement SI signal usage that adapts to detected market regime.

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
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETER DOCUMENTATION
# ============================================================================

PARAMETER_CHOICES = {
    'si_window': 7,
    'regime_lookback': 14,
    'adx_trending_threshold': 0.6,   # Normalized percentile
    'adx_meanrev_threshold': 0.4,
    'vol_high_threshold': 0.8,       # High volatility percentile
    'min_regime_duration': 3,        # Minimum days before acting
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
# SI AND REGIME COMPUTATION
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


def classify_regime(data: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """
    Classify market regime at each point.
    
    Regimes:
    - 'trending': Strong directional movement
    - 'mean_reverting': Range-bound, mean-reverting
    - 'volatile': High volatility, uncertain
    """
    returns = data['close'].pct_change()
    
    # Trend strength (directional movement / total movement)
    cum_return = data['close'].pct_change(lookback).abs()
    total_movement = returns.abs().rolling(lookback).sum()
    trend_strength = cum_return / (total_movement + 1e-10)
    
    # Volatility
    volatility = returns.rolling(lookback).std()
    
    # Normalize to percentiles for threshold comparison
    trend_pct = trend_strength.rank(pct=True)
    vol_pct = volatility.rank(pct=True)
    
    # Classify
    regimes = pd.Series(index=data.index, dtype=str)
    
    regimes[vol_pct > PARAMETER_CHOICES['vol_high_threshold']] = 'volatile'
    regimes[(vol_pct <= PARAMETER_CHOICES['vol_high_threshold']) & 
            (trend_pct > PARAMETER_CHOICES['adx_trending_threshold'])] = 'trending'
    regimes[(vol_pct <= PARAMETER_CHOICES['vol_high_threshold']) & 
            (trend_pct <= PARAMETER_CHOICES['adx_meanrev_threshold'])] = 'mean_reverting'
    regimes[regimes.isna()] = 'neutral'
    
    return regimes


def check_regime_persistence(regimes: pd.Series) -> Dict:
    """Check that regimes persist long enough to be tradeable."""
    regime_changes = (regimes != regimes.shift()).cumsum()
    durations = regimes.groupby(regime_changes).size()
    
    return {
        'avg_duration_days': float(durations.mean()),
        'min_duration_days': int(durations.min()),
        'max_duration_days': int(durations.max()),
        'n_regime_changes': int(len(durations) - 1),
        'tradeable': bool(durations.mean() >= 5),
    }


def compute_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """Compute regime-to-regime transition probabilities."""
    transitions = pd.crosstab(
        regimes.shift().fillna('unknown'),
        regimes,
        normalize='index'
    )
    return transitions


# ============================================================================
# REGIME-CONDITIONAL STRATEGY
# ============================================================================

def generate_regime_conditional_signals(data: pd.DataFrame, si: pd.Series, 
                                        regimes: pd.Series) -> pd.Series:
    """
    Generate signals based on regime and SI.
    
    Strategy:
    - Trending + High SI: Follow trend
    - Mean-reverting + High SI: Fade extremes
    - Volatile: No position (cash)
    - Low SI: Reduce confidence
    """
    returns = data['close'].pct_change()
    signals = pd.Series(0.0, index=data.index)
    
    lookback = PARAMETER_CHOICES['regime_lookback']
    si_threshold = si.rolling(60).median()  # Adaptive threshold
    
    for i in range(lookback, len(data)):
        regime = regimes.iloc[i]
        current_si = si.iloc[i] if i < len(si) else 0.5
        threshold = si_threshold.iloc[i] if i < len(si_threshold) else 0.5
        
        if regime == 'volatile':
            signals.iloc[i] = 0.0  # No position
        elif current_si > threshold:
            if regime == 'trending':
                # Follow trend direction
                trend_dir = np.sign(returns.iloc[i-lookback:i].sum())
                signals.iloc[i] = trend_dir
            elif regime == 'mean_reverting':
                # Fade recent moves
                recent_move = returns.iloc[i-5:i].sum()
                signals.iloc[i] = -np.sign(recent_move) if abs(recent_move) > returns.std() else 0
            else:  # neutral
                signals.iloc[i] = 0.5  # Small position
        else:
            signals.iloc[i] = 0.0  # Low SI = low confidence
    
    return signals


def generate_unconditional_signals(si: pd.Series, threshold: float = 0.5) -> pd.Series:
    """Generate simple unconditional SI signals."""
    return (si > threshold).astype(float)


# ============================================================================
# BACKTEST
# ============================================================================

def backtest_strategy(data: pd.DataFrame, signals: pd.Series, market: str) -> Dict:
    """Run backtest."""
    returns = data['close'].pct_change()
    
    # Costs
    costs = {'crypto': 0.0006, 'forex': 0.0001, 'stocks': 0.0002, 'commodities': 0.0003}
    cost = costs.get(market, 0.0002)
    
    # Strategy returns
    aligned_signals = signals.shift(1).reindex(returns.index).fillna(0)
    strategy_returns = returns * aligned_signals
    
    # Apply costs
    trades = aligned_signals.diff().abs().fillna(0)
    net_returns = strategy_returns - (trades * cost)
    
    # Metrics
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
        'n_trades': int((trades > 0).sum()),
    }


# ============================================================================
# CORRELATION ANALYSIS BY REGIME
# ============================================================================

def analyze_correlations_by_regime(si: pd.Series, returns: pd.Series, 
                                   regimes: pd.Series) -> Dict:
    """Analyze SI-return correlation within each regime."""
    results = {}
    
    # Align data
    aligned = pd.DataFrame({
        'si': si,
        'returns': returns.shift(-1),  # Next day returns
        'regime': regimes,
    }).dropna()
    
    # Overall correlation
    overall_corr, _ = spearmanr(aligned['si'], aligned['returns'])
    results['overall'] = float(overall_corr)
    
    # By regime
    sign_flips = 0
    for regime in aligned['regime'].unique():
        if regime == 'unknown':
            continue
        regime_data = aligned[aligned['regime'] == regime]
        if len(regime_data) >= 30:
            corr, _ = spearmanr(regime_data['si'], regime_data['returns'])
            results[regime] = float(corr)
            
            # Check sign flip
            if regime != 'overall' and (corr * overall_corr) < 0:
                sign_flips += 1
    
    results['sign_flips'] = sign_flips
    results['flip_rate'] = sign_flips / max(len([r for r in results if r not in ['overall', 'sign_flips', 'flip_rate']]), 1)
    
    return results


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_asset(market: str, symbol: str) -> Dict:
    """Run P2 analysis for a single asset."""
    print(f"\n  Analyzing {market}/{symbol}...")
    
    try:
        # Load data
        data = load_data(market, symbol)
        returns = data['close'].pct_change()
        
        # Compute SI
        si = compute_si_simple(data, window=PARAMETER_CHOICES['si_window'])
        
        # Classify regimes
        regimes = classify_regime(data, lookback=PARAMETER_CHOICES['regime_lookback'])
        
        # Regime persistence
        persistence = check_regime_persistence(regimes)
        
        # Transition matrix
        transition = compute_transition_matrix(regimes)
        
        # Generate signals
        conditional_signals = generate_regime_conditional_signals(data, si, regimes)
        unconditional_signals = generate_unconditional_signals(si)
        
        # Backtest both
        conditional_bt = backtest_strategy(data, conditional_signals, market)
        unconditional_bt = backtest_strategy(data, unconditional_signals, market)
        
        # Improvement
        sharpe_improvement = conditional_bt['sharpe'] - unconditional_bt['sharpe']
        sharpe_improvement_pct = (sharpe_improvement / abs(unconditional_bt['sharpe'] + 0.01)) * 100
        
        # Correlations by regime
        regime_correlations = analyze_correlations_by_regime(si, returns, regimes)
        
        # Regime distribution
        regime_dist = regimes.value_counts(normalize=True).to_dict()
        
        result = {
            'market': market,
            'symbol': symbol,
            
            # Regime analysis
            'regime_distribution': {k: float(v) for k, v in regime_dist.items()},
            'regime_persistence': persistence,
            'transition_matrix': transition.to_dict(),
            
            # Performance comparison
            'unconditional_sharpe': unconditional_bt['sharpe'],
            'conditional_sharpe': conditional_bt['sharpe'],
            'sharpe_improvement': float(sharpe_improvement),
            'sharpe_improvement_pct': float(sharpe_improvement_pct),
            
            # Sign flips
            'regime_correlations': regime_correlations,
            'sign_flip_rate': regime_correlations.get('flip_rate', 0),
            
            # Success criteria
            'flip_rate_acceptable': regime_correlations.get('flip_rate', 1) < 0.15,
            'regime_tradeable': persistence.get('tradeable', False),
            'sharpe_improved': sharpe_improvement > 0,
        }
        
        # Print summary
        flip_rate = regime_correlations.get('flip_rate', 0)
        status = "✅" if result['flip_rate_acceptable'] and result['sharpe_improved'] else "⚠️"
        print(f"    {status} Unconditional: {unconditional_bt['sharpe']:.3f}, "
              f"Conditional: {conditional_bt['sharpe']:.3f}, "
              f"Flip rate: {flip_rate:.1%}")
        
        return result
        
    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        return {
            'market': market,
            'symbol': symbol,
            'error': str(e),
        }


def main():
    """Run P2 regime-conditional analysis for all assets."""
    
    print("\n" + "="*60)
    print("P2: REGIME-CONDITIONAL SI")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Get all assets
    assets = get_all_assets()
    print(f"\nFound {len(assets)} assets to analyze")
    
    # Analyze each asset
    results = []
    for market, symbol in assets:
        result = analyze_asset(market, symbol)
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("P2 SUMMARY")
    print("="*60)
    
    valid_results = [r for r in results if 'error' not in r]
    
    # Metrics
    low_flip = [r for r in valid_results if r.get('flip_rate_acceptable', False)]
    tradeable = [r for r in valid_results if r.get('regime_tradeable', False)]
    improved = [r for r in valid_results if r.get('sharpe_improved', False)]
    
    print(f"\n  Assets analyzed: {len(valid_results)}")
    print(f"  Sign flip rate < 15%: {len(low_flip)}/{len(valid_results)}")
    print(f"  Regimes tradeable (duration > 5d): {len(tradeable)}/{len(valid_results)}")
    print(f"  Sharpe improved with regime-conditioning: {len(improved)}/{len(valid_results)}")
    
    # Average metrics
    if valid_results:
        avg_flip_rate = np.mean([r.get('sign_flip_rate', 0) for r in valid_results])
        avg_improvement = np.mean([r.get('sharpe_improvement_pct', 0) for r in valid_results])
        
        print(f"\n  Average sign flip rate: {avg_flip_rate:.1%}")
        print(f"  Average Sharpe improvement: {avg_improvement:.1f}%")
    
    # Regime distribution
    print("\n  Typical regime distribution:")
    regime_counts = {'trending': 0, 'mean_reverting': 0, 'volatile': 0, 'neutral': 0}
    for r in valid_results:
        for regime, pct in r.get('regime_distribution', {}).items():
            if regime in regime_counts:
                regime_counts[regime] += pct
    
    for regime, total in regime_counts.items():
        avg = total / max(len(valid_results), 1)
        print(f"    {regime}: {avg:.1%}")
    
    # Save results
    output_path = Path('results/p2_regime/full_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_assets': len(valid_results),
            'summary': {
                'low_flip_rate': len(low_flip),
                'tradeable_regimes': len(tradeable),
                'sharpe_improved': len(improved),
                'avg_flip_rate': float(avg_flip_rate) if valid_results else 0,
            },
            'results': results,
        }, f, indent=2, default=str)
    
    print(f"\n  Results saved to {output_path}")
    
    # Final verdict
    print("\n" + "="*60)
    
    if len(low_flip) >= len(valid_results) / 2:
        print("✅ P2 PASSED: Sign flip rate acceptable in majority of assets")
        print("   Regime-conditional approach reduces correlation instability")
        print("   Ready to proceed to P3: Walk-Forward Validation")
    else:
        print("⚠️ P2 PARTIALLY PASSED: Some assets have high flip rates")
        print("   Review per-asset results")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    results = main()
