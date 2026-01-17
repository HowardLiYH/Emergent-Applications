#!/usr/bin/env python3
"""
P1: Backtest with Realistic Costs - NEXT_STEPS_PLAN v4.1

This script runs comprehensive backtest analysis including:
- Cost model with market impact
- Alpha decay analysis
- Turnover analysis
- Structural break test
- Grinold metrics (IC, IR, Breadth)

Author: Yuhao Li, University of Pennsylvania
Date: January 17, 2026
"""

import sys
sys.path.insert(0, '.')

import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETER DOCUMENTATION (Prof. Weber Requirement)
# ============================================================================

PARAMETER_CHOICES = {
    # SI Computation
    'si_window': 7,              # Days for SI rolling window
    'n_agents_per_strategy': 3,  # Agents per strategy type

    # Cost Model
    'costs': {
        'crypto': {'fee': 0.0004, 'slippage': 0.0002},      # 6 bps one-way
        'forex': {'fee': 0.0000, 'slippage': 0.0001},       # 1 bp one-way
        'stocks': {'fee': 0.0001, 'slippage': 0.0001},      # 2 bps one-way
        'commodities': {'fee': 0.0002, 'slippage': 0.0001}, # 3 bps one-way
    },

    # Alpha Decay
    'max_decay_lag': 20,         # Maximum lag for decay analysis

    # Structural Break
    'break_window_pct': 0.25,    # Rolling window as % of data

    # Grinold
    'ic_window': 60,             # Rolling IC window

    # Backtest
    'train_ratio': 0.70,

    # Random seed
    'random_seed': 42,
}

PARAMETER_RATIONALE = {
    'si_window': "7 days balances responsiveness vs stability",
    'costs': "Research-based transaction costs by market",
    'max_decay_lag': "20 days captures medium-term decay",
}

# Set seeds
np.random.seed(PARAMETER_CHOICES['random_seed'])

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(market: str, symbol: str) -> pd.DataFrame:
    """Load data for a specific asset."""
    data_dir = Path('data')

    # Try daily first
    daily_path = data_dir / market / f"{symbol}_1d.csv"
    if daily_path.exists():
        df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        return df

    raise FileNotFoundError(f"Data not found for {market}/{symbol}")


def get_all_assets() -> List[Tuple[str, str]]:
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
# SI COMPUTATION (Simplified for speed)
# ============================================================================

def compute_si_simple(data: pd.DataFrame, window: int = 7) -> pd.Series:
    """
    Compute simplified SI based on strategy performance dispersion.

    This is a proxy for full NichePopulation SI - measures how
    different strategies would perform in different regimes.
    """
    returns = data['close'].pct_change()

    # Strategy proxies
    momentum = returns.rolling(window).mean()
    mean_rev = -returns.rolling(window).mean()  # Opposite of momentum
    volatility = returns.rolling(window).std()

    # Combine into strategy matrix
    strategies = pd.DataFrame({
        'momentum': momentum,
        'mean_reversion': mean_rev,
        'low_vol': -volatility,
    }).dropna()

    # Normalize each strategy
    strategies_norm = strategies.apply(lambda x: (x - x.mean()) / (x.std() + 1e-10))

    # SI = dispersion of strategy returns (higher = more specialized opportunities)
    si = strategies_norm.std(axis=1)

    # Normalize to 0-1 range
    si = (si - si.min()) / (si.max() - si.min() + 1e-10)

    return si


# ============================================================================
# COST MODEL
# ============================================================================

def get_cost(market: str, multiplier: float = 1.0) -> float:
    """Get one-way transaction cost for a market."""
    costs = PARAMETER_CHOICES['costs'].get(market, PARAMETER_CHOICES['costs']['stocks'])
    return (costs['fee'] + costs['slippage']) * multiplier


def apply_costs(returns: pd.Series, positions: pd.Series, market: str,
                cost_multiplier: float = 1.0) -> pd.Series:
    """Apply transaction costs to returns."""
    cost = get_cost(market, cost_multiplier)
    trades = positions.diff().abs().fillna(0)
    costs = trades * cost
    return returns - costs


def compute_turnover(positions: pd.Series) -> Dict:
    """Compute turnover metrics."""
    daily_turnover = positions.diff().abs().fillna(0)

    n_days = len(positions)
    annual_factor = 252 / n_days if n_days > 0 else 1

    annual_turnover = daily_turnover.sum() * annual_factor
    n_trades = (daily_turnover > 0).sum()
    trades_per_year = n_trades * annual_factor

    return {
        'annual_turnover_pct': annual_turnover * 100,
        'trades_per_year': trades_per_year,
        'avg_holding_period_days': 252 / max(trades_per_year, 1),
        'acceptable': annual_turnover < 2.0,  # < 200%
    }


# ============================================================================
# STRATEGY & BACKTEST
# ============================================================================

def generate_si_signals(si: pd.Series, threshold: float = 0.5) -> pd.Series:
    """Generate trading signals based on SI."""
    # Long when SI > threshold (market is "readable")
    signals = (si > threshold).astype(float)
    return signals


def backtest_strategy(data: pd.DataFrame, signals: pd.Series, market: str,
                     cost_multiplier: float = 1.0) -> Dict:
    """Run backtest with transaction costs."""
    returns = data['close'].pct_change()

    # Align signals with returns (signal at t trades at t+1)
    aligned_signals = signals.shift(1).reindex(returns.index).fillna(0)

    # Strategy returns
    strategy_returns = returns * aligned_signals

    # Apply costs
    net_returns = apply_costs(strategy_returns, aligned_signals, market, cost_multiplier)

    # Metrics
    n_days = len(net_returns.dropna())
    annual_factor = 252 / n_days if n_days > 0 else 1

    sharpe = (net_returns.mean() / (net_returns.std() + 1e-10)) * np.sqrt(252)
    total_return = (1 + net_returns).prod() - 1
    annual_return = (1 + total_return) ** annual_factor - 1

    # Drawdown
    cumulative = (1 + net_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    max_dd = drawdown.min()

    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_dd,
        'n_days': n_days,
    }


# ============================================================================
# ALPHA DECAY ANALYSIS
# ============================================================================

def compute_alpha_decay(si: pd.Series, returns: pd.Series, max_lag: int = 20) -> Dict:
    """
    Compute how SI correlation with future returns decays over time.
    """
    decay_curve = []

    for lag in range(1, max_lag + 1):
        # SI at t vs returns at t+lag
        si_lagged = si.iloc[:-lag]
        returns_future = returns.iloc[lag:]

        # Align
        aligned = pd.concat([
            si_lagged.reset_index(drop=True),
            returns_future.reset_index(drop=True)
        ], axis=1).dropna()

        if len(aligned) < 30:
            continue

        corr, pval = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])

        decay_curve.append({
            'lag': lag,
            'correlation': corr,
            'p_value': pval,
            'significant': pval < 0.05,
        })

    if len(decay_curve) == 0:
        return {'error': 'Insufficient data', 'half_life': None, 'tradeable': False}

    df = pd.DataFrame(decay_curve)

    # Compute half-life
    lag1_corr = abs(df['correlation'].iloc[0]) if len(df) > 0 else 0

    if lag1_corr > 0:
        half_corr = lag1_corr / 2
        below_half = df[df['correlation'].abs() < half_corr]
        half_life = below_half['lag'].min() if len(below_half) > 0 else max_lag
    else:
        half_life = 0

    return {
        'decay_curve': df.to_dict('records'),
        'lag1_correlation': df['correlation'].iloc[0] if len(df) > 0 else 0,
        'half_life': int(half_life) if pd.notna(half_life) else None,
        'tradeable': half_life is not None and half_life >= 3,
    }


# ============================================================================
# STRUCTURAL BREAK TEST
# ============================================================================

def test_structural_breaks(si: pd.Series, feature: pd.Series) -> Dict:
    """
    Test for structural breaks in SI-feature relationship.
    """
    # Align
    aligned = pd.concat([si.rename('si'), feature.rename('feature')], axis=1).dropna()

    if len(aligned) < 60:
        return {'error': 'Insufficient data', 'stable': None}

    # Split-half test
    midpoint = len(aligned) // 2
    first_half = aligned.iloc[:midpoint]
    second_half = aligned.iloc[midpoint:]

    corr_first = first_half['si'].corr(first_half['feature'])
    corr_second = second_half['si'].corr(second_half['feature'])

    same_sign = (corr_first > 0) == (corr_second > 0)

    if max(abs(corr_first), abs(corr_second)) > 0:
        magnitude_ratio = min(abs(corr_first), abs(corr_second)) / max(abs(corr_first), abs(corr_second))
    else:
        magnitude_ratio = 0

    # Rolling correlation for visual inspection
    window = max(30, len(aligned) // 4)
    rolling_corr = aligned['si'].rolling(window).corr(aligned['feature'])

    return {
        'first_half_corr': float(corr_first),
        'second_half_corr': float(corr_second),
        'same_sign': bool(same_sign),
        'magnitude_ratio': float(magnitude_ratio),
        'stable': bool(same_sign and magnitude_ratio > 0.5),
        'rolling_corr_std': float(rolling_corr.std()) if not rolling_corr.isna().all() else None,
    }


# ============================================================================
# GRINOLD METRICS
# ============================================================================

def compute_grinold_metrics(si: pd.Series, returns: pd.Series,
                           positions: pd.Series, strategy_returns: pd.Series) -> Dict:
    """
    Compute Grinold metrics: IC, IR, Breadth.
    """
    # 1. Information Coefficient (correlation of signal with next-day returns)
    si_lagged = si.shift(1)
    aligned = pd.concat([si_lagged, returns], axis=1).dropna()

    if len(aligned) < 30:
        return {'error': 'Insufficient data'}

    ic, ic_pval = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])

    # 2. Rolling IC for ICIR
    window = PARAMETER_CHOICES['ic_window']
    rolling_ics = []

    for i in range(window, len(aligned)):
        window_data = aligned.iloc[i-window:i]
        r, _ = spearmanr(window_data.iloc[:, 0], window_data.iloc[:, 1])
        rolling_ics.append(r)

    if len(rolling_ics) > 0:
        icir = np.mean(rolling_ics) / (np.std(rolling_ics) + 1e-10)
    else:
        icir = 0

    # 3. Breadth (trades per year adjusted for autocorrelation)
    n_days = len(positions)
    trades = (positions.diff().abs() > 0).sum()
    trades_per_year = trades * (252 / n_days) if n_days > 0 else 0

    # Autocorrelation adjustment
    signal_autocorr = si.autocorr(lag=1) if len(si) > 1 else 0
    if pd.isna(signal_autocorr):
        signal_autocorr = 0
    independence_factor = max(0.1, 1 - abs(signal_autocorr))
    effective_breadth = trades_per_year * independence_factor

    # 4. Transfer Coefficient (assume 1.0 for unconstrained)
    tc = 1.0

    # 5. Expected IR from Fundamental Law
    expected_ir = abs(ic) * np.sqrt(max(effective_breadth, 1)) * tc

    # 6. Realized IR
    if strategy_returns.std() > 0:
        realized_ir = (strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252))
    else:
        realized_ir = 0

    return {
        'information_coefficient': float(ic),
        'ic_t_stat': float(ic / (1 / np.sqrt(len(aligned)))),
        'ic_significant': bool(ic_pval < 0.05),
        'icir': float(icir),
        'breadth_raw': float(trades_per_year),
        'breadth_effective': float(effective_breadth),
        'expected_ir': float(expected_ir),
        'realized_ir': float(realized_ir),
        'ir_ratio': float(realized_ir / expected_ir) if expected_ir > 0 else 0,
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_asset(market: str, symbol: str) -> Dict:
    """Run full P1 analysis for a single asset."""
    print(f"\n  Analyzing {market}/{symbol}...")

    try:
        # Load data
        data = load_data(market, symbol)

        # Compute SI
        si = compute_si_simple(data, window=PARAMETER_CHOICES['si_window'])

        # Generate signals
        signals = generate_si_signals(si, threshold=0.5)

        # Compute returns
        returns = data['close'].pct_change()

        # 1. Cost sensitivity analysis
        cost_results = {}
        for multiplier in [0.5, 1.0, 2.0, 3.0]:
            bt = backtest_strategy(data, signals, market, cost_multiplier=multiplier)
            cost_results[f'{multiplier}x'] = bt

        # 2. Turnover analysis
        turnover = compute_turnover(signals)

        # 3. Alpha decay
        alpha_decay = compute_alpha_decay(si, returns, max_lag=PARAMETER_CHOICES['max_decay_lag'])

        # 4. Structural break (SI vs returns)
        structural_break = test_structural_breaks(si, returns)

        # 5. Grinold metrics
        strategy_returns = returns * signals.shift(1)
        net_returns = apply_costs(strategy_returns, signals, market)
        grinold = compute_grinold_metrics(si, returns, signals, net_returns)

        # Summary
        base_sharpe = cost_results['1.0x']['sharpe']

        result = {
            'market': market,
            'symbol': symbol,
            'n_observations': len(data),
            'date_range': f"{data.index[0]} to {data.index[-1]}",

            # Cost analysis
            'cost_sensitivity': cost_results,
            'net_sharpe_1x': cost_results['1.0x']['sharpe'],
            'net_sharpe_2x': cost_results['2.0x']['sharpe'],
            'profitable_at_1x': cost_results['1.0x']['sharpe'] > 0,
            'profitable_at_2x': cost_results['2.0x']['sharpe'] > 0,

            # Turnover
            'turnover': turnover,
            'turnover_acceptable': turnover['acceptable'],

            # Alpha decay
            'alpha_decay': {
                'half_life': alpha_decay.get('half_life'),
                'lag1_correlation': alpha_decay.get('lag1_correlation'),
                'tradeable': alpha_decay.get('tradeable', False),
            },

            # Structural break
            'structural_break': structural_break,
            'relationship_stable': structural_break.get('stable', False),

            # Grinold
            'grinold': grinold,
            'ic': grinold.get('information_coefficient', 0),
            'expected_ir': grinold.get('expected_ir', 0),

            # Overall assessment
            'passes_all_gates': (
                cost_results['1.0x']['sharpe'] > 0 and
                turnover['acceptable'] and
                alpha_decay.get('tradeable', False) and
                structural_break.get('stable', False) and
                abs(grinold.get('information_coefficient', 0)) > 0.03
            ),
        }

        # Print summary
        status = "✅" if result['passes_all_gates'] else "⚠️"
        print(f"    {status} Sharpe: {base_sharpe:.3f}, IC: {result['ic']:.3f}, "
              f"Half-life: {alpha_decay.get('half_life', 'N/A')}, "
              f"Stable: {structural_break.get('stable', 'N/A')}")

        return result

    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        return {
            'market': market,
            'symbol': symbol,
            'error': str(e),
            'passes_all_gates': False,
        }


def main():
    """Run P1 analysis for all assets."""

    print("\n" + "="*60)
    print("P1: BACKTEST WITH REALISTIC COSTS")
    print("NEXT_STEPS_PLAN v4.1")
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
    print("P1 SUMMARY")
    print("="*60)

    # Count passes
    valid_results = [r for r in results if 'error' not in r]
    passed = [r for r in valid_results if r['passes_all_gates']]
    profitable_1x = [r for r in valid_results if r.get('profitable_at_1x', False)]
    profitable_2x = [r for r in valid_results if r.get('profitable_at_2x', False)]

    print(f"\n  Assets analyzed: {len(valid_results)}")
    print(f"  Profitable at 1x costs: {len(profitable_1x)}/{len(valid_results)}")
    print(f"  Profitable at 2x costs: {len(profitable_2x)}/{len(valid_results)}")
    print(f"  Pass all gates: {len(passed)}/{len(valid_results)}")

    # Gate breakdown
    print("\n  Gate breakdown:")
    tradeable = [r for r in valid_results if r.get('alpha_decay', {}).get('tradeable', False)]
    stable = [r for r in valid_results if r.get('relationship_stable', False)]
    good_ic = [r for r in valid_results if abs(r.get('ic', 0)) > 0.03]
    good_turnover = [r for r in valid_results if r.get('turnover_acceptable', False)]

    print(f"    Alpha half-life ≥ 3 days: {len(tradeable)}/{len(valid_results)}")
    print(f"    Relationship stable: {len(stable)}/{len(valid_results)}")
    print(f"    IC > 0.03: {len(good_ic)}/{len(valid_results)}")
    print(f"    Turnover acceptable: {len(good_turnover)}/{len(valid_results)}")

    # By market
    print("\n  By market:")
    for market in ['crypto', 'forex', 'stocks', 'commodities']:
        market_results = [r for r in valid_results if r.get('market') == market]
        market_passed = [r for r in market_results if r['passes_all_gates']]
        if market_results:
            avg_sharpe = np.mean([r.get('net_sharpe_1x', 0) for r in market_results])
            print(f"    {market}: {len(market_passed)}/{len(market_results)} pass, avg Sharpe: {avg_sharpe:.3f}")

    # Save results
    output_path = Path('results/p1_backtest/full_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_assets': len(valid_results),
            'n_passed': len(passed),
            'summary': {
                'profitable_1x': len(profitable_1x),
                'profitable_2x': len(profitable_2x),
                'tradeable_halflife': len(tradeable),
                'stable_relationship': len(stable),
                'good_ic': len(good_ic),
            },
            'results': results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to {output_path}")

    # Final verdict
    print("\n" + "="*60)

    # Success criteria from plan
    markets_profitable = sum(1 for m in ['crypto', 'forex', 'stocks', 'commodities']
                            if any(r.get('profitable_at_1x', False)
                                  for r in valid_results if r.get('market') == m))

    if markets_profitable >= 2 and len(profitable_1x) > 0:
        print("✅ P1 PASSED: Profitable in 2+ markets")
        print("   Ready to proceed to P1.5: Factor Regression")
    else:
        print("⚠️ P1 PARTIALLY PASSED: Review results before proceeding")
    print("="*60 + "\n")

    return results


if __name__ == "__main__":
    results = main()
