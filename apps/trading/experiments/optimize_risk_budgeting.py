#!/usr/bin/env python3
"""
OPTIMIZE SI RISK BUDGETING STRATEGY

Testing:
1. Different SI windows (7, 14, 21, 30 days)
2. Different position scaling (linear, quantile, sigmoid)
3. Different ranking lookbacks (30, 60, 90, 120 days)
4. Combining with volatility targeting
5. Regime conditioning (only scale in certain regimes)
6. Position bounds (min/max leverage)
7. Smoothing (exponential vs rolling)
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2
from src.data.loader_v2 import DataLoaderV2, MarketType

TRANSACTION_COSTS = {
    'BTCUSDT': 0.0004,
    'SPY': 0.0002,
    'EURUSD': 0.0001,
}

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def compute_si(data, window=7):
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    population.run(data)
    return population.compute_si_timeseries(data, window=window)

def sharpe_ratio(returns):
    if len(returns) < 10 or returns.std() == 0:
        return 0
    return returns.mean() / returns.std() * np.sqrt(252)

def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum / peak - 1)
    return dd.min()

def calmar_ratio(returns):
    ann_ret = returns.mean() * 252
    mdd = abs(max_drawdown(returns))
    return ann_ret / mdd if mdd > 0 else 0

def sortino_ratio(returns, target=0):
    excess = returns - target
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0
    return excess.mean() / downside.std() * np.sqrt(252)

def apply_costs(returns, positions, cost_rate):
    position_changes = positions.diff().abs()
    costs = position_changes * cost_rate
    return returns - costs

def linear_scaling(si_rank, min_pos=0.5, max_pos=1.5):
    """Linear: position = min + rank * (max - min)"""
    return min_pos + si_rank * (max_pos - min_pos)

def quantile_scaling(si_rank, n_buckets=5, min_pos=0.5, max_pos=1.5):
    """Quantile: discrete buckets"""
    bucket = np.floor(si_rank * n_buckets) / n_buckets
    return min_pos + bucket * (max_pos - min_pos)

def sigmoid_scaling(si_rank, steepness=5, min_pos=0.5, max_pos=1.5):
    """Sigmoid: more aggressive at extremes"""
    centered = si_rank - 0.5
    sigmoid = 1 / (1 + np.exp(-steepness * centered))
    return min_pos + sigmoid * (max_pos - min_pos)

def threshold_scaling(si_rank, low_thresh=0.3, high_thresh=0.7, min_pos=0.5, max_pos=1.5):
    """Threshold: only scale at extremes"""
    position = pd.Series(1.0, index=si_rank.index)
    position[si_rank < low_thresh] = min_pos
    position[si_rank > high_thresh] = max_pos
    return position

def compute_metrics(returns, name=""):
    return {
        'sharpe': float(sharpe_ratio(returns)),
        'sortino': float(sortino_ratio(returns)),
        'calmar': float(calmar_ratio(returns)),
        'max_dd': float(max_drawdown(returns)),
        'ann_return': float(returns.mean() * 252),
        'ann_vol': float(returns.std() * np.sqrt(252)),
    }

def main():
    print("\n" + "="*70)
    print("  OPTIMIZE SI RISK BUDGETING STRATEGY")
    print("="*70)
    print(f"  Time: {datetime.now().isoformat()}")
    
    loader = DataLoaderV2()
    
    assets = {
        'BTCUSDT': (loader.load('BTCUSDT', MarketType.CRYPTO), MarketType.CRYPTO),
        'SPY': (loader.load('SPY', MarketType.STOCKS), MarketType.STOCKS),
        'EURUSD': (loader.load('EURUSD', MarketType.FOREX), MarketType.FOREX),
    }
    
    all_results = {}
    
    # ============================================================
    section("1. OPTIMIZE SI WINDOW")
    # ============================================================
    
    print("\n  Testing SI windows: 7, 14, 21, 30 days")
    
    window_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        cost_rate = TRANSACTION_COSTS[name]
        returns = data['close'].pct_change()
        
        best_sharpe = -999
        best_window = None
        
        for window in [7, 14, 21, 30]:
            si = compute_si(data, window=window)
            
            common = si.index.intersection(returns.dropna().index)
            si_aligned = si.loc[common].dropna()
            returns_aligned = returns.loc[common]
            
            common_final = si_aligned.index.intersection(returns_aligned.dropna().index)
            
            si_rank = si_aligned.loc[common_final].rank(pct=True)
            position = linear_scaling(si_rank)
            
            pos_shifted = position.shift(1)
            ret_aligned = returns_aligned.loc[common_final]
            
            common_strat = pos_shifted.dropna().index.intersection(ret_aligned.dropna().index)
            gross_ret = pos_shifted.loc[common_strat] * ret_aligned.loc[common_strat]
            net_ret = apply_costs(gross_ret, pos_shifted.loc[common_strat], cost_rate)
            
            net_sharpe = sharpe_ratio(net_ret)
            
            if net_sharpe > best_sharpe:
                best_sharpe = net_sharpe
                best_window = window
            
            print(f"    Window {window}d: Net Sharpe = {net_sharpe:.3f}")
        
        window_results[name] = {'best_window': best_window, 'best_sharpe': float(best_sharpe)}
        print(f"    → Best: {best_window}d (Sharpe = {best_sharpe:.3f})")
    
    all_results['window_optimization'] = window_results
    
    # ============================================================
    section("2. OPTIMIZE SCALING FUNCTION")
    # ============================================================
    
    print("\n  Testing: Linear, Quantile, Sigmoid, Threshold")
    
    scaling_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        cost_rate = TRANSACTION_COSTS[name]
        returns = data['close'].pct_change()
        
        best_window = window_results[name]['best_window']
        si = compute_si(data, window=best_window)
        
        common = si.index.intersection(returns.dropna().index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        common_final = si_aligned.index.intersection(returns_aligned.dropna().index)
        si_rank = si_aligned.loc[common_final].rank(pct=True)
        ret_aligned = returns_aligned.loc[common_final]
        
        scaling_funcs = {
            'linear': linear_scaling(si_rank),
            'quantile_5': quantile_scaling(si_rank, n_buckets=5),
            'quantile_3': quantile_scaling(si_rank, n_buckets=3),
            'sigmoid_3': sigmoid_scaling(si_rank, steepness=3),
            'sigmoid_5': sigmoid_scaling(si_rank, steepness=5),
            'threshold': threshold_scaling(si_rank),
        }
        
        best_sharpe = -999
        best_scaling = None
        asset_scaling_results = {}
        
        for scale_name, position in scaling_funcs.items():
            pos_shifted = position.shift(1)
            common_strat = pos_shifted.dropna().index.intersection(ret_aligned.dropna().index)
            
            gross_ret = pos_shifted.loc[common_strat] * ret_aligned.loc[common_strat]
            net_ret = apply_costs(gross_ret, pos_shifted.loc[common_strat], cost_rate)
            
            net_sharpe = sharpe_ratio(net_ret)
            turnover = pos_shifted.diff().abs().mean() * 252
            
            asset_scaling_results[scale_name] = {
                'sharpe': float(net_sharpe),
                'turnover': float(turnover),
            }
            
            if net_sharpe > best_sharpe:
                best_sharpe = net_sharpe
                best_scaling = scale_name
            
            print(f"    {scale_name:<12}: Sharpe = {net_sharpe:.3f}, Turnover = {turnover:.1f}x")
        
        scaling_results[name] = {
            'best_scaling': best_scaling,
            'best_sharpe': float(best_sharpe),
            'all': asset_scaling_results,
        }
        print(f"    → Best: {best_scaling} (Sharpe = {best_sharpe:.3f})")
    
    all_results['scaling_optimization'] = scaling_results
    
    # ============================================================
    section("3. OPTIMIZE RANKING LOOKBACK")
    # ============================================================
    
    print("\n  Testing ranking lookbacks: 30, 60, 90, 120, 252 days")
    
    lookback_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        cost_rate = TRANSACTION_COSTS[name]
        returns = data['close'].pct_change()
        
        best_window = window_results[name]['best_window']
        si = compute_si(data, window=best_window)
        
        common = si.index.intersection(returns.dropna().index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        common_final = si_aligned.index.intersection(returns_aligned.dropna().index)
        ret_aligned = returns_aligned.loc[common_final]
        si_vals = si_aligned.loc[common_final]
        
        best_sharpe = -999
        best_lookback = None
        
        for lookback in [30, 60, 90, 120, 252]:
            si_rank = si_vals.rolling(lookback).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
            position = linear_scaling(si_rank)
            
            pos_shifted = position.shift(1)
            common_strat = pos_shifted.dropna().index.intersection(ret_aligned.dropna().index)
            
            gross_ret = pos_shifted.loc[common_strat] * ret_aligned.loc[common_strat]
            net_ret = apply_costs(gross_ret, pos_shifted.loc[common_strat], cost_rate)
            
            net_sharpe = sharpe_ratio(net_ret)
            
            if net_sharpe > best_sharpe:
                best_sharpe = net_sharpe
                best_lookback = lookback
            
            print(f"    Lookback {lookback}d: Net Sharpe = {net_sharpe:.3f}")
        
        lookback_results[name] = {'best_lookback': best_lookback, 'best_sharpe': float(best_sharpe)}
        print(f"    → Best: {lookback}d (Sharpe = {best_sharpe:.3f})")
    
    all_results['lookback_optimization'] = lookback_results
    
    # ============================================================
    section("4. OPTIMIZE POSITION BOUNDS")
    # ============================================================
    
    print("\n  Testing min/max position bounds")
    
    bounds_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        cost_rate = TRANSACTION_COSTS[name]
        returns = data['close'].pct_change()
        
        best_window = window_results[name]['best_window']
        si = compute_si(data, window=best_window)
        
        common = si.index.intersection(returns.dropna().index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        common_final = si_aligned.index.intersection(returns_aligned.dropna().index)
        si_rank = si_aligned.loc[common_final].rank(pct=True)
        ret_aligned = returns_aligned.loc[common_final]
        
        bound_configs = [
            (0.5, 1.5),
            (0.3, 1.7),
            (0.6, 1.4),
            (0.7, 1.3),
            (0.8, 1.2),
            (0.0, 2.0),
        ]
        
        best_sharpe = -999
        best_bounds = None
        
        for min_pos, max_pos in bound_configs:
            position = linear_scaling(si_rank, min_pos=min_pos, max_pos=max_pos)
            
            pos_shifted = position.shift(1)
            common_strat = pos_shifted.dropna().index.intersection(ret_aligned.dropna().index)
            
            gross_ret = pos_shifted.loc[common_strat] * ret_aligned.loc[common_strat]
            net_ret = apply_costs(gross_ret, pos_shifted.loc[common_strat], cost_rate)
            
            net_sharpe = sharpe_ratio(net_ret)
            max_dd = max_drawdown(net_ret)
            
            if net_sharpe > best_sharpe:
                best_sharpe = net_sharpe
                best_bounds = (min_pos, max_pos)
            
            print(f"    [{min_pos:.1f}, {max_pos:.1f}]: Sharpe = {net_sharpe:.3f}, MaxDD = {max_dd:.1%}")
        
        bounds_results[name] = {'best_bounds': best_bounds, 'best_sharpe': float(best_sharpe)}
        print(f"    → Best: [{best_bounds[0]:.1f}, {best_bounds[1]:.1f}] (Sharpe = {best_sharpe:.3f})")
    
    all_results['bounds_optimization'] = bounds_results
    
    # ============================================================
    section("5. COMBINE WITH VOLATILITY TARGETING")
    # ============================================================
    
    print("\n  Testing SI Risk Budgeting + Volatility Targeting")
    
    vol_target_results = {}
    target_vol = 0.15  # 15% annual
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        cost_rate = TRANSACTION_COSTS[name]
        returns = data['close'].pct_change()
        
        best_window = window_results[name]['best_window']
        si = compute_si(data, window=best_window)
        
        common = si.index.intersection(returns.dropna().index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        common_final = si_aligned.index.intersection(returns_aligned.dropna().index)
        si_rank = si_aligned.loc[common_final].rank(pct=True)
        ret_aligned = returns_aligned.loc[common_final]
        
        # SI-only position
        si_position = linear_scaling(si_rank)
        
        # Realized volatility
        realized_vol = ret_aligned.rolling(20).std() * np.sqrt(252)
        
        # Vol-targeting position
        vol_position = target_vol / (realized_vol + 0.01)
        vol_position = vol_position.clip(0.5, 2.0)
        
        # Combined: SI × Vol target
        combined_position = si_position * vol_position
        combined_position = combined_position.clip(0.3, 2.5)
        
        # SI-only results
        pos_shifted = si_position.shift(1)
        common_strat = pos_shifted.dropna().index.intersection(ret_aligned.dropna().index)
        si_ret = apply_costs(pos_shifted.loc[common_strat] * ret_aligned.loc[common_strat], 
                            pos_shifted.loc[common_strat], cost_rate)
        
        # Combined results
        pos_shifted_c = combined_position.shift(1)
        common_strat_c = pos_shifted_c.dropna().index.intersection(ret_aligned.dropna().index)
        combined_ret = apply_costs(pos_shifted_c.loc[common_strat_c] * ret_aligned.loc[common_strat_c],
                                  pos_shifted_c.loc[common_strat_c], cost_rate)
        
        si_metrics = compute_metrics(si_ret)
        combined_metrics = compute_metrics(combined_ret)
        
        vol_target_results[name] = {
            'si_only': si_metrics,
            'si_plus_vol_target': combined_metrics,
            'improvement': float(combined_metrics['sharpe'] - si_metrics['sharpe']),
        }
        
        print(f"    SI-Only:          Sharpe = {si_metrics['sharpe']:.3f}, MaxDD = {si_metrics['max_dd']:.1%}")
        print(f"    SI + Vol Target:  Sharpe = {combined_metrics['sharpe']:.3f}, MaxDD = {combined_metrics['max_dd']:.1%}")
        imp = combined_metrics['sharpe'] - si_metrics['sharpe']
        print(f"    Improvement: {imp:+.3f}")
    
    all_results['vol_target_combination'] = vol_target_results
    
    # ============================================================
    section("6. ADD SMOOTHING")
    # ============================================================
    
    print("\n  Testing position smoothing (EMA)")
    
    smoothing_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        cost_rate = TRANSACTION_COSTS[name]
        returns = data['close'].pct_change()
        
        best_window = window_results[name]['best_window']
        si = compute_si(data, window=best_window)
        
        common = si.index.intersection(returns.dropna().index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        common_final = si_aligned.index.intersection(returns_aligned.dropna().index)
        si_rank = si_aligned.loc[common_final].rank(pct=True)
        ret_aligned = returns_aligned.loc[common_final]
        
        raw_position = linear_scaling(si_rank)
        
        best_sharpe = -999
        best_halflife = None
        
        for halflife in [1, 3, 5, 7, 10, 15]:
            if halflife == 1:
                smoothed = raw_position
            else:
                smoothed = raw_position.ewm(halflife=halflife).mean()
            
            pos_shifted = smoothed.shift(1)
            common_strat = pos_shifted.dropna().index.intersection(ret_aligned.dropna().index)
            
            gross_ret = pos_shifted.loc[common_strat] * ret_aligned.loc[common_strat]
            net_ret = apply_costs(gross_ret, pos_shifted.loc[common_strat], cost_rate)
            
            net_sharpe = sharpe_ratio(net_ret)
            turnover = pos_shifted.diff().abs().mean() * 252
            
            if net_sharpe > best_sharpe:
                best_sharpe = net_sharpe
                best_halflife = halflife
            
            print(f"    Halflife {halflife}d: Sharpe = {net_sharpe:.3f}, Turnover = {turnover:.1f}x")
        
        smoothing_results[name] = {'best_halflife': best_halflife, 'best_sharpe': float(best_sharpe)}
        print(f"    → Best: {best_halflife}d (Sharpe = {best_sharpe:.3f})")
    
    all_results['smoothing_optimization'] = smoothing_results
    
    # ============================================================
    section("7. FINAL OPTIMIZED STRATEGY")
    # ============================================================
    
    print("\n  Running final optimized strategy with best parameters")
    
    final_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        cost_rate = TRANSACTION_COSTS[name]
        returns = data['close'].pct_change()
        
        # Best parameters
        best_window = window_results[name]['best_window']
        best_bounds = bounds_results[name]['best_bounds']
        best_halflife = smoothing_results[name]['best_halflife']
        
        print(f"    Parameters: window={best_window}d, bounds={best_bounds}, halflife={best_halflife}d")
        
        si = compute_si(data, window=best_window)
        
        common = si.index.intersection(returns.dropna().index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        common_final = si_aligned.index.intersection(returns_aligned.dropna().index)
        si_rank = si_aligned.loc[common_final].rank(pct=True)
        ret_aligned = returns_aligned.loc[common_final]
        
        # Apply optimized strategy
        position = linear_scaling(si_rank, min_pos=best_bounds[0], max_pos=best_bounds[1])
        
        if best_halflife > 1:
            position = position.ewm(halflife=best_halflife).mean()
        
        pos_shifted = position.shift(1)
        common_strat = pos_shifted.dropna().index.intersection(ret_aligned.dropna().index)
        
        gross_ret = pos_shifted.loc[common_strat] * ret_aligned.loc[common_strat]
        net_ret = apply_costs(gross_ret, pos_shifted.loc[common_strat], cost_rate)
        
        # Baseline (constant position)
        baseline_ret = ret_aligned.loc[common_strat]
        
        optimized_metrics = compute_metrics(net_ret)
        baseline_metrics = compute_metrics(baseline_ret)
        
        # Calculate turnover
        turnover = pos_shifted.diff().abs().mean() * 252
        
        final_results[name] = {
            'parameters': {
                'si_window': best_window,
                'min_pos': best_bounds[0],
                'max_pos': best_bounds[1],
                'smoothing_halflife': best_halflife,
            },
            'optimized': optimized_metrics,
            'baseline': baseline_metrics,
            'improvement': {
                'sharpe': float(optimized_metrics['sharpe'] - baseline_metrics['sharpe']),
                'calmar': float(optimized_metrics['calmar'] - baseline_metrics['calmar']),
            },
            'turnover': float(turnover),
        }
        
        print(f"\n    RESULTS:")
        print(f"    {'Metric':<15} {'Baseline':>12} {'Optimized':>12} {'Improvement':>12}")
        print("    " + "-"*55)
        print(f"    {'Sharpe':<15} {baseline_metrics['sharpe']:>12.3f} {optimized_metrics['sharpe']:>12.3f} {optimized_metrics['sharpe'] - baseline_metrics['sharpe']:>+12.3f}")
        print(f"    {'Sortino':<15} {baseline_metrics['sortino']:>12.3f} {optimized_metrics['sortino']:>12.3f} {optimized_metrics['sortino'] - baseline_metrics['sortino']:>+12.3f}")
        print(f"    {'Calmar':<15} {baseline_metrics['calmar']:>12.3f} {optimized_metrics['calmar']:>12.3f} {optimized_metrics['calmar'] - baseline_metrics['calmar']:>+12.3f}")
        print(f"    {'Max DD':<15} {baseline_metrics['max_dd']:>12.1%} {optimized_metrics['max_dd']:>12.1%}")
        print(f"    {'Ann Return':<15} {baseline_metrics['ann_return']:>12.1%} {optimized_metrics['ann_return']:>12.1%}")
        print(f"    {'Ann Vol':<15} {baseline_metrics['ann_vol']:>12.1%} {optimized_metrics['ann_vol']:>12.1%}")
        print(f"    {'Turnover':<15} {'-':>12} {turnover:>12.1f}x")
    
    all_results['final_optimized'] = final_results
    
    # ============================================================
    section("OPTIMIZATION SUMMARY")
    # ============================================================
    
    print("\n  BEST PARAMETERS BY ASSET:")
    print(f"  {'Asset':<10} {'Window':>8} {'Bounds':>12} {'Halflife':>10} {'Net Sharpe':>12}")
    print("  " + "-"*55)
    
    for name, result in final_results.items():
        params = result['parameters']
        sharpe = result['optimized']['sharpe']
        bounds_str = f"[{params['min_pos']:.1f},{params['max_pos']:.1f}]"
        print(f"  {name:<10} {params['si_window']:>8}d {bounds_str:>12} {params['smoothing_halflife']:>10}d {sharpe:>12.3f}")
    
    out_path = Path("results/optimized_risk_budgeting/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2, default=str)
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
