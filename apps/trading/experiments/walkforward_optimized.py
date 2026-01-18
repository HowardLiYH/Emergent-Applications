#!/usr/bin/env python3
"""
WALK-FORWARD VALIDATION: Optimized SI Risk Budgeting

Test the optimized parameters with quarterly out-of-sample windows.
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
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

# Optimized parameters from previous analysis
OPTIMAL_PARAMS = {
    'BTCUSDT': {'si_window': 7, 'min_pos': 0.0, 'max_pos': 2.0, 'halflife': 3},
    'SPY': {'si_window': 7, 'min_pos': 0.8, 'max_pos': 1.2, 'halflife': 15},
    'EURUSD': {'si_window': 21, 'min_pos': 0.0, 'max_pos': 2.0, 'halflife': 1},
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

def apply_costs(returns, positions, cost_rate):
    position_changes = positions.diff().abs()
    costs = position_changes * cost_rate
    return returns - costs

def linear_scaling(si_rank, min_pos, max_pos):
    return min_pos + si_rank * (max_pos - min_pos)

def run_walkforward(name, data, params, cost_rate, train_days=252, test_days=63):
    """
    Run walk-forward validation with optimized parameters.
    """
    returns = data['close'].pct_change()
    
    # Pre-compute SI
    si = compute_si(data, window=params['si_window'])
    
    # Align data
    common = si.index.intersection(returns.dropna().index)
    si_aligned = si.loc[common].dropna()
    returns_aligned = returns.loc[common]
    
    common_final = si_aligned.index.intersection(returns_aligned.dropna().index)
    n = len(common_final)
    
    n_folds = (n - train_days) // test_days
    
    fold_results = []
    all_oos_returns = []
    
    print(f"\n  Running {n_folds} walk-forward folds...")
    print(f"  {'Fold':<6} {'Period':<25} {'Gross':>10} {'Net':>10} {'MaxDD':>10}")
    print("  " + "-"*65)
    
    for fold in range(n_folds):
        train_start = fold * test_days
        train_end = train_start + train_days
        test_start = train_end
        test_end = min(test_start + test_days, n)
        
        if test_end <= test_start:
            break
        
        # Get data for this fold
        train_idx = common_final[train_start:train_end]
        test_idx = common_final[test_start:test_end]
        
        # Use training data to compute ranking statistics
        train_si = si_aligned.loc[train_idx]
        train_mean = train_si.mean()
        train_std = train_si.std()
        
        # Test data
        test_si = si_aligned.loc[test_idx]
        test_returns = returns_aligned.loc[test_idx]
        
        # Compute SI rank using expanding window (to avoid look-ahead)
        # Use percentile rank relative to training distribution
        test_si_rank = test_si.apply(lambda x: (x - train_mean) / train_std)
        test_si_rank = test_si_rank.clip(-3, 3)  # Clip extreme z-scores
        test_si_rank = (test_si_rank + 3) / 6  # Convert to 0-1 range
        
        # Apply scaling
        position = linear_scaling(test_si_rank, params['min_pos'], params['max_pos'])
        
        # Apply smoothing
        if params['halflife'] > 1:
            position = position.ewm(halflife=params['halflife']).mean()
        
        # Calculate returns
        pos_shifted = position.shift(1).fillna(1.0)
        
        gross_ret = pos_shifted * test_returns
        net_ret = apply_costs(gross_ret, pos_shifted, cost_rate)
        
        fold_gross_sharpe = sharpe_ratio(gross_ret)
        fold_net_sharpe = sharpe_ratio(net_ret)
        fold_max_dd = max_drawdown(net_ret)
        
        # Store OOS returns
        all_oos_returns.append(net_ret)
        
        period_str = f"{test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')}"
        
        print(f"  {fold+1:<6} {period_str:<25} {fold_gross_sharpe:>10.2f} {fold_net_sharpe:>10.2f} {fold_max_dd:>10.1%}")
        
        fold_results.append({
            'fold': fold + 1,
            'period': period_str,
            'gross_sharpe': float(fold_gross_sharpe),
            'net_sharpe': float(fold_net_sharpe),
            'max_dd': float(fold_max_dd),
            'n_days': test_end - test_start,
        })
    
    # Combine all OOS returns
    combined_oos = pd.concat(all_oos_returns)
    
    return fold_results, combined_oos

def main():
    print("\n" + "="*70)
    print("  WALK-FORWARD VALIDATION: Optimized SI Risk Budgeting")
    print("="*70)
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Train: 252 days | Test: 63 days (quarterly)")
    
    loader = DataLoaderV2()
    
    assets = {
        'BTCUSDT': (loader.load('BTCUSDT', MarketType.CRYPTO), MarketType.CRYPTO),
        'SPY': (loader.load('SPY', MarketType.STOCKS), MarketType.STOCKS),
        'EURUSD': (loader.load('EURUSD', MarketType.FOREX), MarketType.FOREX),
    }
    
    all_results = {}
    
    for name, (data, mtype) in assets.items():
        section(f"{name} - Walk-Forward Analysis")
        
        params = OPTIMAL_PARAMS[name]
        cost_rate = TRANSACTION_COSTS[name]
        
        print(f"\n  Optimized Parameters:")
        print(f"    SI Window: {params['si_window']}d")
        print(f"    Position Bounds: [{params['min_pos']}, {params['max_pos']}]")
        print(f"    Smoothing Halflife: {params['halflife']}d")
        
        fold_results, combined_oos = run_walkforward(name, data, params, cost_rate)
        
        # Summary statistics
        if fold_results:
            gross_sharpes = [f['gross_sharpe'] for f in fold_results]
            net_sharpes = [f['net_sharpe'] for f in fold_results]
            max_dds = [f['max_dd'] for f in fold_results]
            
            avg_gross = np.mean(gross_sharpes)
            avg_net = np.mean(net_sharpes)
            std_net = np.std(net_sharpes)
            pct_positive = sum(1 for s in net_sharpes if s > 0) / len(net_sharpes)
            
            # Combined OOS metrics
            combined_sharpe = sharpe_ratio(combined_oos)
            combined_return = combined_oos.mean() * 252
            combined_vol = combined_oos.std() * np.sqrt(252)
            combined_dd = max_drawdown(combined_oos)
            
            print(f"\n  QUARTERLY SUMMARY:")
            print(f"    Average Gross Sharpe: {avg_gross:.2f}")
            print(f"    Average Net Sharpe:   {avg_net:.2f} ± {std_net:.2f}")
            print(f"    % Positive Quarters:  {pct_positive:.0%}")
            print(f"    Best Quarter:         {max(net_sharpes):.2f}")
            print(f"    Worst Quarter:        {min(net_sharpes):.2f}")
            
            print(f"\n  COMBINED OOS PERFORMANCE:")
            print(f"    Sharpe Ratio:    {combined_sharpe:.2f}")
            print(f"    Annual Return:   {combined_return:.1%}")
            print(f"    Annual Vol:      {combined_vol:.1%}")
            print(f"    Max Drawdown:    {combined_dd:.1%}")
            
            # Verdict
            if pct_positive >= 0.6:
                verdict = "✅ ROBUST"
            elif pct_positive >= 0.5:
                verdict = "⚠️ MARGINAL"
            else:
                verdict = "❌ FAILS"
            
            print(f"\n  VERDICT: {verdict}")
            
            all_results[name] = {
                'parameters': params,
                'n_folds': len(fold_results),
                'folds': fold_results,
                'quarterly': {
                    'avg_gross_sharpe': float(avg_gross),
                    'avg_net_sharpe': float(avg_net),
                    'std_net_sharpe': float(std_net),
                    'pct_positive': float(pct_positive),
                    'best_quarter': float(max(net_sharpes)),
                    'worst_quarter': float(min(net_sharpes)),
                },
                'combined_oos': {
                    'sharpe': float(combined_sharpe),
                    'annual_return': float(combined_return),
                    'annual_vol': float(combined_vol),
                    'max_dd': float(combined_dd),
                },
                'verdict': verdict,
            }
    
    # ============================================================
    section("FINAL SUMMARY")
    # ============================================================
    
    print("\n  WALK-FORWARD VALIDATION RESULTS:")
    print(f"  {'Asset':<10} {'Avg Net Sharpe':>15} {'% Positive':>12} {'Combined OOS':>15} {'Verdict':>12}")
    print("  " + "-"*70)
    
    for name, result in all_results.items():
        q = result['quarterly']
        c = result['combined_oos']
        print(f"  {name:<10} {q['avg_net_sharpe']:>15.2f} {q['pct_positive']:>12.0%} {c['sharpe']:>15.2f} {result['verdict']:>12}")
    
    print("\n  COMPARISON: Optimized vs Baseline Walk-Forward")
    print(f"  {'Asset':<10} {'Baseline %+':>12} {'Optimized %+':>12} {'Improvement':>12}")
    print("  " + "-"*50)
    
    # Baseline from previous validation (approximate)
    baseline_pct = {'BTCUSDT': 0.42, 'SPY': 0.33, 'EURUSD': 0.31}
    
    for name, result in all_results.items():
        opt_pct = result['quarterly']['pct_positive']
        base_pct = baseline_pct.get(name, 0.5)
        improvement = opt_pct - base_pct
        print(f"  {name:<10} {base_pct:>12.0%} {opt_pct:>12.0%} {improvement:>+12.0%}")
    
    out_path = Path("results/walkforward_optimized/results.json")
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
