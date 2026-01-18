#!/usr/bin/env python3
"""
PHASE 2 VALIDATION:

1. Backtest best strategies WITH transaction costs
2. Walk-forward validation of SI-ADX spread strategy
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2
from src.data.loader_v2 import DataLoaderV2, MarketType

# Transaction costs (one-way, in decimal)
TRANSACTION_COSTS = {
    'BTCUSDT': 0.0004,   # 4 bps (crypto)
    'SPY': 0.0002,       # 2 bps (stocks)
    'EURUSD': 0.0001,    # 1 bp (forex)
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

def compute_adx(data):
    """Compute ADX indicator."""
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_dm = (data['high'] - data['high'].shift()).clip(lower=0)
    minus_dm = (data['low'].shift() - data['low']).clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(14).mean()
    return adx

def apply_transaction_costs(returns, positions, cost_rate):
    """Apply transaction costs based on position changes."""
    position_changes = positions.diff().abs()
    costs = position_changes * cost_rate
    net_returns = returns - costs
    return net_returns

def si_adx_spread_strategy(si, adx, returns, lookback=20):
    """SI-ADX spread trading strategy."""
    common = si.index.intersection(adx.dropna().index)
    si_aligned = si.loc[common].dropna()
    adx_aligned = adx.loc[common].dropna()
    
    common_final = si_aligned.index.intersection(adx_aligned.index)
    si_vals = si_aligned.loc[common_final]
    adx_vals = adx_aligned.loc[common_final]
    
    # Normalize
    si_norm = (si_vals - si_vals.mean()) / si_vals.std()
    adx_norm = (adx_vals - adx_vals.mean()) / adx_vals.std()
    
    # Spread
    spread = si_norm - 0.5 * adx_norm
    
    # Rolling z-score
    spread_mean = spread.rolling(lookback).mean()
    spread_std = spread.rolling(lookback).std()
    z_score = (spread - spread_mean) / (spread_std + 1e-10)
    
    # Signal
    position = pd.Series(0.0, index=z_score.index)
    position[z_score > 2] = -1.0   # Short spread
    position[z_score < -2] = 1.0   # Long spread
    
    # Position changes only at thresholds
    position = position.ffill()  # Hold position until next signal
    
    return position, z_score

def risk_budgeting_strategy(si, returns):
    """SI-based risk budgeting."""
    common = si.index.intersection(returns.index)
    si_aligned = si.loc[common].dropna()
    
    si_rank = si_aligned.rank(pct=True)
    position = 0.5 + si_rank  # Range: 0.5 to 1.5
    
    return position

def factor_timing_strategy(si, returns, momentum_lookback=7):
    """SI-based factor timing."""
    common = si.index.intersection(returns.index)
    si_aligned = si.loc[common].dropna()
    returns_aligned = returns.loc[common]
    
    # Momentum signal
    momentum = returns_aligned.rolling(momentum_lookback).sum()
    
    common_final = si_aligned.index.intersection(momentum.dropna().index)
    
    si_vals = si_aligned.loc[common_final]
    mom_vals = momentum.loc[common_final]
    
    si_median = si_vals.median()
    high_si = si_vals > si_median
    
    # High SI: follow momentum; Low SI: mean revert
    position = pd.Series(0.0, index=common_final)
    position[high_si] = np.sign(mom_vals[high_si])
    position[~high_si] = -np.sign(mom_vals[~high_si])
    
    return position

def main():
    print("\n" + "="*70)
    print("  PHASE 2 VALIDATION: Transaction Costs & Walk-Forward")
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
    section("PART 1: BACKTEST WITH TRANSACTION COSTS")
    # ============================================================
    
    print("\n  Testing 3 best strategies with realistic transaction costs")
    print(f"  Costs: BTC=4bps, SPY=2bps, EUR=1bp (one-way)")
    
    cost_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {'='*50}")
        print(f"  {name}")
        print(f"  {'='*50}")
        
        cost_rate = TRANSACTION_COSTS[name]
        si = compute_si(data)
        adx = compute_adx(data)
        returns = data['close'].pct_change()
        
        asset_results = {}
        
        # Strategy 1: SI-ADX Spread
        print(f"\n  Strategy 1: SI-ADX Spread Trading")
        position, z_score = si_adx_spread_strategy(si, adx, returns)
        
        common = position.index.intersection(returns.dropna().index)
        pos_aligned = position.loc[common].shift(1)  # Previous day position
        ret_aligned = returns.loc[common]
        
        common_final = pos_aligned.dropna().index.intersection(ret_aligned.dropna().index)
        
        # Gross returns
        gross_returns = pos_aligned.loc[common_final] * ret_aligned.loc[common_final]
        gross_sharpe = sharpe_ratio(gross_returns)
        
        # Net returns (with costs)
        net_returns = apply_transaction_costs(gross_returns, pos_aligned.loc[common_final], cost_rate)
        net_sharpe = sharpe_ratio(net_returns)
        
        # Turnover
        position_changes = pos_aligned.diff().abs()
        turnover_daily = position_changes.mean()
        turnover_annual = turnover_daily * 252
        
        asset_results['si_adx_spread'] = {
            'gross_sharpe': float(gross_sharpe),
            'net_sharpe': float(net_sharpe),
            'cost_drag': float(gross_sharpe - net_sharpe),
            'turnover_annual': float(turnover_annual),
            'max_dd_net': float(max_drawdown(net_returns)),
        }
        
        print(f"    Gross Sharpe: {gross_sharpe:.2f}")
        print(f"    Net Sharpe:   {net_sharpe:.2f}")
        print(f"    Cost Drag:    {gross_sharpe - net_sharpe:.2f}")
        print(f"    Annual Turnover: {turnover_annual:.1f}x")
        print(f"    Max DD (net): {max_drawdown(net_returns):.1%}")
        
        # Strategy 2: Risk Budgeting
        print(f"\n  Strategy 2: SI Risk Budgeting")
        position_rb = risk_budgeting_strategy(si, returns)
        
        common = position_rb.index.intersection(returns.dropna().index)
        pos_aligned = position_rb.loc[common].shift(1)
        ret_aligned = returns.loc[common]
        
        common_final = pos_aligned.dropna().index.intersection(ret_aligned.dropna().index)
        
        gross_returns = pos_aligned.loc[common_final] * ret_aligned.loc[common_final]
        gross_sharpe = sharpe_ratio(gross_returns)
        
        net_returns = apply_transaction_costs(gross_returns, pos_aligned.loc[common_final], cost_rate)
        net_sharpe = sharpe_ratio(net_returns)
        
        position_changes = pos_aligned.diff().abs()
        turnover_annual = position_changes.mean() * 252
        
        asset_results['risk_budgeting'] = {
            'gross_sharpe': float(gross_sharpe),
            'net_sharpe': float(net_sharpe),
            'cost_drag': float(gross_sharpe - net_sharpe),
            'turnover_annual': float(turnover_annual),
            'max_dd_net': float(max_drawdown(net_returns)),
        }
        
        print(f"    Gross Sharpe: {gross_sharpe:.2f}")
        print(f"    Net Sharpe:   {net_sharpe:.2f}")
        print(f"    Cost Drag:    {gross_sharpe - net_sharpe:.2f}")
        print(f"    Annual Turnover: {turnover_annual:.1f}x")
        
        # Strategy 3: Factor Timing
        print(f"\n  Strategy 3: SI Factor Timing")
        position_ft = factor_timing_strategy(si, returns)
        
        common = position_ft.index.intersection(returns.dropna().index)
        pos_aligned = position_ft.loc[common].shift(1)
        ret_aligned = returns.loc[common]
        
        common_final = pos_aligned.dropna().index.intersection(ret_aligned.dropna().index)
        
        future_ret = ret_aligned.shift(-1)
        common_strat = pos_aligned.loc[common_final].index.intersection(future_ret.dropna().index)
        
        gross_returns = pos_aligned.loc[common_strat] * future_ret.loc[common_strat]
        gross_sharpe = sharpe_ratio(gross_returns)
        
        net_returns = apply_transaction_costs(gross_returns, pos_aligned.loc[common_strat], cost_rate)
        net_sharpe = sharpe_ratio(net_returns)
        
        position_changes = pos_aligned.diff().abs()
        turnover_annual = position_changes.mean() * 252
        
        asset_results['factor_timing'] = {
            'gross_sharpe': float(gross_sharpe),
            'net_sharpe': float(net_sharpe),
            'cost_drag': float(gross_sharpe - net_sharpe),
            'turnover_annual': float(turnover_annual),
            'max_dd_net': float(max_drawdown(net_returns)),
        }
        
        print(f"    Gross Sharpe: {gross_sharpe:.2f}")
        print(f"    Net Sharpe:   {net_sharpe:.2f}")
        print(f"    Cost Drag:    {gross_sharpe - net_sharpe:.2f}")
        print(f"    Annual Turnover: {turnover_annual:.1f}x")
        
        cost_results[name] = asset_results
    
    all_results['transaction_costs'] = cost_results
    
    # ============================================================
    section("PART 2: WALK-FORWARD VALIDATION")
    # ============================================================
    
    print("\n  Testing SI-ADX Spread with rolling out-of-sample windows")
    print("  Window: 252 days training, 63 days testing (quarterly)")
    
    wf_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {'='*50}")
        print(f"  {name} - Walk-Forward Analysis")
        print(f"  {'='*50}")
        
        cost_rate = TRANSACTION_COSTS[name]
        si = compute_si(data)
        adx = compute_adx(data)
        returns = data['close'].pct_change()
        
        # Align all data
        common = si.index.intersection(adx.dropna().index).intersection(returns.dropna().index)
        si_aligned = si.loc[common]
        adx_aligned = adx.loc[common]
        returns_aligned = returns.loc[common]
        
        train_window = 252  # 1 year
        test_window = 63    # 1 quarter
        
        n = len(common)
        n_folds = (n - train_window) // test_window
        
        fold_results = []
        
        print(f"\n  Running {n_folds} walk-forward folds...")
        print(f"  {'Fold':<6} {'Period':<25} {'Gross':>10} {'Net':>10} {'Trades':>8}")
        print("  " + "-"*65)
        
        for fold in range(n_folds):
            train_start = fold * test_window
            train_end = train_start + train_window
            test_start = train_end
            test_end = min(test_start + test_window, n)
            
            if test_end <= test_start:
                break
            
            # Training period - compute lookback parameters
            train_si = si_aligned.iloc[train_start:train_end]
            train_adx = adx_aligned.iloc[train_start:train_end]
            
            # Normalize using training stats
            si_mean = train_si.mean()
            si_std = train_si.std()
            adx_mean = train_adx.mean()
            adx_std = train_adx.std()
            
            # Test period
            test_si = si_aligned.iloc[test_start:test_end]
            test_adx = adx_aligned.iloc[test_start:test_end]
            test_returns = returns_aligned.iloc[test_start:test_end]
            
            # Apply normalization from training
            si_norm = (test_si - si_mean) / si_std
            adx_norm = (test_adx - adx_mean) / adx_std
            
            spread = si_norm - 0.5 * adx_norm
            
            # Use training spread stats for z-score
            train_spread = (train_si - si_mean) / si_std - 0.5 * (train_adx - adx_mean) / adx_std
            spread_mean = train_spread.mean()
            spread_std = train_spread.std()
            
            z_score = (spread - spread_mean) / (spread_std + 1e-10)
            
            # Generate signals
            position = pd.Series(0.0, index=z_score.index)
            position[z_score > 2] = -1.0
            position[z_score < -2] = 1.0
            position = position.ffill().fillna(0)
            
            # Calculate returns
            pos_shifted = position.shift(1).fillna(0)
            gross_ret = pos_shifted * test_returns
            net_ret = apply_transaction_costs(gross_ret, pos_shifted, cost_rate)
            
            fold_gross_sharpe = sharpe_ratio(gross_ret)
            fold_net_sharpe = sharpe_ratio(net_ret)
            n_trades = (position.diff() != 0).sum()
            
            period_str = f"{test_si.index[0].strftime('%Y-%m')} to {test_si.index[-1].strftime('%Y-%m')}"
            
            print(f"  {fold+1:<6} {period_str:<25} {fold_gross_sharpe:>10.2f} {fold_net_sharpe:>10.2f} {n_trades:>8}")
            
            fold_results.append({
                'fold': fold + 1,
                'period': period_str,
                'gross_sharpe': float(fold_gross_sharpe),
                'net_sharpe': float(fold_net_sharpe),
                'n_trades': int(n_trades),
                'n_days': test_end - test_start,
            })
        
        # Summary
        if fold_results:
            gross_sharpes = [f['gross_sharpe'] for f in fold_results]
            net_sharpes = [f['net_sharpe'] for f in fold_results]
            
            avg_gross = np.mean(gross_sharpes)
            avg_net = np.mean(net_sharpes)
            std_net = np.std(net_sharpes)
            pct_positive = sum(1 for s in net_sharpes if s > 0) / len(net_sharpes)
            
            print(f"\n  SUMMARY:")
            print(f"    Average Gross Sharpe: {avg_gross:.2f}")
            print(f"    Average Net Sharpe:   {avg_net:.2f} ± {std_net:.2f}")
            print(f"    % Positive Quarters:  {pct_positive:.0%}")
            print(f"    Best Quarter:         {max(net_sharpes):.2f}")
            print(f"    Worst Quarter:        {min(net_sharpes):.2f}")
            
            wf_results[name] = {
                'n_folds': len(fold_results),
                'folds': fold_results,
                'avg_gross_sharpe': float(avg_gross),
                'avg_net_sharpe': float(avg_net),
                'std_net_sharpe': float(std_net),
                'pct_positive': float(pct_positive),
                'best_quarter': float(max(net_sharpes)),
                'worst_quarter': float(min(net_sharpes)),
            }
    
    all_results['walk_forward'] = wf_results
    
    # ============================================================
    section("FINAL SUMMARY")
    # ============================================================
    
    print("\n  TRANSACTION COST IMPACT:")
    print(f"  {'Asset':<10} {'Strategy':<20} {'Gross':>8} {'Net':>8} {'Drag':>8}")
    print("  " + "-"*58)
    
    for asset, strategies in cost_results.items():
        for strat_name, metrics in strategies.items():
            print(f"  {asset:<10} {strat_name:<20} {metrics['gross_sharpe']:>8.2f} {metrics['net_sharpe']:>8.2f} {metrics['cost_drag']:>8.2f}")
    
    print("\n  WALK-FORWARD VALIDATION (SI-ADX Spread):")
    print(f"  {'Asset':<10} {'Avg Net Sharpe':>15} {'% Positive':>12} {'Verdict':>15}")
    print("  " + "-"*55)
    
    for asset, metrics in wf_results.items():
        verdict = "✅ ROBUST" if metrics['pct_positive'] >= 0.6 else "⚠️ MIXED" if metrics['pct_positive'] >= 0.4 else "❌ FAILS"
        print(f"  {asset:<10} {metrics['avg_net_sharpe']:>15.2f} {metrics['pct_positive']:>12.0%} {verdict:>15}")
    
    out_path = Path("results/phase2_validation/results.json")
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
