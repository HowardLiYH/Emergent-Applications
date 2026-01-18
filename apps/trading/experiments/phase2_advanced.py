#!/usr/bin/env python3
"""
PHASE 2 ADVANCED: More SI Applications

Building on successful Phase 2 Part 1 findings:
- SI-ADX spread trading (Sharpe 0.7-1.3)
- Factor timing improvements
- Regime detection

New Applications:
1. SI Momentum (trading SI changes)
2. Cross-Asset SI Arbitrage
3. Multi-Timeframe SI
4. SI-Based Risk Budgeting
5. SI Mean Reversion Strategy
6. SI Breakout Strategy
7. Ensemble of SI Strategies
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

applications = []

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def result(msg):
    applications.append(msg)
    print(f"  ✅ {msg}")

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

def main():
    print("\n" + "="*70)
    print("  PHASE 2 ADVANCED: MORE SI APPLICATIONS")
    print("="*70)
    print(f"  Time: {datetime.now().isoformat()}")
    
    loader = DataLoaderV2()
    
    assets = {
        'BTCUSDT': (loader.load('BTCUSDT', MarketType.CRYPTO), MarketType.CRYPTO),
        'SPY': (loader.load('SPY', MarketType.STOCKS), MarketType.STOCKS),
        'EURUSD': (loader.load('EURUSD', MarketType.FOREX), MarketType.FOREX),
    }
    
    all_results = {}
    si_cache = {}
    
    print("\n  Pre-computing SI for all assets...")
    for name, (data, mtype) in assets.items():
        si_cache[name] = compute_si(data)
        print(f"    {name}: {len(si_cache[name])} SI values")
    
    # ============================================================
    section("APP 9: SI MOMENTUM STRATEGY")
    # ============================================================
    
    print("\n  Trading SI changes: rising SI = bullish, falling SI = bearish")
    
    si_mom_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = si_cache[name]
        returns = data['close'].pct_change()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        # SI momentum signal
        si_change = si_aligned.diff()
        si_signal = np.sign(si_change)  # +1 if SI rising, -1 if falling
        
        common_final = si_signal.dropna().index.intersection(returns_aligned.dropna().index)
        
        # Next day return
        future_ret = returns_aligned.shift(-1)
        common_ret = si_signal.loc[common_final].index.intersection(future_ret.dropna().index)
        
        signal = si_signal.loc[common_ret]
        ret = future_ret.loc[common_ret]
        
        # Strategy returns
        strat_returns = signal * ret
        
        strat_sharpe = sharpe_ratio(strat_returns)
        strat_mdd = max_drawdown(strat_returns)
        win_rate = (strat_returns > 0).mean()
        
        # Benchmark: buy and hold
        bh_sharpe = sharpe_ratio(ret)
        
        si_mom_results[name] = {
            'sharpe': float(strat_sharpe),
            'max_dd': float(strat_mdd),
            'win_rate': float(win_rate),
            'benchmark_sharpe': float(bh_sharpe),
        }
        
        print(f"    SI Momentum Sharpe: {strat_sharpe:.2f} (B&H: {bh_sharpe:.2f})")
        print(f"    Win Rate: {win_rate:.1%}")
        print(f"    Max Drawdown: {strat_mdd:.1%}")
        
        if strat_sharpe > 0.3:
            result(f"SI momentum works in {name} (Sharpe={strat_sharpe:.2f})")
    
    all_results['si_momentum'] = si_mom_results
    
    # ============================================================
    section("APP 10: CROSS-ASSET SI DIVERGENCE")
    # ============================================================
    
    print("\n  Trading when same-market asset SIs diverge")
    
    # For this we need pairs of same-market assets
    # BTC-ETH, SPY-QQQ would be ideal but we only have one per market
    # So we'll look at cross-asset SI correlation instead
    
    cross_results = {}
    
    asset_names = list(si_cache.keys())
    for i, name1 in enumerate(asset_names):
        for name2 in asset_names[i+1:]:
            si1 = si_cache[name1]
            si2 = si_cache[name2]
            
            common = si1.index.intersection(si2.index)
            si1_aligned = si1.loc[common].dropna()
            si2_aligned = si2.loc[common].dropna()
            
            common_final = si1_aligned.index.intersection(si2_aligned.index)
            
            if len(common_final) < 100:
                continue
            
            s1 = si1_aligned.loc[common_final]
            s2 = si2_aligned.loc[common_final]
            
            # Normalize
            s1_norm = (s1 - s1.mean()) / s1.std()
            s2_norm = (s2 - s2.mean()) / s2.std()
            
            # Spread
            spread = s1_norm - s2_norm
            spread_z = (spread - spread.rolling(20).mean()) / (spread.rolling(20).std() + 1e-10)
            
            # Mean reversion signal
            signal = pd.Series(0, index=spread_z.index)
            signal[spread_z > 2] = -1  # Short spread when too high
            signal[spread_z < -2] = 1  # Long spread when too low
            
            # Trade asset1 returns (simplification)
            ret1 = assets[name1][0]['close'].pct_change()
            common_ret = signal.index.intersection(ret1.index)
            signal_aligned = signal.loc[common_ret].shift(1)
            ret_aligned = ret1.loc[common_ret]
            
            common_strat = signal_aligned.dropna().index.intersection(ret_aligned.dropna().index)
            strat_returns = signal_aligned.loc[common_strat] * ret_aligned.loc[common_strat]
            
            pair_key = f"{name1}-{name2}"
            cross_results[pair_key] = {
                'correlation': float(spearmanr(s1, s2)[0]),
                'sharpe': float(sharpe_ratio(strat_returns)),
                'n_trades': int((signal != 0).sum()),
            }
            
            print(f"\n  {pair_key}:")
            print(f"    SI correlation: {spearmanr(s1, s2)[0]:.3f}")
            print(f"    Divergence strategy Sharpe: {sharpe_ratio(strat_returns):.2f}")
            print(f"    Number of trades: {(signal != 0).sum()}")
    
    all_results['cross_asset'] = cross_results
    
    # ============================================================
    section("APP 11: SI MEAN REVERSION STRATEGY")
    # ============================================================
    
    print("\n  Using SI's 4-5 day mean reversion for trading")
    
    si_mr_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = si_cache[name]
        returns = data['close'].pct_change()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        # Z-score of SI
        si_mean = si_aligned.rolling(20).mean()
        si_std = si_aligned.rolling(20).std()
        si_z = (si_aligned - si_mean) / (si_std + 1e-10)
        
        # Mean reversion signal: expect SI to revert to mean
        # If SI is high (z > 1), expect it to fall → bearish for asset?
        # If SI is low (z < -1), expect it to rise → bullish?
        
        # Actually based on our findings: High SI = clearer market = better trends
        # So we trade WITH SI direction, expecting continuation short-term
        
        signal = pd.Series(0, index=si_z.index)
        signal[si_z > 1] = 1   # High SI, expect good trends
        signal[si_z < -1] = -1  # Low SI, expect choppy
        
        future_ret = returns_aligned.shift(-1)
        common_final = signal.index.intersection(future_ret.dropna().index)
        
        strat_returns = signal.loc[common_final] * future_ret.loc[common_final]
        
        si_mr_results[name] = {
            'sharpe': float(sharpe_ratio(strat_returns)),
            'max_dd': float(max_drawdown(strat_returns)),
            'win_rate': float((strat_returns > 0).mean()),
        }
        
        print(f"    SI Mean-Rev Sharpe: {sharpe_ratio(strat_returns):.2f}")
        print(f"    Win Rate: {(strat_returns > 0).mean():.1%}")
        
        if sharpe_ratio(strat_returns) > 0.3:
            result(f"SI mean reversion works in {name} (Sharpe={sharpe_ratio(strat_returns):.2f})")
    
    all_results['si_mean_reversion'] = si_mr_results
    
    # ============================================================
    section("APP 12: SI BREAKOUT STRATEGY")
    # ============================================================
    
    print("\n  Trading SI breakouts (new highs/lows)")
    
    si_breakout_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = si_cache[name]
        returns = data['close'].pct_change()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        # Rolling high/low
        lookback = 20
        si_high = si_aligned.rolling(lookback).max()
        si_low = si_aligned.rolling(lookback).min()
        
        # Breakout signals
        signal = pd.Series(0, index=si_aligned.index)
        signal[si_aligned >= si_high] = 1  # SI at new high → bullish
        signal[si_aligned <= si_low] = -1  # SI at new low → bearish
        
        future_ret = returns_aligned.shift(-1)
        common_final = signal.dropna().index.intersection(future_ret.dropna().index)
        
        strat_returns = signal.loc[common_final] * future_ret.loc[common_final]
        
        si_breakout_results[name] = {
            'sharpe': float(sharpe_ratio(strat_returns)),
            'max_dd': float(max_drawdown(strat_returns)),
            'win_rate': float((strat_returns > 0).mean()),
            'n_signals': int((signal != 0).sum()),
        }
        
        print(f"    SI Breakout Sharpe: {sharpe_ratio(strat_returns):.2f}")
        print(f"    Win Rate: {(strat_returns > 0).mean():.1%}")
        print(f"    Number of signals: {(signal != 0).sum()}")
        
        if sharpe_ratio(strat_returns) > 0.3:
            result(f"SI breakout works in {name} (Sharpe={sharpe_ratio(strat_returns):.2f})")
    
    all_results['si_breakout'] = si_breakout_results
    
    # ============================================================
    section("APP 13: SI-BASED RISK BUDGETING")
    # ============================================================
    
    print("\n  Allocating risk budget based on SI level")
    
    risk_budget_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = si_cache[name]
        returns = data['close'].pct_change()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        common_final = si_aligned.index.intersection(returns_aligned.dropna().index)
        
        si_vals = si_aligned.loc[common_final]
        ret_vals = returns_aligned.loc[common_final]
        
        # Strategy: Scale position by SI quantile
        si_rank = si_vals.rank(pct=True)
        position_size = 0.5 + si_rank  # Range: 0.5 to 1.5
        
        scaled_returns = ret_vals * position_size
        
        # Benchmark: constant position
        constant_sharpe = sharpe_ratio(ret_vals)
        scaled_sharpe = sharpe_ratio(scaled_returns)
        
        # Risk-adjusted
        constant_calmar = calmar_ratio(ret_vals)
        scaled_calmar = calmar_ratio(scaled_returns)
        
        risk_budget_results[name] = {
            'constant_sharpe': float(constant_sharpe),
            'si_scaled_sharpe': float(scaled_sharpe),
            'constant_calmar': float(constant_calmar),
            'si_scaled_calmar': float(scaled_calmar),
            'sharpe_improvement': float(scaled_sharpe - constant_sharpe),
        }
        
        print(f"    Constant: Sharpe={constant_sharpe:.2f}, Calmar={constant_calmar:.2f}")
        print(f"    SI-Scaled: Sharpe={scaled_sharpe:.2f}, Calmar={scaled_calmar:.2f}")
        
        if scaled_sharpe > constant_sharpe:
            result(f"SI risk budgeting improves {name} ({constant_sharpe:.2f} → {scaled_sharpe:.2f})")
    
    all_results['risk_budgeting'] = risk_budget_results
    
    # ============================================================
    section("APP 14: ENSEMBLE OF SI STRATEGIES")
    # ============================================================
    
    print("\n  Combining multiple SI strategies")
    
    ensemble_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = si_cache[name]
        returns = data['close'].pct_change()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        # Strategy 1: SI momentum
        si_change = si_aligned.diff()
        signal1 = np.sign(si_change)
        
        # Strategy 2: SI z-score
        si_z = (si_aligned - si_aligned.rolling(20).mean()) / (si_aligned.rolling(20).std() + 1e-10)
        signal2 = pd.Series(0, index=si_z.index)
        signal2[si_z > 1] = 1
        signal2[si_z < -1] = -1
        
        # Strategy 3: SI breakout
        si_high = si_aligned.rolling(20).max()
        signal3 = pd.Series(0, index=si_aligned.index)
        signal3[si_aligned >= si_high] = 1
        
        # Ensemble: majority vote
        common_all = signal1.dropna().index.intersection(signal2.dropna().index).intersection(signal3.dropna().index)
        
        ensemble_signal = (signal1.loc[common_all] + signal2.loc[common_all] + signal3.loc[common_all]) / 3
        ensemble_signal = np.sign(ensemble_signal)  # Majority direction
        
        future_ret = returns_aligned.shift(-1)
        common_final = ensemble_signal.index.intersection(future_ret.dropna().index)
        
        ensemble_returns = ensemble_signal.loc[common_final] * future_ret.loc[common_final]
        
        # Individual strategy returns
        strat1_returns = signal1.loc[common_final] * future_ret.loc[common_final]
        strat2_returns = signal2.loc[common_final] * future_ret.loc[common_final]
        strat3_returns = signal3.loc[common_final] * future_ret.loc[common_final]
        
        ensemble_results[name] = {
            'momentum_sharpe': float(sharpe_ratio(strat1_returns)),
            'zscore_sharpe': float(sharpe_ratio(strat2_returns)),
            'breakout_sharpe': float(sharpe_ratio(strat3_returns)),
            'ensemble_sharpe': float(sharpe_ratio(ensemble_returns)),
            'ensemble_max_dd': float(max_drawdown(ensemble_returns)),
        }
        
        print(f"    Individual Sharpes: Mom={sharpe_ratio(strat1_returns):.2f}, Z={sharpe_ratio(strat2_returns):.2f}, Break={sharpe_ratio(strat3_returns):.2f}")
        print(f"    Ensemble Sharpe: {sharpe_ratio(ensemble_returns):.2f}")
        print(f"    Ensemble Max DD: {max_drawdown(ensemble_returns):.1%}")
        
        best_individual = max(sharpe_ratio(strat1_returns), sharpe_ratio(strat2_returns), sharpe_ratio(strat3_returns))
        if sharpe_ratio(ensemble_returns) > best_individual:
            result(f"SI ensemble outperforms individuals in {name} (Sharpe={sharpe_ratio(ensemble_returns):.2f})")
    
    all_results['ensemble'] = ensemble_results
    
    # ============================================================
    section("APP 15: SI VOLATILITY TARGETING")
    # ============================================================
    
    print("\n  Using SI to target constant volatility")
    
    vol_target_results = {}
    target_vol = 0.15 / np.sqrt(252)  # 15% annual
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = si_cache[name]
        returns = data['close'].pct_change()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        # Realized volatility
        realized_vol = returns_aligned.rolling(20).std()
        
        # SI-predicted volatility (inverse relationship)
        # Higher SI → lower vol → larger position
        si_normalized = (si_aligned - si_aligned.min()) / (si_aligned.max() - si_aligned.min() + 1e-10)
        predicted_vol_factor = 1.5 - si_normalized  # Range: 0.5 to 1.5
        
        common_final = si_aligned.index.intersection(realized_vol.dropna().index)
        
        # Standard vol targeting
        standard_leverage = target_vol / (realized_vol.loc[common_final] + 1e-10)
        standard_leverage = standard_leverage.clip(0.1, 3)
        
        # SI-enhanced vol targeting
        si_vol = realized_vol.loc[common_final] * predicted_vol_factor.loc[common_final]
        si_leverage = target_vol / (si_vol + 1e-10)
        si_leverage = si_leverage.clip(0.1, 3)
        
        ret_vals = returns_aligned.loc[common_final]
        
        standard_returns = ret_vals * standard_leverage
        si_returns = ret_vals * si_leverage
        
        vol_target_results[name] = {
            'unscaled_vol': float(ret_vals.std() * np.sqrt(252)),
            'standard_vol_target_vol': float(standard_returns.std() * np.sqrt(252)),
            'si_vol_target_vol': float(si_returns.std() * np.sqrt(252)),
            'standard_sharpe': float(sharpe_ratio(standard_returns)),
            'si_sharpe': float(sharpe_ratio(si_returns)),
        }
        
        print(f"    Unscaled annual vol: {ret_vals.std() * np.sqrt(252):.1%}")
        print(f"    Standard vol-target: {standard_returns.std() * np.sqrt(252):.1%} (Sharpe={sharpe_ratio(standard_returns):.2f})")
        print(f"    SI vol-target: {si_returns.std() * np.sqrt(252):.1%} (Sharpe={sharpe_ratio(si_returns):.2f})")
        
        if sharpe_ratio(si_returns) > sharpe_ratio(standard_returns):
            result(f"SI vol targeting outperforms standard in {name}")
    
    all_results['vol_targeting'] = vol_target_results
    
    # ============================================================
    section("PHASE 2 ADVANCED SUMMARY")
    # ============================================================
    
    print(f"\n  Total successful applications: {len(applications)}")
    for i, app in enumerate(applications, 1):
        print(f"    {i}. {app}")
    
    out_path = Path("results/phase2_advanced/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'applications': applications,
            'results': all_results,
        }, f, indent=2, default=str)
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
