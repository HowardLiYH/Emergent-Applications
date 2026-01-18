#!/usr/bin/env python3
"""
EXTENSIVE SI EXPLORATION
Cover ALL possible angles.
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, pearsonr, entropy, ks_2samp, mannwhitneyu
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2
from src.data.loader_v2 import DataLoaderV2, MarketType

discoveries = []

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def discover(msg):
    discoveries.append(msg)
    print(f"  📍 DISCOVERY: {msg}")

def compute_si(data, window=7):
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    population.run(data)
    return population.compute_si_timeseries(data, window=window)

def main():
    print("\n" + "="*70)
    print("  EXTENSIVE SI EXPLORATION")
    print("="*70)
    
    loader = DataLoaderV2()
    
    # Load all assets
    assets = {
        'BTCUSDT': loader.load('BTCUSDT', MarketType.CRYPTO),
        'ETHUSDT': loader.load('ETHUSDT', MarketType.CRYPTO),
        'SPY': loader.load('SPY', MarketType.STOCKS),
        'QQQ': loader.load('QQQ', MarketType.STOCKS),
    }
    
    # Compute SI for all
    si_dict = {}
    for name, data in assets.items():
        print(f"\n  Computing SI for {name}...")
        si_dict[name] = compute_si(data)
    
    # ============================================================
    section("1. CROSS-ASSET SI CORRELATION")
    # ============================================================
    
    print("\n  Do different assets specialize together?")
    print(f"\n  {'Pair':<20} {'Corr':>10} {'Sig':>8}")
    print("  " + "-"*40)
    
    pairs = [('BTCUSDT', 'ETHUSDT'), ('BTCUSDT', 'SPY'), ('SPY', 'QQQ')]
    for a, b in pairs:
        si_a = si_dict[a]
        si_b = si_dict[b]
        common = si_a.index.intersection(si_b.index)
        if len(common) > 100:
            r, p = spearmanr(si_a.loc[common], si_b.loc[common])
            sig = "✅" if p < 0.05 else ""
            print(f"  {a}-{b:<12} {r:>+10.3f} {sig:>8}")
            if abs(r) > 0.3 and p < 0.05:
                discover(f"Cross-asset SI correlation: {a}-{b} r={r:.3f}")
    
    # ============================================================
    section("2. SI PREDICTS REGIME CHANGES?")
    # ============================================================
    
    print("\n  Does SI change before volatility regime changes?")
    
    for name, data in list(assets.items())[:2]:  # Just BTC and ETH
        returns = data['close'].pct_change()
        vol = returns.rolling(7).std()
        vol_change = vol.diff()
        
        # High vol change = regime shift
        vol_threshold = vol_change.std() * 1.5
        regime_change = abs(vol_change) > vol_threshold
        
        si = si_dict[name]
        si_before = si.shift(3)  # SI 3 days before
        
        common = regime_change.index.intersection(si_before.dropna().index)
        if len(common) > 100:
            si_at_change = si_before.loc[common][regime_change.loc[common]]
            si_at_normal = si_before.loc[common][~regime_change.loc[common]]
            
            if len(si_at_change) > 10 and len(si_at_normal) > 10:
                stat, p = mannwhitneyu(si_at_change.dropna(), si_at_normal.dropna())
                mean_change = si_at_change.mean()
                mean_normal = si_at_normal.mean()
                
                print(f"\n  {name}:")
                print(f"    SI before regime change: {mean_change:.4f}")
                print(f"    SI before normal:        {mean_normal:.4f}")
                print(f"    Difference significant:  {'YES' if p < 0.05 else 'NO'} (p={p:.4f})")
                
                if p < 0.05:
                    discover(f"SI predicts regime changes in {name} (p={p:.4f})")
    
    # ============================================================
    section("3. SI AND VOLUME RELATIONSHIP")
    # ============================================================
    
    print("\n  Does high volume lead to specialization?")
    
    for name, data in list(assets.items())[:2]:
        if 'volume' not in data.columns:
            continue
        
        vol_norm = data['volume'] / data['volume'].rolling(30).mean()
        si = si_dict[name]
        
        common = vol_norm.dropna().index.intersection(si.dropna().index)
        if len(common) > 100:
            r, p = spearmanr(vol_norm.loc[common], si.loc[common])
            print(f"\n  {name}: SI-Volume corr = {r:+.3f} (p={p:.4f})")
            
            if abs(r) > 0.1 and p < 0.05:
                discover(f"SI correlates with volume in {name} (r={r:.3f})")
    
    # ============================================================
    section("4. SI PERSISTENCE (AUTOCORRELATION)")
    # ============================================================
    
    print("\n  How persistent is SI?")
    
    for name, si in si_dict.items():
        autocorrs = [si.autocorr(lag=lag) for lag in [1, 5, 10, 20]]
        print(f"\n  {name}: autocorr lag=[1,5,10,20] = [{', '.join(f'{a:.3f}' for a in autocorrs)}]")
        
        half_life = None
        for lag in range(1, 50):
            if si.autocorr(lag=lag) < 0.5:
                half_life = lag
                break
        
        if half_life:
            print(f"    Half-life: {half_life} days")
            discover(f"SI half-life in {name}: {half_life} days")
    
    # ============================================================
    section("5. SI AND MOMENTUM FACTOR")
    # ============================================================
    
    print("\n  Does SI help time momentum?")
    
    for name, data in list(assets.items())[:2]:
        returns = data['close'].pct_change()
        
        # Momentum return (past 7d return as signal for next day)
        mom_signal = returns.rolling(7).sum()
        mom_return = mom_signal.shift(1) * returns  # Long when past 7d positive
        
        si = si_dict[name]
        common = si.dropna().index.intersection(mom_return.dropna().index)
        
        if len(common) > 100:
            si_aligned = si.loc[common]
            mom_ret_aligned = mom_return.loc[common]
            
            # Compare momentum return in high vs low SI periods
            high_si = si_aligned > si_aligned.median()
            
            mom_high_si = mom_ret_aligned[high_si]
            mom_low_si = mom_ret_aligned[~high_si]
            
            sharpe_high = mom_high_si.mean() / mom_high_si.std() * np.sqrt(252) if mom_high_si.std() > 0 else 0
            sharpe_low = mom_low_si.mean() / mom_low_si.std() * np.sqrt(252) if mom_low_si.std() > 0 else 0
            
            print(f"\n  {name}:")
            print(f"    Momentum Sharpe (high SI): {sharpe_high:.2f}")
            print(f"    Momentum Sharpe (low SI):  {sharpe_low:.2f}")
            
            if sharpe_high > sharpe_low + 0.3:
                discover(f"High SI improves momentum in {name} (Sharpe: {sharpe_high:.2f} vs {sharpe_low:.2f})")
    
    # ============================================================
    section("6. SI QUANTILE ANALYSIS")
    # ============================================================
    
    print("\n  Return distribution by SI quintile:")
    
    for name, data in list(assets.items())[:1]:
        returns = data['close'].pct_change()
        si = si_dict[name]
        
        common = si.dropna().index.intersection(returns.dropna().index)
        si_aligned = si.loc[common]
        ret_aligned = returns.loc[common]
        
        # Future returns (1 day ahead)
        ret_future = ret_aligned.shift(-1)
        
        si_quintile = pd.qcut(si_aligned, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        print(f"\n  {name}:")
        print(f"  {'Quintile':<10} {'Mean Ret':>12} {'Std Ret':>12} {'Sharpe':>10}")
        print("  " + "-"*46)
        
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            mask = si_quintile == q
            qret = ret_future[mask].dropna()
            if len(qret) > 10:
                mean_r = qret.mean() * 100
                std_r = qret.std() * 100
                sharpe = qret.mean() / qret.std() * np.sqrt(252) if qret.std() > 0 else 0
                print(f"  {q:<10} {mean_r:>+12.3f}% {std_r:>12.3f}% {sharpe:>10.2f}")
    
    # ============================================================
    section("7. SI REGIME TRANSITION MATRIX")
    # ============================================================
    
    print("\n  How often does SI switch between low/medium/high?")
    
    for name in ['BTCUSDT']:
        si = si_dict[name]
        
        # Discretize into 3 regimes
        si_regime = pd.qcut(si.dropna(), q=3, labels=['Low', 'Med', 'High'])
        
        # Transition matrix
        transitions = pd.crosstab(si_regime.shift(1), si_regime, normalize='index')
        
        print(f"\n  {name} SI Transition Probabilities:")
        print(transitions.round(2).to_string())
        
        # Persistence
        persistence = np.diag(transitions.values).mean()
        print(f"\n    Average persistence: {persistence:.2%}")
        
        if persistence > 0.5:
            discover(f"SI is persistent in {name} (avg stay probability: {persistence:.2%})")
    
    # ============================================================
    section("8. NOVEL: SI AS MARKET READABILITY")
    # ============================================================
    
    print("\n  Testing SI as 'Market Readability' metric:")
    print("  If SI is high, simple strategies should work better")
    
    for name, data in list(assets.items())[:1]:
        returns = data['close'].pct_change()
        si = si_dict[name]
        
        common = si.dropna().index.intersection(returns.dropna().index)
        si_aligned = si.loc[common]
        ret_aligned = returns.loc[common]
        
        # Simple strategy: follow previous day's return
        strategy_signal = ret_aligned.shift(1)  # Yesterday's return as signal
        strategy_return = np.sign(strategy_signal) * ret_aligned.shift(-1)
        
        high_si = si_aligned > si_aligned.median()
        
        acc_high = (strategy_return[high_si] > 0).mean()
        acc_low = (strategy_return[~high_si] > 0).mean()
        
        print(f"\n  {name}:")
        print(f"    Trend-following accuracy (high SI): {acc_high:.2%}")
        print(f"    Trend-following accuracy (low SI):  {acc_low:.2%}")
        
        if acc_high > acc_low + 0.02:
            discover(f"SI = 'Market Readability': simple strategies work better when SI high ({acc_high:.2%} vs {acc_low:.2%})")
    
    # ============================================================
    section("DISCOVERIES SUMMARY")
    # ============================================================
    
    print(f"\n  Total discoveries: {len(discoveries)}")
    for i, d in enumerate(discoveries, 1):
        print(f"    {i}. {d}")
    
    # Save to discoveries log
    discoveries_path = Path("docs/DISCOVERIES_LOG.md")
    with open(discoveries_path, 'a') as f:
        f.write(f"\n\n## Extensive Exploration - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for d in discoveries:
            f.write(f"- {d}\n")
    
    print(f"\n  Discoveries logged to {discoveries_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
