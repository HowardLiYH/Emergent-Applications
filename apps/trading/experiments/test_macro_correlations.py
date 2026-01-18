#!/usr/bin/env python3
"""
Test SI correlations with macro features.
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

def load_macro_features():
    """Load all downloaded macro features."""
    macro_dir = Path("data/macro")
    macro_data = {}
    
    for filepath in macro_dir.glob("*_1d.csv"):
        name = filepath.stem.replace("_1d", "")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if 'close' in df.columns:
            macro_data[name] = df['close']
        elif 'Close' in df.columns:
            macro_data[name] = df['Close']
    
    return macro_data

def compute_si(data, window=7):
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    population.run(data)
    return population.compute_si_timeseries(data, window=window)

def main():
    print("\n" + "="*70)
    print("  SI vs MACRO FEATURES CORRELATION")
    print("="*70)
    
    # Load macro data
    macro_data = load_macro_features()
    print(f"\n  Loaded {len(macro_data)} macro indicators")
    
    # Compute SI for multiple assets
    loader = DataLoaderV2()
    assets = [
        ('BTCUSDT', MarketType.CRYPTO),
        ('SPY', MarketType.STOCKS),
        ('EURUSD', MarketType.FOREX),
    ]
    
    results = {}
    
    for symbol, mtype in assets:
        print(f"\n  Analyzing {symbol}...")
        data = loader.load(symbol, mtype)
        si = compute_si(data)
        
        asset_results = {}
        
        for macro_name, macro_series in macro_data.items():
            # Align indices
            common = si.index.intersection(macro_series.index)
            if len(common) < 100:
                continue
            
            si_aligned = si.loc[common]
            macro_aligned = macro_series.loc[common]
            
            # Compute correlations
            # Level
            r_level, p_level = spearmanr(si_aligned, macro_aligned)
            
            # Returns/changes
            si_diff = si_aligned.diff().dropna()
            macro_diff = macro_aligned.diff().dropna()
            common_diff = si_diff.index.intersection(macro_diff.index)
            r_diff, p_diff = spearmanr(si_diff.loc[common_diff], macro_diff.loc[common_diff])
            
            asset_results[macro_name] = {
                'level_r': float(r_level),
                'level_p': float(p_level),
                'diff_r': float(r_diff),
                'diff_p': float(p_diff),
            }
        
        results[symbol] = asset_results
    
    # Summary
    print("\n" + "="*70)
    print("  RESULTS: SI vs MACRO CORRELATIONS")
    print("="*70)
    
    print(f"\n  {'Macro Feature':<15} {'BTC':>10} {'SPY':>10} {'EUR':>10} {'Consistent':>12}")
    print("  " + "-"*60)
    
    discoveries = []
    
    for macro_name in macro_data.keys():
        rs = []
        for symbol in ['BTCUSDT', 'SPY', 'EURUSD']:
            if symbol in results and macro_name in results[symbol]:
                rs.append(results[symbol][macro_name]['level_r'])
            else:
                rs.append(np.nan)
        
        if all(np.isnan(r) for r in rs):
            continue
        
        consistent = (all(r > 0 for r in rs if not np.isnan(r)) or 
                     all(r < 0 for r in rs if not np.isnan(r)))
        
        btc_r = rs[0] if not np.isnan(rs[0]) else 0
        spy_r = rs[1] if not np.isnan(rs[1]) else 0
        eur_r = rs[2] if not np.isnan(rs[2]) else 0
        
        mark = "✅" if consistent and max(abs(btc_r), abs(spy_r), abs(eur_r)) > 0.1 else ""
        
        print(f"  {macro_name:<15} {btc_r:>+10.3f} {spy_r:>+10.3f} {eur_r:>+10.3f} {mark:>12}")
        
        if consistent and max(abs(btc_r), abs(spy_r), abs(eur_r)) > 0.1:
            direction = "positive" if btc_r > 0 else "negative"
            discoveries.append(f"SI-{macro_name}: {direction} correlation across assets")
    
    # Key macro insights
    print("\n" + "="*70)
    print("  KEY MACRO INSIGHTS")
    print("="*70)
    
    # VIX relationship
    if 'VIX' in macro_data:
        vix_rs = [results[s].get('VIX', {}).get('level_r', 0) for s in ['BTCUSDT', 'SPY', 'EURUSD'] if s in results]
        if vix_rs:
            mean_vix = np.mean([r for r in vix_rs if r != 0])
            print(f"\n  SI vs VIX: mean r = {mean_vix:+.3f}")
            if mean_vix < -0.1:
                print("    → High SI = Low market fear (bullish)")
                discoveries.append(f"SI negatively correlates with VIX (r={mean_vix:.3f}) - SI captures risk-off")
            elif mean_vix > 0.1:
                print("    → High SI = High market fear (risk specialization)")
    
    # DXY relationship  
    if 'DXY' in macro_data:
        dxy_rs = [results[s].get('DXY', {}).get('level_r', 0) for s in ['BTCUSDT', 'SPY', 'EURUSD'] if s in results]
        if dxy_rs:
            mean_dxy = np.mean([r for r in dxy_rs if r != 0])
            print(f"\n  SI vs DXY: mean r = {mean_dxy:+.3f}")
            if abs(mean_dxy) > 0.1:
                discoveries.append(f"SI correlates with dollar index (r={mean_dxy:.3f})")
    
    # Save results
    out_path = Path("results/macro_correlations/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'discoveries': discoveries,
        }, f, indent=2)
    
    # Log discoveries
    discoveries_path = Path("docs/DISCOVERIES_LOG.md")
    with open(discoveries_path, 'a') as f:
        f.write(f"\n\n## Macro Correlations - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for d in discoveries:
            f.write(f"- {d}\n")
    
    print(f"\n  Results saved: {out_path}")
    print(f"  Discoveries logged: {discoveries_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
