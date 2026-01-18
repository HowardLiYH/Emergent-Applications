#!/usr/bin/env python3
"""
PHASE 1 DEEPER EXPLORATION
- Mathematical properties of SI
- Information-theoretic analysis
- Change point detection
- Spectral analysis
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
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
    print(f"  📍 {msg}")

def compute_si(data, window=7):
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    population.run(data)
    return population.compute_si_timeseries(data, window=window)

def main():
    print("\n" + "="*70)
    print("  PHASE 1 DEEPER EXPLORATION")
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
    section("1. SI DISTRIBUTION ANALYSIS")
    # ============================================================
    
    print("\n  Analyzing the statistical distribution of SI values")
    
    dist_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        
        dist_results[name] = {
            'mean': float(si.mean()),
            'std': float(si.std()),
            'min': float(si.min()),
            'max': float(si.max()),
            'skewness': float(si.skew()),
            'kurtosis': float(si.kurtosis()),
            'range': float(si.max() - si.min()),
        }
        
        print(f"    Mean: {si.mean():.4f} ± {si.std():.4f}")
        print(f"    Range: [{si.min():.4f}, {si.max():.4f}]")
        print(f"    Skewness: {si.skew():.3f} (0=symmetric)")
        print(f"    Kurtosis: {si.kurtosis():.3f} (0=normal)")
        
        if abs(si.skew()) > 0.5:
            discover(f"SI distribution is skewed in {name} (skew={si.skew():.3f})")
    
    all_results['distribution'] = dist_results
    
    # ============================================================
    section("2. SI CHANGE DYNAMICS")
    # ============================================================
    
    print("\n  How does SI change over time?")
    
    dynamics_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        si_diff = si.diff().dropna()
        
        # Absolute changes
        abs_changes = si_diff.abs()
        
        dynamics_results[name] = {
            'mean_daily_change': float(abs_changes.mean()),
            'max_daily_change': float(abs_changes.max()),
            'volatility_of_si': float(si_diff.std()),
            'autocorr_1': float(si_diff.autocorr(1)),
            'autocorr_5': float(si_diff.autocorr(5)),
        }
        
        print(f"    Mean daily |ΔSI|: {abs_changes.mean():.6f}")
        print(f"    Max daily |ΔSI|:  {abs_changes.max():.6f}")
        print(f"    ΔSI autocorr(1):  {si_diff.autocorr(1):.3f}")
        print(f"    ΔSI autocorr(5):  {si_diff.autocorr(5):.3f}")
        
        # Check for mean reversion in SI changes
        if si_diff.autocorr(1) < -0.1:
            discover(f"SI changes are mean-reverting in {name} (autocorr={si_diff.autocorr(1):.3f})")
    
    all_results['dynamics'] = dynamics_results
    
    # ============================================================
    section("3. SI REGIME DETECTION")
    # ============================================================
    
    print("\n  Detecting high/low SI regimes")
    
    regime_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        returns = data['close'].pct_change()
        
        # Define regimes based on SI quantiles
        high_si = si > si.quantile(0.75)
        low_si = si < si.quantile(0.25)
        medium_si = ~high_si & ~low_si
        
        # Align with returns
        common = si.index.intersection(returns.index)
        
        high_ret = returns[common][high_si[common]].mean() * 252
        low_ret = returns[common][low_si[common]].mean() * 252
        medium_ret = returns[common][medium_si[common]].mean() * 252
        
        high_vol = returns[common][high_si[common]].std() * np.sqrt(252)
        low_vol = returns[common][low_si[common]].std() * np.sqrt(252)
        medium_vol = returns[common][medium_si[common]].std() * np.sqrt(252)
        
        regime_results[name] = {
            'high_si_return': float(high_ret),
            'medium_si_return': float(medium_ret),
            'low_si_return': float(low_ret),
            'high_si_vol': float(high_vol),
            'low_si_vol': float(low_vol),
        }
        
        print(f"    {'Regime':<12} {'Return':>12} {'Volatility':>12}")
        print("    " + "-"*40)
        print(f"    {'High SI':<12} {high_ret:>+12.1%} {high_vol:>12.1%}")
        print(f"    {'Medium SI':<12} {medium_ret:>+12.1%} {medium_vol:>12.1%}")
        print(f"    {'Low SI':<12} {low_ret:>+12.1%} {low_vol:>12.1%}")
        
        # Check if high SI = lower volatility
        if high_vol < low_vol * 0.9:
            discover(f"High SI periods have lower volatility in {name} ({high_vol:.1%} vs {low_vol:.1%})")
    
    all_results['si_regimes'] = regime_results
    
    # ============================================================
    section("4. SI SPECTRAL ANALYSIS")
    # ============================================================
    
    print("\n  Looking for cyclical patterns in SI")
    
    spectral_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        
        # Detrend
        si_detrended = si - si.rolling(30).mean()
        si_clean = si_detrended.dropna()
        
        if len(si_clean) < 100:
            print("    Not enough data for spectral analysis")
            continue
        
        # FFT
        n = len(si_clean)
        fft_vals = np.abs(fft(si_clean.values))[:n//2]
        freqs = fftfreq(n, d=1)[:n//2]  # 1 day sampling
        
        # Find dominant frequencies
        peaks, _ = find_peaks(fft_vals, height=np.percentile(fft_vals, 90))
        
        if len(peaks) > 0:
            dominant_periods = [1/freqs[p] for p in peaks[:5] if freqs[p] > 0]
            dominant_periods = [p for p in dominant_periods if p < 100]  # Filter unrealistic
            
            spectral_results[name] = {
                'dominant_periods_days': [float(p) for p in dominant_periods],
            }
            
            print(f"    Dominant cycles (days): {[f'{p:.1f}' for p in dominant_periods[:3]]}")
            
            # Check for weekly cycle
            if any(6 < p < 8 for p in dominant_periods):
                discover(f"Weekly SI cycle detected in {name}")
        else:
            print("    No clear cyclical patterns")
    
    all_results['spectral'] = spectral_results
    
    # ============================================================
    section("5. SI PERSISTENCE ANALYSIS")
    # ============================================================
    
    print("\n  How long do SI states persist?")
    
    persistence_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        
        # Define states
        high_threshold = si.quantile(0.7)
        low_threshold = si.quantile(0.3)
        
        states = pd.Series(index=si.index, dtype=str)
        states[si > high_threshold] = 'high'
        states[si < low_threshold] = 'low'
        states[(si >= low_threshold) & (si <= high_threshold)] = 'medium'
        
        # Count consecutive runs
        runs = []
        current_state = states.iloc[0]
        current_run = 1
        
        for i in range(1, len(states)):
            if states.iloc[i] == current_state:
                current_run += 1
            else:
                runs.append({'state': current_state, 'length': current_run})
                current_state = states.iloc[i]
                current_run = 1
        runs.append({'state': current_state, 'length': current_run})
        
        runs_df = pd.DataFrame(runs)
        avg_run = runs_df.groupby('state')['length'].mean()
        
        persistence_results[name] = {
            'avg_high_run': float(avg_run.get('high', 0)),
            'avg_medium_run': float(avg_run.get('medium', 0)),
            'avg_low_run': float(avg_run.get('low', 0)),
            'total_transitions': len(runs) - 1,
        }
        
        print(f"    Avg high SI run:   {avg_run.get('high', 0):.1f} days")
        print(f"    Avg medium SI run: {avg_run.get('medium', 0):.1f} days")
        print(f"    Avg low SI run:    {avg_run.get('low', 0):.1f} days")
        print(f"    Total transitions: {len(runs) - 1}")
        
        if avg_run.get('high', 0) > 10:
            discover(f"High SI persists {avg_run.get('high', 0):.1f} days on average in {name}")
    
    all_results['persistence'] = persistence_results
    
    # ============================================================
    section("6. MATHEMATICAL FORMULA VERIFICATION")
    # ============================================================
    
    print("\n  Verifying SI formula predictions")
    
    # SI = 1 - mean(H_normalized)
    # H_normalized = -sum(p log p) / log(K)
    # 
    # Predictions:
    # 1. SI range should be [0, 1]
    # 2. SI should increase when winners are consistent
    # 3. SI should correlate with entropy of regime distribution
    
    formula_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        
        # Check SI range
        in_range = (si.min() >= 0) and (si.max() <= 1)
        
        # Compute regime entropy proxy (from returns)
        returns = data['close'].pct_change().dropna()
        up = (returns > 0.01).rolling(7).mean()
        down = (returns < -0.01).rolling(7).mean()
        flat = 1 - up - down
        
        # Market entropy
        def calc_entropy(p):
            p = np.clip(p, 1e-10, 1)
            return -p * np.log(p)
        
        market_entropy = calc_entropy(up) + calc_entropy(down) + calc_entropy(flat)
        market_entropy = market_entropy / np.log(3)  # Normalize
        
        common = si.index.intersection(market_entropy.index)
        si_aligned = si.loc[common]
        entropy_aligned = market_entropy.loc[common]
        
        # SI should be negatively correlated with market entropy
        r, _ = spearmanr(si_aligned.dropna(), entropy_aligned.dropna())
        
        formula_results[name] = {
            'si_in_range': in_range,
            'si_entropy_corr': float(r),
        }
        
        print(f"    SI in [0,1]: {in_range}")
        print(f"    SI vs Market Entropy: r={r:.3f}")
        
        if r < -0.1:
            discover(f"SI negatively correlates with market entropy in {name} (r={r:.3f}) - THEORY CONFIRMED")
    
    all_results['formula_verification'] = formula_results
    
    # ============================================================
    section("SUMMARY")
    # ============================================================
    
    print(f"\n  Total discoveries: {len(discoveries)}")
    for i, d in enumerate(discoveries, 1):
        print(f"    {i}. {d}")
    
    out_path = Path("results/phase1_deeper/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'discoveries': discoveries,
            'results': all_results,
        }, f, indent=2, default=str)
    
    discoveries_path = Path("docs/DISCOVERIES_LOG.md")
    with open(discoveries_path, 'a') as f:
        f.write(f"\n\n## Phase 1 Deeper - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for d in discoveries:
            f.write(f"- {d}\n")
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
