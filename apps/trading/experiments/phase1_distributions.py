#!/usr/bin/env python3
"""
PHASE 1 DISTRIBUTION ANALYSIS
- Compare SI distributions across assets
- Normality tests
- Distribution fitting
- Correlation distribution analysis
- Joint distribution (copula fitting)
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import (
    spearmanr, normaltest, shapiro, kstest, 
    anderson, skew, kurtosis, norm, t, 
    ks_2samp, mannwhitneyu
)
from scipy.special import kolmogorov
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

def compute_features(data):
    features = pd.DataFrame(index=data.index)
    returns = data['close'].pct_change()
    
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
    features['adx'] = dx.rolling(14).mean()
    
    delta = data['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    features['rsi_extremity'] = abs(rsi - 50)
    
    features['volatility'] = returns.rolling(14).std()
    features['returns'] = returns
    
    return features.dropna()

def main():
    print("\n" + "="*70)
    print("  PHASE 1 DISTRIBUTION ANALYSIS")
    print("="*70)
    print(f"  Time: {datetime.now().isoformat()}")
    
    loader = DataLoaderV2()
    
    assets = {
        'BTCUSDT': (loader.load('BTCUSDT', MarketType.CRYPTO), MarketType.CRYPTO),
        'ETHUSDT': (loader.load('ETHUSDT', MarketType.CRYPTO), MarketType.CRYPTO),
        'SPY': (loader.load('SPY', MarketType.STOCKS), MarketType.STOCKS),
        'QQQ': (loader.load('QQQ', MarketType.STOCKS), MarketType.STOCKS),
        'EURUSD': (loader.load('EURUSD', MarketType.FOREX), MarketType.FOREX),
    }
    
    si_dict = {}
    all_results = {}
    
    # Compute SI for all assets
    print("\n  Computing SI for all assets...")
    for name, (data, mtype) in assets.items():
        si_dict[name] = compute_si(data)
        print(f"    {name}: n={len(si_dict[name])}")
    
    # ============================================================
    section("1. SI DISTRIBUTION COMPARISON ACROSS ASSETS")
    # ============================================================
    
    print("\n  Comparing SI distributions across different assets")
    
    dist_comparison = {}
    
    print(f"\n  {'Asset':<10} {'Mean':>10} {'Std':>10} {'Skew':>10} {'Kurt':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-"*72)
    
    for name, si in si_dict.items():
        si_clean = si.dropna()
        stats = {
            'mean': float(si_clean.mean()),
            'std': float(si_clean.std()),
            'skewness': float(skew(si_clean)),
            'kurtosis': float(kurtosis(si_clean)),
            'min': float(si_clean.min()),
            'max': float(si_clean.max()),
            'median': float(si_clean.median()),
            'iqr': float(si_clean.quantile(0.75) - si_clean.quantile(0.25)),
        }
        dist_comparison[name] = stats
        
        print(f"  {name:<10} {stats['mean']:>10.5f} {stats['std']:>10.5f} {stats['skewness']:>+10.3f} {stats['kurtosis']:>+10.3f} {stats['min']:>10.5f} {stats['max']:>10.5f}")
    
    # Compare means
    means = [dist_comparison[name]['mean'] for name in si_dict.keys()]
    stds = [dist_comparison[name]['std'] for name in si_dict.keys()]
    
    if max(means) / min(means) > 1.2:
        discover(f"SI mean varies significantly across assets (range: {min(means):.5f} to {max(means):.5f})")
    
    all_results['distribution_comparison'] = dist_comparison
    
    # ============================================================
    section("2. NORMALITY TESTS")
    # ============================================================
    
    print("\n  Testing if SI follows a normal distribution")
    
    normality_results = {}
    
    print(f"\n  {'Asset':<10} {'Shapiro p':>12} {'DAgostino p':>12} {'KS p':>12} {'Normal?':>10}")
    print("  " + "-"*58)
    
    for name, si in si_dict.items():
        si_clean = si.dropna().values
        
        # Shapiro-Wilk (best for n < 5000)
        if len(si_clean) < 5000:
            shapiro_stat, shapiro_p = shapiro(si_clean[:5000])
        else:
            shapiro_stat, shapiro_p = shapiro(si_clean[:5000])
        
        # D'Agostino and Pearson's test
        dagostino_stat, dagostino_p = normaltest(si_clean)
        
        # Kolmogorov-Smirnov
        si_standardized = (si_clean - si_clean.mean()) / si_clean.std()
        ks_stat, ks_p = kstest(si_standardized, 'norm')
        
        is_normal = (shapiro_p > 0.05 and dagostino_p > 0.05)
        
        normality_results[name] = {
            'shapiro_p': float(shapiro_p),
            'dagostino_p': float(dagostino_p),
            'ks_p': float(ks_p),
            'is_normal': is_normal,
        }
        
        normal_str = "Yes" if is_normal else "No"
        print(f"  {name:<10} {shapiro_p:>12.4f} {dagostino_p:>12.4f} {ks_p:>12.4f} {normal_str:>10}")
    
    non_normal = [name for name, res in normality_results.items() if not res['is_normal']]
    if len(non_normal) > 0:
        discover(f"SI is NOT normally distributed in {len(non_normal)}/{len(si_dict)} assets")
    
    all_results['normality'] = normality_results
    
    # ============================================================
    section("3. DISTRIBUTION FITTING")
    # ============================================================
    
    print("\n  Fitting different distributions to SI")
    
    fitting_results = {}
    
    for name, si in list(si_dict.items())[:3]:
        print(f"\n  {name}:")
        si_clean = si.dropna().values
        
        # Fit normal
        norm_params = norm.fit(si_clean)
        norm_ks, norm_p = kstest(si_clean, 'norm', norm_params)
        
        # Fit t-distribution
        t_params = t.fit(si_clean)
        t_ks, t_p = kstest(si_clean, 't', t_params)
        
        fitting_results[name] = {
            'normal': {'ks_stat': float(norm_ks), 'p': float(norm_p)},
            't_dist': {'ks_stat': float(t_ks), 'p': float(t_p), 'df': float(t_params[0])},
        }
        
        print(f"    Normal:  KS={norm_ks:.4f}, p={norm_p:.4f}")
        print(f"    t-dist:  KS={t_ks:.4f}, p={t_p:.4f}, df={t_params[0]:.2f}")
        
        if t_p > norm_p:
            discover(f"t-distribution fits SI better than normal in {name} (df={t_params[0]:.1f})")
    
    all_results['distribution_fitting'] = fitting_results
    
    # ============================================================
    section("4. SI CHANGE DISTRIBUTION")
    # ============================================================
    
    print("\n  Analyzing distribution of SI changes (ΔSI)")
    
    change_results = {}
    
    print(f"\n  {'Asset':<10} {'Mean ΔSI':>12} {'Std ΔSI':>12} {'Skew':>10} {'Kurt':>10}")
    print("  " + "-"*56)
    
    for name, si in si_dict.items():
        si_diff = si.diff().dropna()
        
        stats = {
            'mean': float(si_diff.mean()),
            'std': float(si_diff.std()),
            'skewness': float(skew(si_diff)),
            'kurtosis': float(kurtosis(si_diff)),
        }
        change_results[name] = stats
        
        print(f"  {name:<10} {stats['mean']:>+12.6f} {stats['std']:>12.6f} {stats['skewness']:>+10.3f} {stats['kurtosis']:>+10.3f}")
        
        if abs(stats['kurtosis']) > 3:
            discover(f"SI changes have fat tails in {name} (kurtosis={stats['kurtosis']:.2f})")
    
    all_results['change_distribution'] = change_results
    
    # ============================================================
    section("5. CROSS-ASSET DISTRIBUTION COMPARISON")
    # ============================================================
    
    print("\n  Testing if SI distributions are the same across assets")
    
    cross_comparison = []
    asset_names = list(si_dict.keys())
    
    print(f"\n  {'Pair':<20} {'KS stat':>12} {'p-value':>12} {'Same dist?':>12}")
    print("  " + "-"*58)
    
    for i in range(len(asset_names)):
        for j in range(i+1, len(asset_names)):
            a, b = asset_names[i], asset_names[j]
            si_a = si_dict[a].dropna().values
            si_b = si_dict[b].dropna().values
            
            ks_stat, ks_p = ks_2samp(si_a, si_b)
            
            cross_comparison.append({
                'pair': f"{a}-{b}",
                'ks_stat': float(ks_stat),
                'p_value': float(ks_p),
                'same_distribution': ks_p > 0.05,
            })
            
            same_str = "Yes" if ks_p > 0.05 else "No"
            print(f"  {a}-{b:<12} {ks_stat:>12.4f} {ks_p:>12.4f} {same_str:>12}")
    
    same_dist = [c for c in cross_comparison if c['same_distribution']]
    if len(same_dist) > 0:
        discover(f"{len(same_dist)} asset pairs have statistically same SI distribution")
    
    all_results['cross_asset_comparison'] = cross_comparison
    
    # ============================================================
    section("6. JOINT DISTRIBUTION SI-ADX")
    # ============================================================
    
    print("\n  Analyzing joint distribution of SI and ADX")
    
    joint_results = {}
    
    for name, (data, mtype) in list(assets.items())[:3]:
        print(f"\n  {name}:")
        si = si_dict[name]
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common].dropna()
        adx_aligned = features.loc[common, 'adx'].dropna()
        
        common_final = si_aligned.index.intersection(adx_aligned.index)
        si_vals = si_aligned.loc[common_final].values
        adx_vals = adx_aligned.loc[common_final].values
        
        # Compute joint statistics
        r, p = spearmanr(si_vals, adx_vals)
        
        # Compute quadrant probabilities
        si_median = np.median(si_vals)
        adx_median = np.median(adx_vals)
        
        q1 = np.mean((si_vals > si_median) & (adx_vals > adx_median))  # High-High
        q2 = np.mean((si_vals < si_median) & (adx_vals > adx_median))  # Low-High
        q3 = np.mean((si_vals < si_median) & (adx_vals < adx_median))  # Low-Low
        q4 = np.mean((si_vals > si_median) & (adx_vals < adx_median))  # High-Low
        
        joint_results[name] = {
            'spearman_r': float(r),
            'quadrant_probs': {
                'high_high': float(q1),
                'low_high': float(q2),
                'low_low': float(q3),
                'high_low': float(q4),
            }
        }
        
        print(f"    Spearman r: {r:.3f}")
        print(f"    Quadrant probabilities:")
        print(f"      High SI + High ADX: {q1:.1%}")
        print(f"      Low SI + Low ADX:   {q3:.1%}")
        print(f"      Low SI + High ADX:  {q2:.1%}")
        print(f"      High SI + Low ADX:  {q4:.1%}")
        
        # Check for asymmetry
        concordant = q1 + q3
        discordant = q2 + q4
        if concordant > 0.55:
            discover(f"SI-ADX are concordant {concordant:.0%} of the time in {name}")
    
    all_results['joint_distribution'] = joint_results
    
    # ============================================================
    section("7. CORRELATION DISTRIBUTION (ROLLING)")
    # ============================================================
    
    print("\n  Distribution of rolling correlations")
    
    rolling_corr_results = {}
    
    for name, (data, mtype) in list(assets.items())[:3]:
        print(f"\n  {name}:")
        si = si_dict[name]
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        adx_aligned = features.loc[common, 'adx']
        
        # Rolling correlation
        window = 60
        rolling_corrs = []
        
        for i in range(window, len(si_aligned)):
            si_window = si_aligned.iloc[i-window:i]
            adx_window = adx_aligned.iloc[i-window:i]
            r, _ = spearmanr(si_window, adx_window)
            if not np.isnan(r):
                rolling_corrs.append(r)
        
        if rolling_corrs:
            rolling_corrs = np.array(rolling_corrs)
            
            rolling_corr_results[name] = {
                'mean': float(np.mean(rolling_corrs)),
                'std': float(np.std(rolling_corrs)),
                'pct_positive': float((rolling_corrs > 0).mean()),
                'pct_strong_positive': float((rolling_corrs > 0.2).mean()),
                'min': float(np.min(rolling_corrs)),
                'max': float(np.max(rolling_corrs)),
            }
            
            print(f"    Mean rolling r: {np.mean(rolling_corrs):.3f} ± {np.std(rolling_corrs):.3f}")
            print(f"    Range: [{np.min(rolling_corrs):.3f}, {np.max(rolling_corrs):.3f}]")
            print(f"    % Positive: {(rolling_corrs > 0).mean():.1%}")
            print(f"    % Strong (>0.2): {(rolling_corrs > 0.2).mean():.1%}")
    
    all_results['rolling_correlation'] = rolling_corr_results
    
    # ============================================================
    section("SUMMARY")
    # ============================================================
    
    print(f"\n  Total discoveries: {len(discoveries)}")
    for i, d in enumerate(discoveries, 1):
        print(f"    {i}. {d}")
    
    out_path = Path("results/phase1_distributions/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'discoveries': discoveries,
            'results': all_results,
        }, f, indent=2, default=str)
    
    discoveries_path = Path("docs/DISCOVERIES_LOG.md")
    with open(discoveries_path, 'a') as f:
        f.write(f"\n\n## Phase 1 Distributions - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for d in discoveries:
            f.write(f"- {d}\n")
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
