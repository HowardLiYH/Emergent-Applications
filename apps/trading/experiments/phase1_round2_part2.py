#!/usr/bin/env python3
"""
PHASE 1 ROUND 2 PART 2: More Expert Recommendations

1. Topological Data Analysis (Persistent Homology proxy)
2. Autocorrelation Decay Analysis
3. SI as Multi-Factor Component
4. Causal Graph Discovery
5. Cross-Frequency Analysis
6. SI Volatility of Volatility
7. Asymmetry Analysis
8. Threshold Effects
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, skew, kurtosis
from scipy.signal import find_peaks
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, grangercausalitytests
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
    features['rsi'] = rsi
    features['rsi_extremity'] = abs(rsi - 50)
    
    features['volatility'] = returns.rolling(14).std()
    features['returns'] = returns
    features['momentum'] = data['close'].pct_change(7)
    features['volume'] = data['volume'] if 'volume' in data.columns else 1.0
    
    return features.dropna()

def sliding_window_tda_proxy(ts, window=50, step=10):
    """
    TDA proxy: Measure complexity of SI time series using local statistics.
    Instead of full persistent homology, we use:
    - Number of local extrema (peaks + valleys)
    - Range normalized by std
    - Entropy of binned values
    """
    ts = np.array(ts)
    n = len(ts)
    
    complexities = []
    
    for i in range(0, n - window, step):
        segment = ts[i:i+window]
        
        # Number of peaks
        peaks, _ = find_peaks(segment)
        valleys, _ = find_peaks(-segment)
        n_extrema = len(peaks) + len(valleys)
        
        # Range / std ratio
        range_std = (segment.max() - segment.min()) / (segment.std() + 1e-10)
        
        # Binned entropy
        bins = np.linspace(segment.min(), segment.max(), 10)
        digitized = np.digitize(segment, bins)
        counts = np.bincount(digitized, minlength=11)
        probs = counts / len(segment)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        complexities.append({
            'n_extrema': n_extrema,
            'range_std': range_std,
            'entropy': entropy,
            'complexity': n_extrema * entropy / 10,  # Combined metric
        })
    
    return complexities

def build_causal_graph(si, features, max_lag=5):
    """
    Build a causal graph using pairwise Granger causality.
    """
    graph = {}
    feature_names = ['adx', 'rsi_extremity', 'volatility', 'momentum']
    
    common = si.index.intersection(features.index)
    si_aligned = si.loc[common].dropna()
    
    for feat in feature_names:
        if feat not in features.columns:
            continue
            
        feat_aligned = features.loc[common, feat].dropna()
        common_final = si_aligned.index.intersection(feat_aligned.index)
        
        if len(common_final) < max_lag * 3 + 10:
            continue
        
        df = pd.DataFrame({
            'si': si_aligned.loc[common_final].values,
            'feat': feat_aligned.loc[common_final].values,
        }).dropna()
        
        if len(df) < max_lag * 3:
            continue
        
        try:
            # SI -> Feature
            gc_si_to_feat = grangercausalitytests(df[['feat', 'si']], maxlag=max_lag, verbose=False)
            min_p_si_to_feat = min([gc_si_to_feat[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)])
            
            # Feature -> SI
            gc_feat_to_si = grangercausalitytests(df[['si', 'feat']], maxlag=max_lag, verbose=False)
            min_p_feat_to_si = min([gc_feat_to_si[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)])
            
            graph[feat] = {
                'si_causes_feat': min_p_si_to_feat < 0.05,
                'si_causes_feat_p': float(min_p_si_to_feat),
                'feat_causes_si': min_p_feat_to_si < 0.05,
                'feat_causes_si_p': float(min_p_feat_to_si),
            }
        except:
            pass
    
    return graph

def analyze_si_asymmetry(si, returns):
    """
    Test if SI behaves differently for positive vs negative returns.
    """
    common = si.index.intersection(returns.index)
    si_aligned = si.loc[common].dropna()
    returns_aligned = returns.loc[common].dropna()
    
    common_final = si_aligned.index.intersection(returns_aligned.index)
    si_vals = si_aligned.loc[common_final]
    ret_vals = returns_aligned.loc[common_final]
    
    # Split by return sign
    pos_mask = ret_vals > 0
    neg_mask = ret_vals < 0
    
    si_pos = si_vals[pos_mask]
    si_neg = si_vals[neg_mask]
    
    return {
        'mean_si_pos_returns': float(si_pos.mean()),
        'mean_si_neg_returns': float(si_neg.mean()),
        'std_si_pos_returns': float(si_pos.std()),
        'std_si_neg_returns': float(si_neg.std()),
        'asymmetry': float(si_pos.mean() - si_neg.mean()),
        'interpretation': 'Higher SI in up days' if si_pos.mean() > si_neg.mean() else 'Higher SI in down days',
    }

def analyze_threshold_effects(si, returns, percentiles=[10, 25, 50, 75, 90]):
    """
    Analyze returns conditional on SI percentiles.
    """
    common = si.index.intersection(returns.index)
    si_aligned = si.loc[common].dropna()
    returns_aligned = returns.loc[common].dropna()
    
    common_final = si_aligned.index.intersection(returns_aligned.index)
    si_vals = si_aligned.loc[common_final]
    ret_vals = returns_aligned.loc[common_final].shift(-1)  # Future returns
    
    common_shift = si_vals.index.intersection(ret_vals.dropna().index)
    si_vals = si_vals.loc[common_shift]
    ret_vals = ret_vals.loc[common_shift]
    
    results = {}
    thresholds = [np.percentile(si_vals, p) for p in percentiles]
    
    prev_threshold = si_vals.min()
    for i, (pct, threshold) in enumerate(zip(percentiles, thresholds)):
        mask = (si_vals >= prev_threshold) & (si_vals < threshold)
        if mask.sum() > 10:
            mean_ret = ret_vals[mask].mean() * 252  # Annualized
            sharpe = mean_ret / (ret_vals[mask].std() * np.sqrt(252) + 1e-10)
            results[f'p{pct}'] = {
                'count': int(mask.sum()),
                'mean_ret_ann': float(mean_ret),
                'sharpe': float(sharpe),
            }
        prev_threshold = threshold
    
    # Top percentile
    mask = si_vals >= thresholds[-1]
    if mask.sum() > 10:
        mean_ret = ret_vals[mask].mean() * 252
        sharpe = mean_ret / (ret_vals[mask].std() * np.sqrt(252) + 1e-10)
        results['p100'] = {
            'count': int(mask.sum()),
            'mean_ret_ann': float(mean_ret),
            'sharpe': float(sharpe),
        }
    
    return results

def main():
    print("\n" + "="*70)
    print("  PHASE 1 ROUND 2 PART 2: MORE EXPERT RECOMMENDATIONS")
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
    section("1. TDA PROXY (Sliding Window Complexity)")
    # ============================================================
    
    print("\n  Measuring SI trajectory complexity over time")
    
    tda_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        si_clean = si.dropna().values
        
        complexities = sliding_window_tda_proxy(si_clean, window=60, step=20)
        
        if complexities:
            mean_complexity = np.mean([c['complexity'] for c in complexities])
            std_complexity = np.std([c['complexity'] for c in complexities])
            mean_extrema = np.mean([c['n_extrema'] for c in complexities])
            
            tda_results[name] = {
                'mean_complexity': float(mean_complexity),
                'std_complexity': float(std_complexity),
                'mean_extrema_per_window': float(mean_extrema),
            }
            
            print(f"    Mean complexity: {mean_complexity:.3f} ± {std_complexity:.3f}")
            print(f"    Avg extrema per 60-day window: {mean_extrema:.1f}")
            
            if mean_extrema > 10:
                discover(f"SI trajectory is highly complex in {name} ({mean_extrema:.0f} extrema/window)")
    
    all_results['tda_proxy'] = tda_results
    
    # ============================================================
    section("2. AUTOCORRELATION DECAY ANALYSIS")
    # ============================================================
    
    print("\n  How fast does SI autocorrelation decay?")
    
    acf_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        si_clean = si.dropna().values
        
        # Compute ACF
        acf_values = acf(si_clean, nlags=30)
        
        # Find half-life (where ACF drops below 0.5)
        half_life = None
        for lag, val in enumerate(acf_values):
            if val < 0.5:
                half_life = lag
                break
        
        # Find decorrelation (where ACF drops below 0.1)
        decorr_lag = None
        for lag, val in enumerate(acf_values):
            if val < 0.1:
                decorr_lag = lag
                break
        
        acf_results[name] = {
            'acf_lag1': float(acf_values[1]),
            'acf_lag5': float(acf_values[5]),
            'acf_lag10': float(acf_values[10]),
            'acf_lag20': float(acf_values[20]),
            'half_life': half_life,
            'decorrelation_lag': decorr_lag,
        }
        
        print(f"    ACF(1): {acf_values[1]:.3f}")
        print(f"    ACF(5): {acf_values[5]:.3f}")
        print(f"    ACF(10): {acf_values[10]:.3f}")
        print(f"    Half-life: {half_life} days" if half_life else "    Half-life: >30 days")
        
        if half_life and half_life < 5:
            discover(f"SI decorrelates quickly in {name} (half-life={half_life} days)")
        elif half_life is None:
            discover(f"SI is highly persistent in {name} (half-life >30 days)")
    
    all_results['acf'] = acf_results
    
    # ============================================================
    section("3. CAUSAL GRAPH DISCOVERY")
    # ============================================================
    
    print("\n  Building causal graph with Granger tests")
    
    causal_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        
        graph = build_causal_graph(si, features, max_lag=5)
        causal_results[name] = graph
        
        print(f"    {'Feature':<15} {'SI→Feat':>10} {'Feat→SI':>10}")
        print("    " + "-"*37)
        
        for feat, results in graph.items():
            si_to_feat = "✓" if results['si_causes_feat'] else ""
            feat_to_si = "✓" if results['feat_causes_si'] else ""
            print(f"    {feat:<15} {si_to_feat:>10} {feat_to_si:>10}")
            
            if results['feat_causes_si'] and not results['si_causes_feat']:
                discover(f"{feat} → SI (unidirectional causality) in {name}")
    
    all_results['causal_graph'] = causal_results
    
    # ============================================================
    section("4. SI ASYMMETRY ANALYSIS")
    # ============================================================
    
    print("\n  Does SI behave differently in up vs down days?")
    
    asymmetry_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        returns = data['close'].pct_change()
        
        asym = analyze_si_asymmetry(si, returns)
        asymmetry_results[name] = asym
        
        print(f"    Mean SI (up days):   {asym['mean_si_pos_returns']:.5f}")
        print(f"    Mean SI (down days): {asym['mean_si_neg_returns']:.5f}")
        print(f"    Asymmetry: {asym['asymmetry']:+.5f}")
        print(f"    → {asym['interpretation']}")
        
        if abs(asym['asymmetry']) > 0.001:
            discover(f"SI asymmetry: {asym['interpretation']} in {name}")
    
    all_results['asymmetry'] = asymmetry_results
    
    # ============================================================
    section("5. THRESHOLD EFFECTS (SI Quantile Analysis)")
    # ============================================================
    
    print("\n  Returns by SI quantile (sweet spot analysis)")
    
    threshold_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        returns = data['close'].pct_change()
        
        thresh = analyze_threshold_effects(si, returns)
        threshold_results[name] = thresh
        
        print(f"    {'SI Quantile':<12} {'Count':>8} {'Ann.Ret':>10} {'Sharpe':>8}")
        print("    " + "-"*42)
        
        best_sharpe = -999
        best_quantile = None
        
        for quant, stats in thresh.items():
            print(f"    {quant:<12} {stats['count']:>8} {stats['mean_ret_ann']:>+10.1%} {stats['sharpe']:>+8.2f}")
            if stats['sharpe'] > best_sharpe:
                best_sharpe = stats['sharpe']
                best_quantile = quant
        
        if best_quantile and best_sharpe > 0.5:
            discover(f"SI sweet spot at {best_quantile} in {name} (Sharpe={best_sharpe:.2f})")
    
    all_results['threshold_effects'] = threshold_results
    
    # ============================================================
    section("6. SI VOLATILITY OF VOLATILITY")
    # ============================================================
    
    print("\n  Is SI volatility stable or varying?")
    
    vol_of_vol_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        
        # Rolling volatility of SI
        si_vol = si.rolling(20).std()
        
        # Vol of vol
        vol_of_vol = si_vol.rolling(20).std()
        
        mean_vol = si_vol.mean()
        mean_vol_of_vol = vol_of_vol.mean()
        
        vol_of_vol_results[name] = {
            'mean_si_vol': float(mean_vol),
            'mean_vol_of_vol': float(mean_vol_of_vol),
            'vol_of_vol_ratio': float(mean_vol_of_vol / mean_vol) if mean_vol > 0 else None,
        }
        
        ratio = mean_vol_of_vol / mean_vol if mean_vol > 0 else 0
        print(f"    Mean SI volatility: {mean_vol:.5f}")
        print(f"    Vol of vol: {mean_vol_of_vol:.5f}")
        print(f"    Ratio: {ratio:.3f}")
        
        if ratio > 0.5:
            discover(f"SI volatility is itself volatile in {name} (ratio={ratio:.2f})")
    
    all_results['vol_of_vol'] = vol_of_vol_results
    
    # ============================================================
    section("7. SI-MOMENTUM INTERACTION")
    # ============================================================
    
    print("\n  Does SI × Momentum have predictive power?")
    
    interaction_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        returns = data['close'].pct_change()
        
        common = si.index.intersection(features.index).intersection(returns.index)
        si_aligned = si.loc[common]
        mom_aligned = features.loc[common, 'momentum']
        ret_aligned = returns.loc[common].shift(-1)  # Future return
        
        common_final = si_aligned.dropna().index.intersection(mom_aligned.dropna().index).intersection(ret_aligned.dropna().index)
        
        si_vals = si_aligned.loc[common_final]
        mom_vals = mom_aligned.loc[common_final]
        ret_vals = ret_aligned.loc[common_final]
        
        # Interaction term
        interaction = (si_vals - si_vals.mean()) * (mom_vals - mom_vals.mean())
        
        # Correlation with future returns
        r_si, _ = spearmanr(si_vals, ret_vals)
        r_mom, _ = spearmanr(mom_vals, ret_vals)
        r_interaction, p_interaction = spearmanr(interaction, ret_vals)
        
        interaction_results[name] = {
            'r_si_returns': float(r_si),
            'r_mom_returns': float(r_mom),
            'r_interaction_returns': float(r_interaction),
            'p_interaction': float(p_interaction),
        }
        
        print(f"    SI → Ret:            r = {r_si:+.4f}")
        print(f"    Mom → Ret:           r = {r_mom:+.4f}")
        print(f"    SI×Mom → Ret:        r = {r_interaction:+.4f} (p={p_interaction:.4f})")
        
        if abs(r_interaction) > max(abs(r_si), abs(r_mom)) and p_interaction < 0.05:
            discover(f"SI×Momentum interaction is predictive in {name} (r={r_interaction:.3f})")
    
    all_results['interaction'] = interaction_results
    
    # ============================================================
    section("8. SI HIGHER MOMENTS")
    # ============================================================
    
    print("\n  Rolling skewness and kurtosis of SI")
    
    moments_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        si_clean = si.dropna()
        
        # Rolling moments
        rolling_skew = si_clean.rolling(60).apply(skew)
        rolling_kurt = si_clean.rolling(60).apply(kurtosis)
        
        moments_results[name] = {
            'overall_skew': float(skew(si_clean)),
            'overall_kurt': float(kurtosis(si_clean)),
            'mean_rolling_skew': float(rolling_skew.mean()),
            'std_rolling_skew': float(rolling_skew.std()),
            'mean_rolling_kurt': float(rolling_kurt.mean()),
            'std_rolling_kurt': float(rolling_kurt.std()),
        }
        
        print(f"    Overall skewness: {skew(si_clean):+.3f}")
        print(f"    Overall kurtosis: {kurtosis(si_clean):+.3f}")
        print(f"    Rolling skew: {rolling_skew.mean():+.3f} ± {rolling_skew.std():.3f}")
        print(f"    Rolling kurt: {rolling_kurt.mean():+.3f} ± {rolling_kurt.std():.3f}")
        
        if rolling_skew.std() > 0.5:
            discover(f"SI skewness varies significantly over time in {name}")
    
    all_results['moments'] = moments_results
    
    # ============================================================
    section("SUMMARY")
    # ============================================================
    
    print(f"\n  Total discoveries: {len(discoveries)}")
    for i, d in enumerate(discoveries, 1):
        print(f"    {i}. {d}")
    
    out_path = Path("results/phase1_round2_part2/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'discoveries': discoveries,
            'results': all_results,
        }, f, indent=2, default=str)
    
    discoveries_path = Path("docs/DISCOVERIES_LOG.md")
    with open(discoveries_path, 'a') as f:
        f.write(f"\n\n## Phase 1 Round 2 Part 2 - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for d in discoveries:
            f.write(f"- {d}\n")
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
