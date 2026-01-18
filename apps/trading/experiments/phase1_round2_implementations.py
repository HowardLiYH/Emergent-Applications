#!/usr/bin/env python3
"""
PHASE 1 ROUND 2: Expert Recommendations Implementation

Top 10 from 35 experts:
1. Multifractal Analysis (MFDFA)
2. SI Stress Testing (Crisis periods)
3. Multiple Testing Correction (FDR)
4. Fractional BM proper fitting
5. Change Point Detection
6. Recurrence Quantification Analysis
7. SI Drawdown Prediction
8. Data Source Robustness
9. SI as Factor
10. Regime Clustering
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
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

def mfdfa(ts, q_list=[-5, -3, -1, 1, 3, 5], scales=None):
    """
    Multifractal Detrended Fluctuation Analysis.
    Returns H(q) for different q values.
    """
    ts = np.array(ts)
    n = len(ts)
    
    if scales is None:
        scales = [2**i for i in range(4, int(np.log2(n/4)))]
    
    # Profile (cumulative sum of detrended series)
    profile = np.cumsum(ts - np.mean(ts))
    
    hurst_q = {}
    
    for q in q_list:
        fluctuations = []
        
        for scale in scales:
            n_segments = n // scale
            if n_segments < 2:
                continue
            
            rms_list = []
            for i in range(n_segments):
                segment = profile[i*scale:(i+1)*scale]
                # Detrend with linear fit
                x = np.arange(scale)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                rms = np.sqrt(np.mean((segment - trend)**2))
                rms_list.append(rms)
            
            if rms_list:
                if q == 0:
                    fq = np.exp(0.5 * np.mean(np.log(np.array(rms_list)**2 + 1e-10)))
                else:
                    fq = np.mean(np.array(rms_list)**q)**(1/q)
                fluctuations.append((scale, fq))
        
        if len(fluctuations) > 3:
            log_scales = np.log([f[0] for f in fluctuations])
            log_flucts = np.log([f[1] for f in fluctuations])
            h_q = np.polyfit(log_scales, log_flucts, 1)[0]
            hurst_q[q] = h_q
    
    return hurst_q

def recurrence_quantification(ts, embedding_dim=3, delay=1, threshold=None):
    """
    Recurrence Quantification Analysis.
    """
    ts = np.array(ts)
    n = len(ts) - (embedding_dim - 1) * delay
    
    if n < 50:
        return None
    
    # Create embedded vectors
    embedded = np.zeros((n, embedding_dim))
    for i in range(n):
        for j in range(embedding_dim):
            embedded[i, j] = ts[i + j * delay]
    
    # Normalize
    embedded = (embedded - embedded.mean(axis=0)) / (embedded.std(axis=0) + 1e-10)
    
    # Distance matrix (subsample for speed)
    sample_size = min(500, n)
    indices = np.random.choice(n, sample_size, replace=False)
    embedded_sample = embedded[indices]
    
    dists = np.zeros((sample_size, sample_size))
    for i in range(sample_size):
        dists[i] = np.sqrt(np.sum((embedded_sample - embedded_sample[i])**2, axis=1))
    
    # Auto-threshold: 10% of max distance
    if threshold is None:
        threshold = 0.1 * np.max(dists)
    
    # Recurrence matrix
    R = (dists < threshold).astype(int)
    np.fill_diagonal(R, 0)
    
    # Recurrence rate
    rr = R.sum() / (sample_size * (sample_size - 1))
    
    # Determinism (diagonal lines)
    det_count = 0
    total_recur = 0
    for i in range(-sample_size+2, sample_size-1):
        diag = np.diag(R, i)
        # Count consecutive 1s
        in_line = False
        line_len = 0
        for val in diag:
            if val == 1:
                if not in_line:
                    in_line = True
                    line_len = 1
                else:
                    line_len += 1
            else:
                if in_line and line_len >= 2:
                    det_count += line_len
                in_line = False
                line_len = 0
            total_recur += val
        if in_line and line_len >= 2:
            det_count += line_len
    
    determinism = det_count / (total_recur + 1e-10)
    
    return {
        'recurrence_rate': float(rr),
        'determinism': float(determinism),
        'threshold': float(threshold),
    }

def detect_change_points(ts, method='cusum', threshold=2.0):
    """
    Detect change points in SI using CUSUM or variance-based method.
    """
    ts = np.array(ts)
    n = len(ts)
    
    change_points = []
    
    if method == 'cusum':
        mean = ts.mean()
        cusum = np.cumsum(ts - mean)
        cusum_diff = np.diff(cusum)
        
        # Find peaks in CUSUM derivative
        peaks, _ = find_peaks(np.abs(cusum_diff), height=threshold * np.std(cusum_diff))
        change_points = peaks.tolist()
    
    elif method == 'variance':
        window = 30
        for i in range(window, n - window):
            var_before = ts[i-window:i].var()
            var_after = ts[i:i+window].var()
            ratio = max(var_before, var_after) / (min(var_before, var_after) + 1e-10)
            if ratio > threshold:
                change_points.append(i)
    
    # Remove nearby duplicates
    filtered = []
    for cp in change_points:
        if not filtered or cp - filtered[-1] > 20:
            filtered.append(cp)
    
    return filtered

def regime_clustering(si, n_clusters=3):
    """
    Cluster SI values into regimes.
    """
    si_values = si.dropna().values.reshape(-1, 1)
    
    if len(si_values) < n_clusters * 10:
        return None
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(si_values)
    
    centers = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(centers)
    
    # Rename labels by SI level
    label_map = {sorted_idx[i]: i for i in range(n_clusters)}
    labels = np.array([label_map[l] for l in labels])
    
    # Regime statistics
    regime_stats = {}
    for i in range(n_clusters):
        mask = labels == i
        regime_name = ['Low', 'Medium', 'High'][i] if n_clusters == 3 else f'Regime_{i}'
        regime_stats[regime_name] = {
            'center': float(centers[sorted_idx[i]]),
            'count': int(mask.sum()),
            'pct': float(mask.mean()),
        }
    
    return {
        'n_clusters': n_clusters,
        'regimes': regime_stats,
        'labels': labels.tolist(),
    }

def si_drawdown_prediction(si, returns, lookback=5):
    """
    Test if SI predicts future drawdowns.
    """
    si = si.dropna()
    returns = returns.dropna()
    
    common = si.index.intersection(returns.index)
    si_aligned = si.loc[common]
    returns_aligned = returns.loc[common]
    
    # Compute future max drawdown (next 20 days)
    future_dd = []
    for i in range(len(returns_aligned) - 20):
        future_ret = returns_aligned.iloc[i:i+20]
        cum_ret = (1 + future_ret).cumprod()
        peak = cum_ret.expanding().max()
        dd = (cum_ret / peak - 1).min()
        future_dd.append(dd)
    
    future_dd = pd.Series(future_dd, index=returns_aligned.index[:len(future_dd)])
    
    common_dd = si_aligned.index[:len(future_dd)]
    
    r, p = spearmanr(si_aligned.loc[common_dd], future_dd)
    
    return {
        'correlation': float(r),
        'p_value': float(p),
        'interpretation': 'High SI predicts less drawdown' if r > 0 else 'High SI predicts more drawdown',
        'significant': p < 0.05,
    }

def main():
    print("\n" + "="*70)
    print("  PHASE 1 ROUND 2: EXPERT RECOMMENDATIONS")
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
    section("1. MULTIFRACTAL ANALYSIS (MFDFA)")
    # ============================================================
    
    print("\n  Does Hurst exponent vary with q (moment order)?")
    print("  H(q) constant = monofractal, H(q) varies = multifractal")
    
    mfdfa_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        si_clean = si.dropna().values
        
        h_q = mfdfa(si_clean, q_list=[-4, -2, 0, 2, 4])
        mfdfa_results[name] = {k: float(v) for k, v in h_q.items()}
        
        if h_q:
            h_range = max(h_q.values()) - min(h_q.values())
            print(f"    H(-4) = {h_q.get(-4, 'N/A'):.3f}" if -4 in h_q else "    H(-4) = N/A")
            print(f"    H(0)  = {h_q.get(0, 'N/A'):.3f}" if 0 in h_q else "    H(0)  = N/A")
            print(f"    H(4)  = {h_q.get(4, 'N/A'):.3f}" if 4 in h_q else "    H(4)  = N/A")
            print(f"    ΔH = {h_range:.3f} ({'Multifractal' if h_range > 0.1 else 'Monofractal'})")
            
            if h_range > 0.1:
                discover(f"SI is multifractal in {name} (ΔH={h_range:.3f})")
    
    all_results['mfdfa'] = mfdfa_results
    
    # ============================================================
    section("2. SI STRESS TESTING (Crisis Periods)")
    # ============================================================
    
    print("\n  How does SI behave during market crises?")
    
    crisis_periods = {
        'COVID_2020': ('2020-02-15', '2020-04-15'),
        'Crypto_Crash_2022': ('2022-05-01', '2022-07-01'),
        'Rate_Hike_2022': ('2022-09-01', '2022-11-01'),
    }
    
    stress_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        
        asset_stress = {}
        for crisis_name, (start, end) in crisis_periods.items():
            try:
                crisis_si = si.loc[start:end]
                normal_si = si.loc[:start]
                
                if len(crisis_si) > 5 and len(normal_si) > 30:
                    crisis_mean = crisis_si.mean()
                    normal_mean = normal_si.mean()
                    crisis_std = crisis_si.std()
                    normal_std = normal_si.std()
                    
                    asset_stress[crisis_name] = {
                        'crisis_mean': float(crisis_mean),
                        'normal_mean': float(normal_mean),
                        'change_pct': float((crisis_mean - normal_mean) / normal_mean * 100),
                        'crisis_std': float(crisis_std),
                        'normal_std': float(normal_std),
                    }
                    
                    change = (crisis_mean - normal_mean) / normal_mean * 100
                    direction = "↓" if change < 0 else "↑"
                    print(f"    {crisis_name}: SI {direction} {abs(change):.1f}%")
                    
                    if abs(change) > 20:
                        discover(f"SI {'dropped' if change < 0 else 'spiked'} {abs(change):.0f}% during {crisis_name} in {name}")
            except:
                pass
        
        stress_results[name] = asset_stress
    
    all_results['stress_testing'] = stress_results
    
    # ============================================================
    section("3. MULTIPLE TESTING CORRECTION (FDR)")
    # ============================================================
    
    print("\n  Applying Benjamini-Hochberg FDR correction")
    
    # Collect all p-values from previous analyses
    all_pvalues = []
    test_names = []
    
    for name, (data, mtype) in assets.items():
        si = compute_si(data)
        returns = data['close'].pct_change().dropna()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        common_final = si_aligned.index.intersection(returns_aligned.index)
        
        # SI-Returns correlation
        r, p = spearmanr(si_aligned.loc[common_final], returns_aligned.loc[common_final])
        all_pvalues.append(p)
        test_names.append(f'{name}_SI_Returns')
        
        # SI-Volatility correlation
        vol = returns.rolling(14).std().dropna()
        common_vol = si_aligned.index.intersection(vol.index)
        r, p = spearmanr(si_aligned.loc[common_vol], vol.loc[common_vol])
        all_pvalues.append(p)
        test_names.append(f'{name}_SI_Vol')
    
    # Apply FDR
    rejected, pvals_corrected, _, _ = multipletests(all_pvalues, method='fdr_bh', alpha=0.05)
    
    fdr_results = []
    print(f"\n  {'Test':<25} {'p-value':>10} {'p-adj':>10} {'Sig?':>6}")
    print("  " + "-"*55)
    
    for i, (name, p, p_adj, sig) in enumerate(zip(test_names, all_pvalues, pvals_corrected, rejected)):
        fdr_results.append({
            'test': name,
            'p_value': float(p),
            'p_adjusted': float(p_adj),
            'significant': bool(sig),
        })
        sig_str = "✓" if sig else ""
        print(f"  {name:<25} {p:>10.4f} {p_adj:>10.4f} {sig_str:>6}")
    
    n_sig = sum(rejected)
    discover(f"{n_sig}/{len(all_pvalues)} tests remain significant after FDR correction")
    
    all_results['fdr_correction'] = fdr_results
    
    # ============================================================
    section("4. RECURRENCE QUANTIFICATION ANALYSIS")
    # ============================================================
    
    print("\n  Detecting deterministic structure in SI dynamics")
    
    rqa_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        si_clean = si.dropna().values
        
        rqa = recurrence_quantification(si_clean, embedding_dim=3, delay=1)
        
        if rqa:
            rqa_results[name] = rqa
            print(f"    Recurrence Rate: {rqa['recurrence_rate']:.3f}")
            print(f"    Determinism:     {rqa['determinism']:.3f}")
            
            if rqa['determinism'] > 0.7:
                discover(f"SI has high determinism in {name} (DET={rqa['determinism']:.2f})")
    
    all_results['rqa'] = rqa_results
    
    # ============================================================
    section("5. CHANGE POINT DETECTION")
    # ============================================================
    
    print("\n  Detecting regime changes in SI")
    
    changepoint_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        si_clean = si.dropna()
        
        cps = detect_change_points(si_clean.values, method='cusum', threshold=2.0)
        
        changepoint_results[name] = {
            'n_change_points': len(cps),
            'change_point_dates': [str(si_clean.index[cp]) for cp in cps if cp < len(si_clean)],
        }
        
        print(f"    Found {len(cps)} change points")
        for cp in cps[:3]:
            if cp < len(si_clean):
                print(f"      - {si_clean.index[cp].strftime('%Y-%m-%d')}")
        if len(cps) > 3:
            print(f"      ... and {len(cps) - 3} more")
    
    all_results['change_points'] = changepoint_results
    
    # ============================================================
    section("6. REGIME CLUSTERING")
    # ============================================================
    
    print("\n  Clustering SI into discrete regimes")
    
    clustering_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        
        clustering = regime_clustering(si, n_clusters=3)
        
        if clustering:
            clustering_results[name] = clustering
            for regime, stats in clustering['regimes'].items():
                print(f"    {regime}: center={stats['center']:.4f}, {stats['pct']:.1%} of data")
    
    all_results['clustering'] = clustering_results
    
    # ============================================================
    section("7. SI DRAWDOWN PREDICTION")
    # ============================================================
    
    print("\n  Does SI predict future drawdowns?")
    
    drawdown_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        returns = data['close'].pct_change()
        
        dd = si_drawdown_prediction(si, returns)
        drawdown_results[name] = dd
        
        sig = "✓" if dd['significant'] else ""
        print(f"    SI-Drawdown correlation: r={dd['correlation']:+.3f}, p={dd['p_value']:.4f} {sig}")
        print(f"    {dd['interpretation']}")
        
        if dd['significant'] and dd['correlation'] > 0:
            discover(f"High SI predicts smaller drawdowns in {name} (r={dd['correlation']:.2f})")
    
    all_results['drawdown_prediction'] = drawdown_results
    
    # ============================================================
    section("SUMMARY")
    # ============================================================
    
    print(f"\n  Total discoveries: {len(discoveries)}")
    for i, d in enumerate(discoveries, 1):
        print(f"    {i}. {d}")
    
    out_path = Path("results/phase1_round2/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'discoveries': discoveries,
            'results': all_results,
        }, f, indent=2, default=str)
    
    discoveries_path = Path("docs/DISCOVERIES_LOG.md")
    with open(discoveries_path, 'a') as f:
        f.write(f"\n\n## Phase 1 Round 2 - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for d in discoveries:
            f.write(f"- {d}\n")
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
