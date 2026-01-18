#!/usr/bin/env python3
"""
PHASE 1 ROUND 2 PART 3: Advanced Statistical Methods

From Expert Panel:
1. Nonlinear Correlation (Maximal Information Coefficient proxy)
2. Dynamic Time Warping between SI series
3. Rolling Beta Stability
4. Information Flow Networks
5. SI Predictability (Out-of-sample R²)
6. Rolling Factor Exposure
7. SI Regime Transitions
8. Persistence Analysis
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
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
    features['rsi'] = 100 - (100 / (1 + rs))
    
    features['volatility'] = returns.rolling(14).std()
    features['returns'] = returns
    features['momentum'] = data['close'].pct_change(7)
    
    return features.dropna()

def mutual_information_proxy(x, y, bins=10):
    """
    Compute mutual information proxy using binned joint histogram.
    """
    x = np.array(x)
    y = np.array(y)
    
    # Bin the data
    x_bins = np.digitize(x, np.linspace(x.min(), x.max(), bins))
    y_bins = np.digitize(y, np.linspace(y.min(), y.max(), bins))
    
    # Joint histogram
    joint = np.zeros((bins+1, bins+1))
    for xi, yi in zip(x_bins, y_bins):
        joint[xi, yi] += 1
    
    joint /= len(x)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    
    # Mutual information
    with np.errstate(divide='ignore', invalid='ignore'):
        mi = np.sum(joint * np.log(joint / (px @ py + 1e-10) + 1e-10))
    
    return float(mi)

def dtw_distance(x, y):
    """
    Compute Dynamic Time Warping distance between two series.
    """
    n, m = len(x), len(y)
    
    # Use a window constraint for efficiency
    window = max(n, m) // 4
    
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(max(1, i-window), min(m+1, i+window+1)):
            cost = abs(x[i-1] - y[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]

def rolling_beta(y, X, window=60):
    """
    Compute rolling regression beta.
    """
    betas = []
    for i in range(window, len(y)):
        y_win = y[i-window:i]
        X_win = X[i-window:i]
        
        if len(y_win) < window * 0.8:
            betas.append(np.nan)
            continue
        
        X_win = np.column_stack([np.ones(len(X_win)), X_win])
        try:
            coeffs = np.linalg.lstsq(X_win, y_win, rcond=None)[0]
            betas.append(coeffs[1])
        except:
            betas.append(np.nan)
    
    return np.array(betas)

def analyze_regime_transitions(si, n_regimes=3):
    """
    Analyze SI regime transition probabilities.
    """
    si_clean = si.dropna()
    
    # Discretize into regimes
    thresholds = [np.percentile(si_clean, 100*i/n_regimes) for i in range(1, n_regimes)]
    
    regimes = np.zeros(len(si_clean), dtype=int)
    for i, thresh in enumerate(thresholds):
        regimes[si_clean.values > thresh] = i + 1
    
    # Transition matrix
    trans_matrix = np.zeros((n_regimes, n_regimes))
    for i in range(len(regimes)-1):
        trans_matrix[regimes[i], regimes[i+1]] += 1
    
    # Normalize
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    trans_matrix = trans_matrix / (row_sums + 1e-10)
    
    # Persistence = diagonal elements
    persistence = np.diag(trans_matrix)
    
    return {
        'transition_matrix': trans_matrix.tolist(),
        'persistence': persistence.tolist(),
        'mean_persistence': float(persistence.mean()),
    }

def si_predictability_test(si, features, horizons=[1, 5, 10]):
    """
    Test if SI is predictable out-of-sample using features.
    """
    common = si.index.intersection(features.index)
    si_aligned = si.loc[common].dropna()
    features_aligned = features.loc[common].dropna()
    
    common_final = si_aligned.index.intersection(features_aligned.index)
    
    if len(common_final) < 100:
        return {}
    
    si_vals = si_aligned.loc[common_final].values
    feat_vals = features_aligned.loc[common_final][['adx', 'rsi', 'volatility', 'momentum']].values
    
    results = {}
    
    for horizon in horizons:
        if len(si_vals) < horizon + 60:
            continue
        
        # Target: SI h days ahead
        y = si_vals[horizon:]
        X = feat_vals[:-horizon]
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        oos_r2s = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            oos_r2s.append(r2)
        
        results[f'h{horizon}'] = {
            'mean_oos_r2': float(np.mean(oos_r2s)),
            'std_oos_r2': float(np.std(oos_r2s)),
        }
    
    return results

def main():
    print("\n" + "="*70)
    print("  PHASE 1 ROUND 2 PART 3: ADVANCED STATISTICAL METHODS")
    print("="*70)
    print(f"  Time: {datetime.now().isoformat()}")
    
    loader = DataLoaderV2()
    
    assets = {
        'BTCUSDT': (loader.load('BTCUSDT', MarketType.CRYPTO), MarketType.CRYPTO),
        'SPY': (loader.load('SPY', MarketType.STOCKS), MarketType.STOCKS),
        'EURUSD': (loader.load('EURUSD', MarketType.FOREX), MarketType.FOREX),
    }
    
    all_results = {}
    si_series = {}
    
    # Compute SI for all assets first
    for name, (data, mtype) in assets.items():
        print(f"  Computing SI for {name}...")
        si_series[name] = compute_si(data)
    
    # ============================================================
    section("1. NONLINEAR CORRELATION (Mutual Information)")
    # ============================================================
    
    print("\n  Detecting nonlinear dependencies SI might have missed")
    
    mi_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = si_series[name]
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common].dropna()
        
        mi_scores = {}
        for feat in ['adx', 'rsi', 'volatility']:
            feat_aligned = features.loc[common, feat].dropna()
            common_final = si_aligned.index.intersection(feat_aligned.index)
            
            if len(common_final) > 100:
                # Linear correlation
                r, _ = spearmanr(si_aligned.loc[common_final], feat_aligned.loc[common_final])
                # Mutual information
                mi = mutual_information_proxy(si_aligned.loc[common_final].values, 
                                              feat_aligned.loc[common_final].values)
                
                mi_scores[feat] = {
                    'spearman': float(r),
                    'mutual_info': float(mi),
                    'nonlinear_ratio': float(mi / (abs(r) + 0.01)),
                }
                
                print(f"    {feat:<12} Spearman: {r:+.3f}, MI: {mi:.3f}, NL ratio: {mi/(abs(r)+0.01):.2f}")
                
                if mi > 0.3 and abs(r) < 0.15:
                    discover(f"Strong nonlinear SI-{feat} relationship in {name} (MI={mi:.2f})")
        
        mi_results[name] = mi_scores
    
    all_results['mutual_info'] = mi_results
    
    # ============================================================
    section("2. CROSS-ASSET SI SIMILARITY (DTW)")
    # ============================================================
    
    print("\n  How similar are SI patterns across assets?")
    
    dtw_results = {}
    asset_names = list(si_series.keys())
    
    for i, name1 in enumerate(asset_names):
        for name2 in asset_names[i+1:]:
            si1 = si_series[name1].dropna()
            si2 = si_series[name2].dropna()
            
            common = si1.index.intersection(si2.index)
            if len(common) < 100:
                continue
            
            # Normalize
            s1 = (si1.loc[common].values - si1.loc[common].mean()) / si1.loc[common].std()
            s2 = (si2.loc[common].values - si2.loc[common].mean()) / si2.loc[common].std()
            
            # Subsample for speed
            step = max(1, len(s1) // 500)
            s1_sub = s1[::step]
            s2_sub = s2[::step]
            
            dtw_dist = dtw_distance(s1_sub, s2_sub)
            euclidean = np.sqrt(np.sum((s1_sub - s2_sub)**2))
            
            dtw_results[f'{name1}-{name2}'] = {
                'dtw_distance': float(dtw_dist),
                'euclidean': float(euclidean),
                'dtw_ratio': float(dtw_dist / (euclidean + 0.01)),
            }
            
            print(f"    {name1}-{name2}: DTW={dtw_dist:.1f}, Euclidean={euclidean:.1f}, Ratio={dtw_dist/euclidean:.2f}")
            
            if dtw_dist < euclidean * 0.8:
                discover(f"SI patterns are time-warped similar: {name1}-{name2}")
    
    all_results['dtw'] = dtw_results
    
    # ============================================================
    section("3. ROLLING BETA STABILITY")
    # ============================================================
    
    print("\n  How stable is SI-feature relationship over time?")
    
    beta_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = si_series[name]
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common].dropna()
        
        for feat in ['adx', 'volatility']:
            feat_aligned = features.loc[common, feat].dropna()
            common_final = si_aligned.index.intersection(feat_aligned.index)
            
            if len(common_final) < 120:
                continue
            
            betas = rolling_beta(si_aligned.loc[common_final].values, 
                                feat_aligned.loc[common_final].values, 
                                window=60)
            
            valid_betas = betas[~np.isnan(betas)]
            if len(valid_betas) > 0:
                mean_beta = np.mean(valid_betas)
                std_beta = np.std(valid_betas)
                stability = abs(mean_beta) / (std_beta + 0.001)
                
                # Sign flips
                sign_changes = np.sum(np.diff(np.sign(valid_betas)) != 0)
                sign_flip_rate = sign_changes / len(valid_betas)
                
                if name not in beta_results:
                    beta_results[name] = {}
                    
                beta_results[name][feat] = {
                    'mean_beta': float(mean_beta),
                    'std_beta': float(std_beta),
                    'stability': float(stability),
                    'sign_flip_rate': float(sign_flip_rate),
                }
                
                print(f"    SI~{feat}: β={mean_beta:+.4f}±{std_beta:.4f}, stability={stability:.2f}, flips={sign_flip_rate:.0%}")
                
                if sign_flip_rate > 0.3:
                    discover(f"SI-{feat} relationship is UNSTABLE in {name} (flips {sign_flip_rate:.0%})")
    
    all_results['beta_stability'] = beta_results
    
    # ============================================================
    section("4. REGIME TRANSITION ANALYSIS")
    # ============================================================
    
    print("\n  SI regime transition probabilities")
    
    transition_results = {}
    
    for name in si_series:
        print(f"\n  {name}:")
        si = si_series[name]
        
        trans = analyze_regime_transitions(si, n_regimes=3)
        transition_results[name] = trans
        
        print(f"    Transition Matrix (Low/Med/High):")
        matrix = np.array(trans['transition_matrix'])
        for i, row in enumerate(matrix):
            regime = ['Low', 'Med', 'High'][i]
            row_str = ' '.join([f'{v:.2f}' for v in row])
            print(f"      {regime}: [{row_str}]")
        
        print(f"    Persistence: Low={trans['persistence'][0]:.2f}, Med={trans['persistence'][1]:.2f}, High={trans['persistence'][2]:.2f}")
        
        if trans['persistence'][2] > 0.8:
            discover(f"High-SI regime is very sticky in {name} (P={trans['persistence'][2]:.0%})")
    
    all_results['transitions'] = transition_results
    
    # ============================================================
    section("5. SI PREDICTABILITY (Out-of-Sample)")
    # ============================================================
    
    print("\n  Can we predict SI using features?")
    
    predict_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = si_series[name]
        features = compute_features(data)
        
        predictability = si_predictability_test(si, features, horizons=[1, 5, 10])
        predict_results[name] = predictability
        
        for horizon, stats in predictability.items():
            r2 = stats['mean_oos_r2']
            print(f"    {horizon}: OOS R² = {r2:.3f} ± {stats['std_oos_r2']:.3f}")
            
            if r2 > 0.1:
                discover(f"SI is predictable at {horizon} in {name} (R²={r2:.2f})")
    
    all_results['predictability'] = predict_results
    
    # ============================================================
    section("6. SI MEAN REVERSION SPEED")
    # ============================================================
    
    print("\n  How fast does SI revert to mean?")
    
    reversion_results = {}
    
    for name in si_series:
        print(f"\n  {name}:")
        si = si_series[name].dropna()
        
        # Compute deviation from mean
        mean_si = si.mean()
        deviation = si - mean_si
        
        # Check next-period deviation
        next_dev = deviation.shift(-1).dropna()
        curr_dev = deviation.iloc[:-1]
        
        # AR(1) coefficient
        ar1_coef, _ = pearsonr(curr_dev, next_dev)
        
        # Half-life
        half_life = -np.log(2) / np.log(abs(ar1_coef)) if abs(ar1_coef) < 1 else np.inf
        
        reversion_results[name] = {
            'ar1_coefficient': float(ar1_coef),
            'half_life_days': float(half_life),
            'mean_si': float(mean_si),
            'std_si': float(si.std()),
        }
        
        print(f"    AR(1) coefficient: {ar1_coef:.3f}")
        print(f"    Mean reversion half-life: {half_life:.1f} days")
        
        if half_life < 5:
            discover(f"SI mean reverts quickly in {name} (half-life={half_life:.1f} days)")
        elif half_life > 20:
            discover(f"SI is very persistent in {name} (half-life={half_life:.1f} days)")
    
    all_results['mean_reversion'] = reversion_results
    
    # ============================================================
    section("7. SI EXTREMES ANALYSIS")
    # ============================================================
    
    print("\n  What happens after SI extremes?")
    
    extremes_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = si_series[name]
        returns = data['close'].pct_change()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        # Future 5-day return
        future_5d = returns_aligned.rolling(5).sum().shift(-5)
        
        common_final = si_aligned.index.intersection(future_5d.dropna().index)
        
        si_vals = si_aligned.loc[common_final]
        ret_vals = future_5d.loc[common_final]
        
        # SI extremes (top/bottom 10%)
        low_thresh = np.percentile(si_vals, 10)
        high_thresh = np.percentile(si_vals, 90)
        
        low_si_ret = ret_vals[si_vals < low_thresh].mean()
        high_si_ret = ret_vals[si_vals > high_thresh].mean()
        mid_si_ret = ret_vals[(si_vals >= low_thresh) & (si_vals <= high_thresh)].mean()
        
        extremes_results[name] = {
            'low_si_future_ret': float(low_si_ret),
            'high_si_future_ret': float(high_si_ret),
            'mid_si_future_ret': float(mid_si_ret),
            'extreme_spread': float(high_si_ret - low_si_ret),
        }
        
        print(f"    After Low SI:  5d return = {low_si_ret:+.2%}")
        print(f"    After High SI: 5d return = {high_si_ret:+.2%}")
        print(f"    After Mid SI:  5d return = {mid_si_ret:+.2%}")
        print(f"    Extreme spread: {(high_si_ret - low_si_ret):+.2%}")
        
        if abs(high_si_ret - low_si_ret) > 0.005:
            discover(f"SI extremes predict different returns in {name} (spread={100*(high_si_ret-low_si_ret):.1f}bp)")
    
    all_results['extremes'] = extremes_results
    
    # ============================================================
    section("SUMMARY")
    # ============================================================
    
    print(f"\n  Total discoveries: {len(discoveries)}")
    for i, d in enumerate(discoveries, 1):
        print(f"    {i}. {d}")
    
    out_path = Path("results/phase1_round2_part3/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'discoveries': discoveries,
            'results': all_results,
        }, f, indent=2, default=str)
    
    discoveries_path = Path("docs/DISCOVERIES_LOG.md")
    with open(discoveries_path, 'a') as f:
        f.write(f"\n\n## Phase 1 Round 2 Part 3 - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for d in discoveries:
            f.write(f"- {d}\n")
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
