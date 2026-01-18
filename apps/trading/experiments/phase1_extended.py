#!/usr/bin/env python3
"""
PHASE 1 EXTENDED EXPLORATION
- Nonlinear methods (Mutual Information, Copula)
- SI window sensitivity
- Reverse causality tests
- Lead-lag macro analysis
- More assets
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests
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

def compute_basic_features(data):
    features = pd.DataFrame(index=data.index)
    returns = data['close'].pct_change()
    
    # ADX
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
    
    # RSI Extremity
    delta = data['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    features['rsi_extremity'] = abs(rsi - 50)
    
    # Volatility
    features['volatility'] = returns.rolling(14).std()
    
    # Returns
    features['returns'] = returns
    
    return features.dropna()

def mutual_information(x, y):
    """Compute mutual information between two series."""
    x_clean = x.dropna()
    y_clean = y.dropna()
    common = x_clean.index.intersection(y_clean.index)
    if len(common) < 50:
        return np.nan
    
    x_vals = x_clean.loc[common].values.reshape(-1, 1)
    y_vals = y_clean.loc[common].values
    
    try:
        mi = mutual_info_regression(x_vals, y_vals, random_state=42)[0]
        return mi
    except:
        return np.nan

def test_granger_both_directions(si, feature, max_lag=5):
    """Test Granger causality in both directions."""
    common = si.index.intersection(feature.index)
    si_aligned = si.loc[common].dropna()
    feat_aligned = feature.loc[common].dropna()
    common2 = si_aligned.index.intersection(feat_aligned.index)
    
    if len(common2) < 50:
        return None, None
    
    df = pd.DataFrame({
        'si': si_aligned.loc[common2],
        'feat': feat_aligned.loc[common2]
    }).dropna()
    
    if len(df) < 50:
        return None, None
    
    results = {'si_causes_feat': None, 'feat_causes_si': None}
    
    try:
        res1 = grangercausalitytests(df[['feat', 'si']], maxlag=max_lag, verbose=False)
        min_p1 = min(res1[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1))
        results['si_causes_feat'] = min_p1 < 0.05
        
        res2 = grangercausalitytests(df[['si', 'feat']], maxlag=max_lag, verbose=False)
        min_p2 = min(res2[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1))
        results['feat_causes_si'] = min_p2 < 0.05
        
        return results['si_causes_feat'], results['feat_causes_si']
    except:
        return None, None

def main():
    print("\n" + "="*70)
    print("  PHASE 1 EXTENDED EXPLORATION")
    print("="*70)
    print(f"  Time: {datetime.now().isoformat()}")
    
    loader = DataLoaderV2()
    
    # Extended asset list
    assets = {
        'BTCUSDT': (loader.load('BTCUSDT', MarketType.CRYPTO), MarketType.CRYPTO),
        'ETHUSDT': (loader.load('ETHUSDT', MarketType.CRYPTO), MarketType.CRYPTO),
        'SOLUSDT': (loader.load('SOLUSDT', MarketType.CRYPTO), MarketType.CRYPTO),
        'SPY': (loader.load('SPY', MarketType.STOCKS), MarketType.STOCKS),
        'QQQ': (loader.load('QQQ', MarketType.STOCKS), MarketType.STOCKS),
        'AAPL': (loader.load('AAPL', MarketType.STOCKS), MarketType.STOCKS),
        'EURUSD': (loader.load('EURUSD', MarketType.FOREX), MarketType.FOREX),
        'GBPUSD': (loader.load('GBPUSD', MarketType.FOREX), MarketType.FOREX),
        'USDJPY': (loader.load('USDJPY', MarketType.FOREX), MarketType.FOREX),
    }
    
    all_results = {}
    
    # ============================================================
    section("1. SI WINDOW SENSITIVITY")
    # ============================================================
    
    print("\n  Testing SI windows: 5, 7, 10, 14, 21, 30 days")
    
    windows = [5, 7, 10, 14, 21, 30]
    window_results = {}
    
    for name, (data, mtype) in list(assets.items())[:3]:
        print(f"\n  {name}:")
        features = compute_basic_features(data)
        
        asset_window_results = []
        for window in windows:
            print(f"    Window={window}...", end=" ", flush=True)
            si = compute_si(data, window=window)
            
            common = si.index.intersection(features.index)
            if len(common) < 100:
                print("skip")
                continue
            
            si_aligned = si.loc[common]
            adx = features.loc[common, 'adx']
            rsi_ext = features.loc[common, 'rsi_extremity']
            
            r_adx, _ = spearmanr(si_aligned, adx)
            r_rsi, _ = spearmanr(si_aligned, rsi_ext)
            
            asset_window_results.append({
                'window': window,
                'r_adx': float(r_adx),
                'r_rsi_ext': float(r_rsi),
            })
            print(f"ADX={r_adx:.3f}, RSI_ext={r_rsi:.3f}")
        
        window_results[name] = asset_window_results
        
        if asset_window_results:
            best = max(asset_window_results, key=lambda x: abs(x['r_adx']))
            discover(f"Optimal SI window for {name}: {best['window']} days (r_adx={best['r_adx']:.3f})")
    
    all_results['window_sensitivity'] = window_results
    
    # ============================================================
    section("2. NONLINEAR ANALYSIS (Mutual Information)")
    # ============================================================
    
    print("\n  Comparing linear (Spearman) vs nonlinear (MI) relationships")
    
    mi_results = {}
    
    for name, (data, mtype) in list(assets.items())[:3]:
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_basic_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        
        asset_mi = {}
        for feat in ['adx', 'rsi_extremity', 'volatility']:
            feat_aligned = features.loc[common, feat]
            
            r, _ = spearmanr(si_aligned, feat_aligned)
            mi = mutual_information(si_aligned, feat_aligned)
            
            mi_str = f"{mi:.4f}" if not np.isnan(mi) else "N/A"
            
            asset_mi[feat] = {
                'spearman': float(r),
                'mutual_info': float(mi) if not np.isnan(mi) else None,
            }
            
            print(f"    {feat}: Spearman={r:.3f}, MI={mi_str}")
            
            if mi is not None and not np.isnan(mi):
                if mi > 0.1 and abs(r) < 0.1:
                    discover(f"Nonlinear relationship: SI-{feat} in {name} (MI={mi:.3f}, r={r:.3f})")
        
        mi_results[name] = asset_mi
    
    all_results['mutual_information'] = mi_results
    
    # ============================================================
    section("3. REVERSE CAUSALITY TESTS")
    # ============================================================
    
    print("\n  Testing: Does SI cause features, or vice versa?")
    
    causality_results = {}
    
    for name, (data, mtype) in list(assets.items())[:3]:
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_basic_features(data)
        
        asset_causality = {}
        for feat in ['adx', 'rsi_extremity', 'volatility']:
            si_causes, feat_causes = test_granger_both_directions(si, features[feat])
            
            asset_causality[feat] = {
                'si_causes_feat': si_causes,
                'feat_causes_si': feat_causes,
            }
            
            if si_causes and not feat_causes:
                direction = "SI → " + feat
            elif feat_causes and not si_causes:
                direction = feat + " → SI"
            elif si_causes and feat_causes:
                direction = "Bidirectional"
            else:
                direction = "No causality"
            
            print(f"    {feat}: {direction}")
            
            if feat_causes and not si_causes:
                discover(f"Reverse causality: {feat} → SI in {name}")
        
        causality_results[name] = asset_causality
    
    all_results['causality'] = causality_results
    
    # ============================================================
    section("4. LEAD-LAG MACRO ANALYSIS")
    # ============================================================
    
    print("\n  Testing SI vs lagged macro features")
    
    macro_dir = Path("data/macro")
    macro_data = {}
    
    for filepath in macro_dir.glob("*_1d.csv"):
        name = filepath.stem.replace("_1d", "")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if 'close' in df.columns:
            macro_data[name] = df['close']
    
    if macro_data:
        print(f"  Loaded {len(macro_data)} macro indicators")
        
        macro_lag_results = {}
        
        for asset_name, (data, mtype) in list(assets.items())[:2]:
            print(f"\n  {asset_name}:")
            si = compute_si(data)
            
            asset_macro = {}
            for macro_name, macro_series in macro_data.items():
                common = si.index.intersection(macro_series.index)
                if len(common) < 100:
                    continue
                
                si_aligned = si.loc[common]
                macro_aligned = macro_series.loc[common]
                
                best_lag = 0
                best_r = 0
                
                for lag in range(-10, 11):
                    if lag < 0:
                        macro_shifted = macro_aligned.shift(-lag)
                    else:
                        macro_shifted = macro_aligned.shift(lag)
                    
                    common_lag = si_aligned.dropna().index.intersection(macro_shifted.dropna().index)
                    if len(common_lag) < 50:
                        continue
                    
                    r, _ = spearmanr(si_aligned.loc[common_lag], macro_shifted.loc[common_lag])
                    
                    if abs(r) > abs(best_r):
                        best_r = r
                        best_lag = lag
                
                asset_macro[macro_name] = {
                    'best_lag': best_lag,
                    'best_r': float(best_r),
                }
                
                if abs(best_r) > 0.1:
                    lag_desc = f"SI leads by {-best_lag}d" if best_lag < 0 else f"Macro leads by {best_lag}d"
                    print(f"    {macro_name}: r={best_r:.3f} @ lag={best_lag} ({lag_desc})")
                    
                    if abs(best_r) > 0.15:
                        discover(f"SI-{macro_name} at lag {best_lag}: r={best_r:.3f}")
            
            macro_lag_results[asset_name] = asset_macro
        
        all_results['macro_lead_lag'] = macro_lag_results
    else:
        print("  No macro data found")
    
    # ============================================================
    section("5. TAIL DEPENDENCE (Extreme Events)")
    # ============================================================
    
    print("\n  How does SI behave before extreme market moves?")
    
    tail_results = {}
    
    for name, (data, mtype) in list(assets.items())[:3]:
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_basic_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        returns = features.loc[common, 'returns']
        
        extreme_down = returns < returns.quantile(0.05)
        extreme_up = returns > returns.quantile(0.95)
        normal = ~extreme_down & ~extreme_up
        
        si_before = si_aligned.shift(1)
        
        si_before_down = si_before[extreme_down].mean()
        si_before_up = si_before[extreme_up].mean()
        si_before_normal = si_before[normal].mean()
        
        tail_results[name] = {
            'si_before_extreme_down': float(si_before_down),
            'si_before_extreme_up': float(si_before_up),
            'si_before_normal': float(si_before_normal),
        }
        
        print(f"    SI before extreme DOWN: {si_before_down:.4f}")
        print(f"    SI before extreme UP:   {si_before_up:.4f}")
        print(f"    SI before normal:       {si_before_normal:.4f}")
        
        diff_down = abs(si_before_down - si_before_normal)
        diff_up = abs(si_before_up - si_before_normal)
        
        if diff_down > 0.002 or diff_up > 0.002:
            discover(f"SI differs before extreme moves in {name} (down: {diff_down:.4f}, up: {diff_up:.4f})")
    
    all_results['tail_dependence'] = tail_results
    
    # ============================================================
    section("6. CROSS-ASSET SI SYNCHRONIZATION")
    # ============================================================
    
    print("\n  Do assets specialize together?")
    
    si_all = {}
    for name, (data, mtype) in assets.items():
        print(f"  Computing SI for {name}...", end=" ", flush=True)
        si_all[name] = compute_si(data)
        print("done")
    
    print("\n  Pairwise SI Correlations (|r| > 0.2):")
    print(f"  {'Pair':<20} {'r':>10} {'Same Market':>12}")
    print("  " + "-"*45)
    
    sync_results = []
    asset_names = list(si_all.keys())
    
    for i in range(len(asset_names)):
        for j in range(i+1, len(asset_names)):
            a, b = asset_names[i], asset_names[j]
            si_a, si_b = si_all[a], si_all[b]
            
            common = si_a.index.intersection(si_b.index)
            if len(common) < 100:
                continue
            
            r, p = spearmanr(si_a.loc[common], si_b.loc[common])
            
            mtype_a = assets[a][1].value
            mtype_b = assets[b][1].value
            same_market = mtype_a == mtype_b
            
            sync_results.append({
                'pair': f"{a}-{b}",
                'r': float(r),
                'same_market': same_market,
            })
            
            if abs(r) > 0.2:
                mark = "✓" if same_market else "⚠️"
                print(f"  {a}-{b:<12} {r:>+10.3f} {mark:>12}")
                
                if abs(r) > 0.3:
                    discover(f"High SI sync: {a}-{b} r={r:.3f}")
    
    all_results['cross_asset_sync'] = sync_results
    
    # ============================================================
    section("SUMMARY")
    # ============================================================
    
    print(f"\n  Total discoveries: {len(discoveries)}")
    for i, d in enumerate(discoveries, 1):
        print(f"    {i}. {d}")
    
    out_path = Path("results/phase1_extended/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'discoveries': discoveries,
            'results': all_results,
        }, f, indent=2, default=str)
    
    discoveries_path = Path("docs/DISCOVERIES_LOG.md")
    with open(discoveries_path, 'a') as f:
        f.write(f"\n\n## Phase 1 Extended - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for d in discoveries:
            f.write(f"- {d}\n")
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
