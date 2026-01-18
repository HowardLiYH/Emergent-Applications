#!/usr/bin/env python3
"""
PHASE 1 EXPANSION - PARALLEL EXECUTION
1.3 Regime-Conditional Analysis
1.4 Stability Analysis  
1.5 Extended Features (Math + Empirical)
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from collections import defaultdict
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

def classify_regime(data, idx, lookback=20):
    """Classify market regime: bull, bear, or volatile."""
    if idx < lookback:
        return 'unknown'
    
    window = data.iloc[idx-lookback:idx]
    returns = window['close'].pct_change().dropna()
    
    cum_return = (window['close'].iloc[-1] / window['close'].iloc[0]) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Classification rules
    if volatility > 0.4:  # Annualized vol > 40%
        return 'volatile'
    elif cum_return > 0.05:  # >5% gain
        return 'bull'
    elif cum_return < -0.05:  # >5% loss
        return 'bear'
    else:
        return 'neutral'

def compute_extended_features(data):
    """Compute 20+ features with mathematical justification."""
    features = pd.DataFrame(index=data.index)
    returns = data['close'].pct_change()
    
    # === TREND INDICATORS ===
    # Mathematical basis: SI emerges when one direction dominates
    
    # 1. ADX (baseline)
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
    
    # 2. RSI Extremity
    delta = data['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    features['rsi'] = rsi
    features['rsi_extremity'] = abs(rsi - 50)
    
    # 3. MACD Signal Strength
    # Math: MACD measures trend momentum; strong signal = consistent winners
    ema12 = data['close'].ewm(span=12).mean()
    ema26 = data['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    features['macd_strength'] = abs(macd - signal) / data['close'] * 100
    
    # 4. Trend Strength (price-based)
    features['trend_strength'] = abs(returns.rolling(7).mean()) / (returns.rolling(7).std() + 1e-10)
    
    # 5. Directional Consistency
    # Math: High consistency = one strategy wins repeatedly = specialization
    up_days = (returns > 0).rolling(14).sum()
    features['dir_consistency'] = abs(up_days - 7) / 7  # 0 = balanced, 1 = all same direction
    
    # === MOMENTUM INDICATORS ===
    # Math: Strong momentum = trending market = high SI
    
    # 6. Rate of Change
    features['roc_7'] = (data['close'] / data['close'].shift(7) - 1) * 100
    features['roc_14'] = (data['close'] / data['close'].shift(14) - 1) * 100
    
    # 7. Momentum (absolute)
    features['momentum'] = data['close'] - data['close'].shift(10)
    
    # 8. Williams %R Extremity
    # Math: Like RSI, extreme values = directional dominance
    high_14 = data['high'].rolling(14).max()
    low_14 = data['low'].rolling(14).min()
    williams_r = (high_14 - data['close']) / (high_14 - low_14 + 1e-10) * -100
    features['williams_extremity'] = abs(williams_r + 50)
    
    # === VOLATILITY INDICATORS ===
    # Math: Low volatility = stable regime = consistent winners = high SI
    
    # 9. ATR Normalized
    features['atr_norm'] = atr / data['close'] * 100
    
    # 10. Bollinger Band Width
    sma20 = data['close'].rolling(20).mean()
    std20 = data['close'].rolling(20).std()
    features['bb_width'] = (std20 * 4) / sma20 * 100
    
    # 11. Realized Volatility
    features['realized_vol'] = returns.rolling(14).std() * np.sqrt(252)
    
    # 12. Volatility of Volatility
    vol = returns.rolling(7).std()
    features['vol_of_vol'] = vol.rolling(14).std()
    
    # === MEAN REVERSION INDICATORS ===
    # Math: Strong mean reversion = mean-rev strategy wins = different SI behavior
    
    # 13. Bollinger %B
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    features['bb_pct'] = (data['close'] - lower) / (upper - lower + 1e-10)
    features['bb_extremity'] = abs(features['bb_pct'] - 0.5) * 2
    
    # 14. Distance from MA
    features['dist_from_ma20'] = (data['close'] - sma20) / sma20 * 100
    
    # 15. Mean Reversion Score
    # Math: Price far from MA but low momentum = mean reversion opportunity
    features['mr_score'] = abs(features['dist_from_ma20']) / (abs(features['roc_7']) + 1)
    
    # === VOLUME INDICATORS ===
    # Math: Volume confirms trend; high volume trend = consistent winners
    
    if 'volume' in data.columns and data['volume'].sum() > 0:
        # 16. Volume Trend
        vol_ma = data['volume'].rolling(20).mean()
        features['vol_ratio'] = data['volume'] / (vol_ma + 1)
        
        # 17. OBV Trend
        obv = (np.sign(returns) * data['volume']).cumsum()
        features['obv_trend'] = obv.diff(7) / (data['volume'].rolling(7).sum() + 1)
        
        # 18. Volume-Price Trend
        features['vpt'] = ((returns * data['volume']).rolling(14).sum() / 
                          (data['volume'].rolling(14).sum() + 1))
    
    # === MARKET STRUCTURE ===
    
    # 19. Higher Highs / Lower Lows ratio
    hh = (data['high'] > data['high'].shift(1)).rolling(14).sum()
    ll = (data['low'] < data['low'].shift(1)).rolling(14).sum()
    features['hh_ll_ratio'] = (hh - ll) / 14
    
    # 20. Price Efficiency (Kaufman)
    # Math: High efficiency = trending = specialization
    change = abs(data['close'] - data['close'].shift(10))
    volatility_sum = abs(returns).rolling(10).sum()
    features['efficiency'] = change / (volatility_sum * data['close'] + 1e-10)
    
    # 21. Fractal Dimension (approximation)
    # Math: Low FD = smooth trend = high SI
    range_high = data['high'].rolling(14).max() - data['low'].rolling(14).min()
    range_low = (data['high'].rolling(7).max() - data['low'].rolling(7).min()).rolling(2).sum()
    features['fractal_dim'] = np.log(2) / np.log(2 * range_high / (range_low + 1e-10) + 1e-10)
    
    # 22. Autocorrelation of returns
    features['return_autocorr'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1), raw=False)
    
    return features.dropna()

def main():
    print("\n" + "="*70)
    print("  PHASE 1 PARALLEL EXECUTION")
    print("="*70)
    print(f"  Time: {datetime.now().isoformat()}")
    
    loader = DataLoaderV2()
    
    # Load all assets
    assets = {
        'BTCUSDT': (loader.load('BTCUSDT', MarketType.CRYPTO), MarketType.CRYPTO),
        'ETHUSDT': (loader.load('ETHUSDT', MarketType.CRYPTO), MarketType.CRYPTO),
        'SPY': (loader.load('SPY', MarketType.STOCKS), MarketType.STOCKS),
        'QQQ': (loader.load('QQQ', MarketType.STOCKS), MarketType.STOCKS),
        'EURUSD': (loader.load('EURUSD', MarketType.FOREX), MarketType.FOREX),
    }
    
    # Compute SI for all
    si_dict = {}
    for name, (data, mtype) in assets.items():
        print(f"\n  Computing SI for {name}...")
        si_dict[name] = compute_si(data)
    
    all_results = {}
    
    # ============================================================
    section("1.3 REGIME-CONDITIONAL ANALYSIS")
    # ============================================================
    
    print("\n  Does SI-feature correlation change across market regimes?")
    
    regime_results = defaultdict(lambda: defaultdict(list))
    
    for name, (data, mtype) in list(assets.items())[:3]:  # Top 3 assets
        si = si_dict[name]
        features = compute_extended_features(data)
        
        # Classify regimes
        regimes = pd.Series(index=data.index, dtype=str)
        for i in range(len(data)):
            regimes.iloc[i] = classify_regime(data, i)
        
        # Align
        common = si.index.intersection(features.index).intersection(regimes.index)
        si_aligned = si.loc[common]
        features_aligned = features.loc[common]
        regimes_aligned = regimes.loc[common]
        
        print(f"\n  {name}:")
        print(f"  {'Regime':<10} {'Count':>8} {'SI-ADX':>10} {'SI-RSI_ext':>12}")
        print("  " + "-"*42)
        
        for regime in ['bull', 'bear', 'volatile', 'neutral']:
            mask = regimes_aligned == regime
            if mask.sum() < 30:
                continue
            
            si_regime = si_aligned[mask]
            feat_regime = features_aligned[mask]
            
            r_adx, _ = spearmanr(si_regime, feat_regime['adx'])
            r_rsi, _ = spearmanr(si_regime, feat_regime['rsi_extremity'])
            
            print(f"  {regime:<10} {mask.sum():>8} {r_adx:>+10.3f} {r_rsi:>+12.3f}")
            
            regime_results[name][regime] = {'adx': r_adx, 'rsi_ext': r_rsi, 'count': int(mask.sum())}
    
    # Check for sign flips
    print("\n  Sign Flip Analysis:")
    for name in regime_results:
        adx_signs = [v['adx'] > 0 for k, v in regime_results[name].items() if v.get('adx')]
        if len(set(adx_signs)) > 1:
            discover(f"SI-ADX sign FLIPS across regimes in {name}!")
        else:
            print(f"    {name}: SI-ADX consistent across regimes ✓")
    
    all_results['regime_conditional'] = dict(regime_results)
    
    # ============================================================
    section("1.4 STABILITY ANALYSIS")
    # ============================================================
    
    print("\n  Testing correlation stability over time...")
    
    stability_results = {}
    
    for name, (data, mtype) in list(assets.items())[:3]:
        si = si_dict[name]
        features = compute_extended_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        features_aligned = features.loc[common]
        
        n = len(si_aligned)
        
        # Rolling correlation (1-year window)
        window = min(252, n // 3)
        rolling_corrs = []
        
        for i in range(window, n, 20):  # Step by 20 days
            si_window = si_aligned.iloc[i-window:i]
            adx_window = features_aligned['adx'].iloc[i-window:i]
            r, _ = spearmanr(si_window, adx_window)
            rolling_corrs.append({'date': si_aligned.index[i], 'r': r})
        
        if rolling_corrs:
            rs = [x['r'] for x in rolling_corrs]
            stability_results[name] = {
                'mean_r': float(np.mean(rs)),
                'std_r': float(np.std(rs)),
                'min_r': float(np.min(rs)),
                'max_r': float(np.max(rs)),
                'all_positive': all(r > 0 for r in rs),
                'all_negative': all(r < 0 for r in rs),
            }
            
            print(f"\n  {name}:")
            print(f"    Rolling SI-ADX: mean={np.mean(rs):.3f}, std={np.std(rs):.3f}")
            print(f"    Range: [{np.min(rs):.3f}, {np.max(rs):.3f}]")
            print(f"    Consistent sign: {'YES ✓' if stability_results[name]['all_positive'] or stability_results[name]['all_negative'] else 'NO (sign flips!)'}")
            
            if stability_results[name]['all_positive']:
                discover(f"SI-ADX correlation STABLE over time in {name}")
    
    # Subperiod analysis
    print("\n  Subperiod Analysis (first half vs second half):")
    
    for name, (data, mtype) in list(assets.items())[:3]:
        si = si_dict[name]
        features = compute_extended_features(data)
        
        common = si.index.intersection(features.index)
        n = len(common)
        mid = n // 2
        
        si_first = si.loc[common[:mid]]
        si_second = si.loc[common[mid:]]
        adx_first = features.loc[common[:mid], 'adx']
        adx_second = features.loc[common[mid:], 'adx']
        
        r1, _ = spearmanr(si_first, adx_first)
        r2, _ = spearmanr(si_second, adx_second)
        
        print(f"    {name}: first_half r={r1:.3f}, second_half r={r2:.3f}, diff={abs(r1-r2):.3f}")
        
        if r1 > 0 and r2 > 0:
            discover(f"SI-ADX stable across time periods in {name} ({r1:.3f} → {r2:.3f})")
    
    all_results['stability'] = stability_results
    
    # ============================================================
    section("1.5 EXTENDED FEATURES (Math + Empirical)")
    # ============================================================
    
    print("\n  Testing 22 features with mathematical justification...")
    
    feature_results = defaultdict(dict)
    
    # Feature mathematical basis
    math_basis = {
        'adx': 'Trend strength → consistent winners → specialization',
        'rsi_extremity': 'Directional imbalance → winner dominance',
        'macd_strength': 'Momentum strength → trend persistence',
        'trend_strength': 'Return/volatility ratio → signal clarity',
        'dir_consistency': 'Up/down ratio → winner consistency',
        'roc_7': 'Short-term momentum → trend regime',
        'roc_14': 'Medium-term momentum → trend persistence',
        'williams_extremity': 'Like RSI - extreme values = dominance',
        'atr_norm': 'Volatility inverse → stable regime',
        'bb_width': 'Volatility proxy → regime stability',
        'realized_vol': 'Direct volatility → inverse relationship',
        'bb_extremity': 'Price at bands → directional extreme',
        'dist_from_ma20': 'Trend deviation → momentum',
        'efficiency': 'Kaufman efficiency → trend quality',
        'hh_ll_ratio': 'Trend structure → direction clarity',
        'return_autocorr': 'Persistence → predictability',
    }
    
    for name, (data, mtype) in list(assets.items())[:3]:
        si = si_dict[name]
        features = compute_extended_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        features_aligned = features.loc[common]
        
        asset_results = []
        
        for feat in features_aligned.columns:
            if features_aligned[feat].isna().sum() > len(features_aligned) * 0.5:
                continue
            
            r, p = spearmanr(si_aligned.dropna(), features_aligned[feat].dropna())
            
            if not np.isnan(r):
                asset_results.append({
                    'feature': feat,
                    'r': float(r),
                    'p': float(p),
                    'significant': p < 0.05,
                    'math_basis': math_basis.get(feat, 'N/A')
                })
        
        # Sort by absolute correlation
        asset_results.sort(key=lambda x: abs(x['r']), reverse=True)
        feature_results[name] = asset_results
    
    # Print top features per asset
    print("\n  TOP 10 FEATURES BY CORRELATION:")
    
    for name in feature_results:
        print(f"\n  {name}:")
        print(f"  {'Rank':<5} {'Feature':<20} {'r':>10} {'Sig':>5} {'Mathematical Basis':<40}")
        print("  " + "-"*85)
        
        for i, feat in enumerate(feature_results[name][:10], 1):
            sig = "✓" if feat['significant'] else ""
            print(f"  {i:<5} {feat['feature']:<20} {feat['r']:>+10.3f} {sig:>5} {feat['math_basis'][:40]:<40}")
    
    # Cross-asset consistency
    print("\n  CROSS-ASSET CONSISTENCY:")
    
    feature_consistency = defaultdict(list)
    for name in feature_results:
        for feat_data in feature_results[name]:
            feature_consistency[feat_data['feature']].append(feat_data['r'])
    
    print(f"\n  {'Feature':<20} {'Mean r':>10} {'Consistent':>12} {'Sig Count':>10}")
    print("  " + "-"*55)
    
    consistent_features = []
    for feat, rs in sorted(feature_consistency.items(), key=lambda x: abs(np.mean(x[1])), reverse=True):
        if len(rs) >= 3:
            mean_r = np.mean(rs)
            consistent = all(r > 0 for r in rs) or all(r < 0 for r in rs)
            sig_count = sum(1 for r in rs if abs(r) > 0.1)
            
            if abs(mean_r) > 0.05:
                mark = "✓" if consistent else ""
                print(f"  {feat:<20} {mean_r:>+10.3f} {mark:>12} {sig_count:>10}")
                
                if consistent and abs(mean_r) > 0.1:
                    consistent_features.append(feat)
                    discover(f"Feature '{feat}' consistently correlates with SI (r={mean_r:.3f})")
    
    all_results['extended_features'] = dict(feature_results)
    all_results['consistent_features'] = consistent_features
    
    # ============================================================
    section("SUMMARY")
    # ============================================================
    
    print(f"\n  Total discoveries: {len(discoveries)}")
    for i, d in enumerate(discoveries, 1):
        print(f"    {i}. {d}")
    
    # Save results
    out_path = Path("results/phase1_parallel/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert defaultdict to regular dict for JSON
    def convert_to_serializable(obj):
        if isinstance(obj, defaultdict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return obj
    
    with open(out_path, 'w') as f:
        json.dump(convert_to_serializable({
            'timestamp': datetime.now().isoformat(),
            'discoveries': discoveries,
            'results': all_results,
        }), f, indent=2, default=str)
    
    # Update discoveries log
    discoveries_path = Path("docs/DISCOVERIES_LOG.md")
    with open(discoveries_path, 'a') as f:
        f.write(f"\n\n## Phase 1 Parallel Execution - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for d in discoveries:
            f.write(f"- {d}\n")
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
