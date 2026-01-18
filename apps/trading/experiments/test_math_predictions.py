#!/usr/bin/env python3
"""
Test predictions from mathematical analysis of SI-ADX formulas.
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

MARKETS = {
    MarketType.CRYPTO: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    MarketType.FOREX: ['EURUSD', 'GBPUSD', 'USDJPY'],
    MarketType.STOCKS: ['SPY', 'QQQ', 'AAPL'],
}

def compute_features(data):
    """Compute all features including new ones from mathematical analysis."""
    features = pd.DataFrame(index=data.index)
    returns = data['close'].pct_change()
    features['returns'] = returns
    
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
    
    # Volatility
    features['volatility'] = returns.rolling(14).std()
    
    # RSI
    delta = data['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # NEW: |RSI - 50| (Discovery 2 from math analysis)
    features['rsi_extremity'] = abs(features['rsi'] - 50)
    
    # NEW: Market Clarity Index (Discovery 4)
    p_plus = plus_di / (plus_di + minus_di + 1e-10)
    p_minus = minus_di / (plus_di + minus_di + 1e-10)
    h_market = -(p_plus * np.log(p_plus + 1e-10) + p_minus * np.log(p_minus + 1e-10))
    features['mci'] = 1 - (h_market / np.log(2))
    
    # NEW: DX component
    features['dx'] = dx
    
    # NEW: Trend strength
    features['trend_strength'] = abs(returns.rolling(7).mean()) / (returns.rolling(7).std() + 1e-10)
    
    return features.dropna()

def compute_si(data, window=7):
    if len(data) < window * 3:
        return pd.Series(dtype=float)
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    try:
        population.run(data)
        return population.compute_si_timeseries(data, window=window)
    except:
        return pd.Series(dtype=float)

def safe_corr(si, feat):
    """Compute correlation with proper alignment."""
    common = si.index.intersection(feat.index)
    if len(common) < 50:
        return np.nan, np.nan
    s = si.loc[common].dropna()
    f = feat.loc[common].dropna()
    common2 = s.index.intersection(f.index)
    if len(common2) < 50:
        return np.nan, np.nan
    return spearmanr(s.loc[common2], f.loc[common2])

def main():
    print("\n" + "="*70)
    print("TESTING MATHEMATICAL PREDICTIONS")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("\nPredictions from formula analysis:")
    print("  P1: SI ~ |RSI - 50| (RSI extremity)")
    print("  P2: SI ~ MCI (Market Clarity Index)")
    print("  P3: SI ~ DX (before MA smoothing)")
    print("  P4: SI ~ Trend Strength")
    print("="*70)
    
    loader = DataLoaderV2()
    results = []
    
    for mtype, symbols in MARKETS.items():
        print(f"\n  [{mtype.value.upper()}]")
        for symbol in symbols:
            print(f"    {symbol}...", end=" ", flush=True)
            try:
                data = loader.load(symbol, mtype)
                if len(data) < 200:
                    print("skip (short)")
                    continue
                
                si = compute_si(data)
                if len(si) < 100:
                    print("skip (no SI)")
                    continue
                
                features = compute_features(data)
                
                asset_result = {'asset': symbol, 'market': mtype.value}
                
                # Test all predictions
                for feat in ['rsi_extremity', 'mci', 'dx', 'trend_strength', 'adx', 'volatility']:
                    if feat in features.columns:
                        r, p = safe_corr(si, features[feat])
                        if not np.isnan(r):
                            asset_result[feat] = {'r': float(r), 'p': float(p)}
                
                results.append(asset_result)
                print(f"done (n={len(si)})")
                
            except Exception as e:
                print(f"error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS: SI CORRELATIONS")
    print("="*70)
    
    features_to_check = ['rsi_extremity', 'mci', 'dx', 'trend_strength', 'adx', 'volatility']
    
    print(f"\n  {'Feature':<20} {'Mean r':>10} {'Consistent':>12} {'Prediction':>12}")
    print("  " + "-"*56)
    
    predictions = {
        'rsi_extremity': ('P1', 'positive'),
        'mci': ('P2', 'positive'),
        'dx': ('P3', 'positive'),
        'trend_strength': ('P4', 'positive'),
        'adx': ('known', 'positive'),
        'volatility': ('known', 'negative'),
    }
    
    summary = {}
    for feat in features_to_check:
        rs = [r[feat]['r'] for r in results if feat in r]
        if not rs:
            continue
        mean_r = np.mean(rs)
        consistent = all(x > 0 for x in rs) or all(x < 0 for x in rs)
        pred_name, pred_sign = predictions.get(feat, ('?', '?'))
        
        # Check if prediction confirmed
        if pred_sign == 'positive':
            confirmed = mean_r > 0.05 and consistent
        elif pred_sign == 'negative':
            confirmed = mean_r < -0.05 and consistent
        else:
            confirmed = False
        
        status = "✅ CONFIRMED" if confirmed else ("⚠️ WEAK" if abs(mean_r) > 0.05 else "❌ FAILED")
        print(f"  {feat:<20} {mean_r:>+10.3f} {'YES' if consistent else 'NO':>12} {status:>12}")
        summary[feat] = {'mean_r': mean_r, 'consistent': consistent, 'confirmed': confirmed}
    
    print("\n" + "="*70)
    print("NEW DISCOVERIES")
    print("="*70)
    
    # Find strongest NEW correlations
    new_features = ['rsi_extremity', 'mci', 'dx', 'trend_strength']
    
    print("\n  Comparing new features to ADX (baseline):")
    adx_mean = summary.get('adx', {}).get('mean_r', 0)
    
    for feat in new_features:
        if feat in summary:
            mean_r = summary[feat]['mean_r']
            stronger = abs(mean_r) > abs(adx_mean)
            mark = "🔥 STRONGER" if stronger else ""
            print(f"    {feat:<20}: r={mean_r:+.3f} (ADX={adx_mean:+.3f}) {mark}")
    
    print("\n  KEY INSIGHTS:")
    
    # Check MCI - the theoretically important one
    if 'mci' in summary:
        mci_r = summary['mci']['mean_r']
        if mci_r > 0.1 and summary['mci']['consistent']:
            print("    🎯 MCI strongly correlates with SI - confirms entropy connection!")
        elif mci_r > 0:
            print("    📊 MCI positively correlates with SI - entropy connection supported")
        else:
            print("    ❌ MCI does not correlate as expected")
    
    # Check RSI extremity
    if 'rsi_extremity' in summary:
        rsi_r = summary['rsi_extremity']['mean_r']
        if rsi_r > 0.1:
            print("    🎯 RSI extremity correlates with SI - momentum-specialization link confirmed!")
    
    # Save results
    out_path = Path("results/math_predictions/test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'per_asset': results,
        }, f, indent=2)
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
