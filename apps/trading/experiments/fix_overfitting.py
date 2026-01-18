#!/usr/bin/env python3
"""
FIX OVERFITTING: Test simpler strategy without threshold optimization.
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
TRANSACTION_COSTS = {MarketType.CRYPTO: 0.001, MarketType.FOREX: 0.0002, MarketType.STOCKS: 0.0005}
TRAIN_RATIO = 0.7

def compute_si_full(data, window=7):
    if len(data) < window * 3:
        return pd.Series(dtype=float)
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    try:
        population.run(data)
        return population.compute_si_timeseries(data, window=window)
    except:
        return pd.Series(dtype=float)

def strategy_no_optimization(si, data, train_end, cost):
    """
    Simple strategy: use fixed 50th percentile threshold (median).
    No optimization on training data.
    """
    returns = data['close'].pct_change().shift(-2)  # 1-day delay
    common = si.index.intersection(returns.index)
    df = pd.DataFrame({'si': si.loc[common], 'ret': returns.loc[common]}).dropna()
    
    if len(df) < 100:
        return {'validated': False, 'reason': 'insufficient data'}
    
    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end:]
    
    if len(train_df) < 50 or len(test_df) < 50:
        return {'validated': False, 'reason': f'train={len(train_df)}, test={len(test_df)}'}
    
    # FIXED threshold: median of training SI
    threshold = train_df['si'].median()
    
    # Train performance
    sig_train = (train_df['si'] >= threshold).astype(int)
    pos_train = sig_train.shift(1).fillna(0)
    net_train = pos_train * train_df['ret'] - pos_train.diff().abs().fillna(0) * cost
    train_sharpe = net_train.mean() / net_train.std() * np.sqrt(252) if net_train.std() > 0 else 0
    
    # Test performance
    sig_test = (test_df['si'] >= threshold).astype(int)
    pos_test = sig_test.shift(1).fillna(0)
    net_test = pos_test * test_df['ret'] - pos_test.diff().abs().fillna(0) * cost
    test_sharpe = net_test.mean() / net_test.std() * np.sqrt(252) if net_test.std() > 0 else 0
    test_ret = (1 + net_test).prod() - 1
    bh_ret = (1 + test_df['ret']).prod() - 1
    
    return {
        'train_sharpe': float(train_sharpe),
        'test_sharpe': float(test_sharpe),
        'test_ret': float(test_ret),
        'bh_ret': float(bh_ret),
        'beats_bh': test_ret > bh_ret,
        'threshold': float(threshold),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'validated': test_sharpe > 0,
    }

def strategy_correlation_based(si, data, train_end, cost):
    """
    Alternative: Scale position by SI z-score instead of binary threshold.
    No threshold optimization.
    """
    returns = data['close'].pct_change().shift(-2)
    common = si.index.intersection(returns.index)
    df = pd.DataFrame({'si': si.loc[common], 'ret': returns.loc[common]}).dropna()
    
    if len(df) < 100:
        return {'validated': False, 'reason': 'insufficient data'}
    
    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end:]
    
    if len(train_df) < 50 or len(test_df) < 50:
        return {'validated': False, 'reason': 'insufficient data'}
    
    # Use z-score of SI as position size
    train_mean = train_df['si'].mean()
    train_std = train_df['si'].std()
    
    # Train
    z_train = (train_df['si'] - train_mean) / train_std
    pos_train = z_train.clip(-1, 1).shift(1).fillna(0)  # Position between -1 and 1
    net_train = pos_train * train_df['ret'] - pos_train.diff().abs().fillna(0) * cost
    train_sharpe = net_train.mean() / net_train.std() * np.sqrt(252) if net_train.std() > 0 else 0
    
    # Test (use train stats for normalization)
    z_test = (test_df['si'] - train_mean) / train_std
    pos_test = z_test.clip(-1, 1).shift(1).fillna(0)
    net_test = pos_test * test_df['ret'] - pos_test.diff().abs().fillna(0) * cost
    test_sharpe = net_test.mean() / net_test.std() * np.sqrt(252) if net_test.std() > 0 else 0
    test_ret = (1 + net_test).prod() - 1
    bh_ret = (1 + test_df['ret']).prod() - 1
    
    return {
        'train_sharpe': float(train_sharpe),
        'test_sharpe': float(test_sharpe),
        'test_ret': float(test_ret),
        'bh_ret': float(bh_ret),
        'beats_bh': test_ret > bh_ret,
        'validated': test_sharpe > 0,
    }

def main():
    print("\n" + "="*70)
    print("OVERFITTING FIX: Testing Simpler Strategies")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    loader = DataLoaderV2()
    results = []
    
    for mtype, symbols in MARKETS.items():
        print(f"\n  [{mtype.value.upper()}]")
        for symbol in symbols:
            print(f"    {symbol}...")
            try:
                data = loader.load(symbol, mtype)
                if len(data) < 200:
                    continue
                
                cost = TRANSACTION_COSTS.get(mtype, 0.001)
                si = compute_si_full(data)
                
                if len(si) < 100:
                    continue
                
                n = len(si)
                train_end = int(n * TRAIN_RATIO)
                
                # Test three strategies
                strat_opt = strategy_no_optimization(si, data, train_end, cost)
                strat_zscore = strategy_correlation_based(si, data, train_end, cost)
                
                results.append({
                    'asset': symbol,
                    'market': mtype.value,
                    'no_opt': strat_opt,
                    'zscore': strat_zscore,
                })
                
                print(f"      No-opt: train={strat_opt.get('train_sharpe',0):.2f}, test={strat_opt.get('test_sharpe',0):.2f}")
                print(f"      Z-score: train={strat_zscore.get('train_sharpe',0):.2f}, test={strat_zscore.get('test_sharpe',0):.2f}")
                
            except Exception as e:
                print(f"      Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print("\n  NO OPTIMIZATION (median threshold):")
    print("  " + "-"*60)
    print(f"  {'Asset':<10} {'Train':>10} {'Test':>10} {'Status':>15}")
    validated_no_opt = 0
    for r in results:
        s = r['no_opt']
        status = "✅ Valid" if s.get('validated') else "❌ Fail"
        if s.get('validated'):
            validated_no_opt += 1
        print(f"  {r['asset']:<10} {s.get('train_sharpe',0):>10.2f} {s.get('test_sharpe',0):>10.2f} {status:>15}")
    print(f"\n  Validated: {validated_no_opt}/{len(results)}")
    
    print("\n  Z-SCORE SCALING:")
    print("  " + "-"*60)
    print(f"  {'Asset':<10} {'Train':>10} {'Test':>10} {'Status':>15}")
    validated_zscore = 0
    for r in results:
        s = r['zscore']
        status = "✅ Valid" if s.get('validated') else "❌ Fail"
        if s.get('validated'):
            validated_zscore += 1
        print(f"  {r['asset']:<10} {s.get('train_sharpe',0):>10.2f} {s.get('test_sharpe',0):>10.2f} {status:>15}")
    print(f"\n  Validated: {validated_zscore}/{len(results)}")
    
    # Save results
    out_path = Path("results/overfitting_fix/comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'no_opt_validated': validated_no_opt,
                'zscore_validated': validated_zscore,
                'total': len(results)
            }
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
