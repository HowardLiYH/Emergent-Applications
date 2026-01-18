#!/usr/bin/env python3
"""
SI DEEP EXPLORATION - FULLY FIXED VERSION
Fixes ALL audit issues.
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from tqdm import tqdm
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

def compute_features(data):
    features = pd.DataFrame(index=data.index)
    returns = data['close'].pct_change()
    features['returns'] = returns
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_dm = (data['high'] - data['high'].shift()).clip(lower=0)
    minus_dm = (data['low'].shift() - data['low']).clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    features['adx'] = dx.rolling(14).mean()
    features['volatility'] = returns.rolling(14).std()
    delta = data['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))
    features['future_return_1d'] = returns.shift(-2)  # 1-day execution delay
    features['future_return_5d'] = returns.rolling(5).sum().shift(-6)
    return features

def compute_si_full(data, window=7):
    if len(data) < window * 3:
        return pd.Series(dtype=float)
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    try:
        population.run(data)
        return population.compute_si_timeseries(data, window=window)
    except Exception as e:
        print(f"    SI Error: {e}")
        return pd.Series(dtype=float)

def test_stationarity(series):
    clean = series.dropna()
    if len(clean) < 20:
        return {'stationary': None, 'p_value': None}
    try:
        result = adfuller(clean, autolag='AIC')
        return {'stationary': result[1] < 0.05, 'p_value': float(result[1])}
    except:
        return {'stationary': None, 'p_value': None}

def compute_effective_n(series):
    clean = series.dropna()
    n = len(clean)
    if n < 10:
        return n
    rho = clean.autocorr(lag=1)
    if pd.isna(rho) or rho <= 0:
        return n
    return max(int(n * (1 - rho) / (1 + rho)), 10)

def block_bootstrap_corr(x, y, n_boot=500):
    common = x.index.intersection(y.index)
    x, y = x.loc[common].dropna(), y.loc[common].dropna()
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    n = len(x)
    if n < 30:
        return {'r': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_adj': np.nan}
    rho = x.autocorr(lag=1)
    block_size = max(1, min(int(1 / (1 - rho + 0.01)), n // 4)) if rho and rho > 0 else 1
    r_orig, _ = spearmanr(x, y)
    boots = []
    n_blocks = (n + block_size - 1) // block_size
    for _ in range(n_boot):
        idx = []
        for bi in np.random.choice(n_blocks, n_blocks, replace=True):
            idx.extend(range(bi * block_size, min((bi + 1) * block_size, n)))
        idx = idx[:n]
        r, _ = spearmanr(x.iloc[idx].values, y.iloc[idx].values)
        if not np.isnan(r):
            boots.append(r)
    if len(boots) < 100:
        return {'r': float(r_orig), 'ci_low': np.nan, 'ci_high': np.nan, 'p_adj': np.nan}
    boots = np.array(boots)
    p_adj = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return {'r': float(r_orig), 'ci_low': float(np.percentile(boots, 2.5)), 
            'ci_high': float(np.percentile(boots, 97.5)), 'p_adj': float(p_adj), 'block': block_size}

def test_granger(si, feat, max_lag=5):
    common = si.index.intersection(feat.index)
    si, feat = si.loc[common].dropna(), feat.loc[common].dropna()
    common = si.index.intersection(feat.index)
    if len(common) < 50:
        return {'causes': None, 'lag': None, 'p': None}
    df = pd.DataFrame({'si': si.loc[common], 'f': feat.loc[common]}).dropna()
    if len(df) < 50:
        return {'causes': None, 'lag': None, 'p': None}
    try:
        res = grangercausalitytests(df[['f', 'si']], maxlag=max_lag, verbose=False)
        best_p, best_lag = 1.0, None
        for lag in range(1, max_lag + 1):
            p = res[lag][0]['ssr_ftest'][1]
            if p < best_p:
                best_p, best_lag = p, lag
        return {'causes': best_p < 0.05, 'lag': best_lag, 'p': float(best_p)}
    except:
        return {'causes': None, 'lag': None, 'p': None}

def analyze_strategy(si, data, train_end, cost):
    returns = data['close'].pct_change().shift(-2)
    common = si.index.intersection(returns.index)
    df = pd.DataFrame({'si': si.loc[common], 'ret': returns.loc[common]}).dropna()
    if len(df) < 100:
        return {'validated': False, 'reason': 'insufficient'}
    train_df, test_df = df.iloc[:train_end], df.iloc[train_end:]
    if len(train_df) < 50 or len(test_df) < 50:
        return {'validated': False, 'reason': f'train={len(train_df)},test={len(test_df)}'}
    best_thresh, best_sharpe = None, -np.inf
    for thresh in np.percentile(train_df['si'], [20, 40, 50, 60, 80]):
        sig = (train_df['si'] >= thresh).astype(int)
        pos = sig.shift(1).fillna(0)
        net = pos * train_df['ret'] - pos.diff().abs().fillna(0) * cost
        sharpe = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else 0
        if sharpe > best_sharpe:
            best_sharpe, best_thresh = sharpe, thresh
    if best_thresh is None:
        return {'validated': False, 'reason': 'no threshold'}
    sig = (test_df['si'] >= best_thresh).astype(int)
    pos = sig.shift(1).fillna(0)
    net = pos * test_df['ret'] - pos.diff().abs().fillna(0) * cost
    test_sharpe = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else 0
    test_ret = (1 + net).prod() - 1
    bh_ret = (1 + test_df['ret']).prod() - 1
    return {'train_sharpe': float(best_sharpe), 'test_sharpe': float(test_sharpe),
            'test_ret': float(test_ret), 'bh_ret': float(bh_ret), 'beats_bh': test_ret > bh_ret,
            'thresh': float(best_thresh), 'n_train': len(train_df), 'n_test': len(test_df),
            'validated': test_sharpe > 0}

def analyze_asset(symbol, mtype, data):
    result = {'asset': symbol, 'market': mtype.value, 'n': len(data)}
    cost = TRANSACTION_COSTS.get(mtype, 0.001)
    print(f"      SI on {len(data)} bars...")
    si = compute_si_full(data)
    if len(si) < 100:
        result['status'] = 'si_failed'
        return result
    result['n_si'] = len(si)
    features = compute_features(data)
    common = si.index.intersection(features.index)
    si, features = si.loc[common], features.loc[common]
    n = len(si)
    train_end = int(n * TRAIN_RATIO)
    train_idx, test_idx = si.index[:train_end], si.index[train_end:]
    result['n_train'], result['n_test'] = len(train_idx), len(test_idx)
    print(f"      Train={result['n_train']}, Test={result['n_test']}")
    result['stationarity'] = {'si': test_stationarity(si), 'si_diff': test_stationarity(si.diff())}
    result['eff_n'] = compute_effective_n(si)
    result['corr'] = {}
    si_tr, feat_tr = si.loc[train_idx], features.loc[train_idx]
    for f in ['adx', 'volatility', 'rsi']:
        if f not in feat_tr.columns:
            continue
        result['corr'][f] = {
            'level': block_bootstrap_corr(si_tr, feat_tr[f]),
            'diff': block_bootstrap_corr(si_tr.diff(), feat_tr[f].diff()),
            'granger': test_granger(si_tr, feat_tr[f])
        }
    result['pred'] = {}
    for h in ['future_return_1d', 'future_return_5d']:
        if h not in feat_tr.columns:
            continue
        tr_corr = block_bootstrap_corr(si_tr, feat_tr[h])
        te_corr = block_bootstrap_corr(si.loc[test_idx], features.loc[test_idx][h])
        result['pred'][h] = {
            'train': tr_corr, 'test': te_corr,
            'train_sig': tr_corr.get('p_adj', 1) < 0.05,
            'test_confirms': (te_corr.get('r', 0) * tr_corr.get('r', 0) > 0 and te_corr.get('p_adj', 1) < 0.10)
        }
    result['strat'] = analyze_strategy(si, data, train_end, cost)
    result['status'] = 'ok'
    return result

def main():
    print("\n" + "="*70)
    print("SI EXPLORATION - FIXED VERSION")
    print("="*70)
    print(f"Time: {datetime.now().isoformat()}")
    print("Fixes: Full SI->split, Bootstrap CIs, Granger, ADF, Exec delay")
    print("="*70)
    loader = DataLoaderV2()
    results = []
    for mtype, syms in MARKETS.items():
        print(f"\n  [{mtype.value.upper()}]")
        for sym in syms:
            print(f"    {sym}...")
            try:
                data = loader.load(sym, mtype)
                if len(data) < 200:
                    print(f"      Skip: {len(data)} bars")
                    continue
                r = analyze_asset(sym, mtype, data)
                results.append(r)
                print(f"      Done (eff_n={r.get('eff_n','?')})")
            except Exception as e:
                print(f"      Err: {e}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n  Stationarity: {sum(1 for r in results if r.get('stationarity',{}).get('si',{}).get('stationary'))}/{len(results)}")
    for f in ['adx', 'volatility', 'rsi']:
        rs = [r['corr'].get(f,{}).get('level',{}).get('r',np.nan) for r in results if 'corr' in r]
        rs = [x for x in rs if not np.isnan(x)]
        if rs:
            print(f"  SI-{f}: mean r={np.mean(rs):.3f}, all_same_sign={all(x>0 for x in rs) or all(x<0 for x in rs)}")
    for f in ['adx', 'volatility', 'rsi']:
        rs = [r['corr'].get(f,{}).get('diff',{}).get('r',np.nan) for r in results if 'corr' in r]
        rs = [x for x in rs if not np.isnan(x)]
        if rs:
            print(f"  dSI-d{f}: mean r={np.mean(rs):.3f}")
    for f in ['adx', 'volatility', 'rsi']:
        gc = [r['corr'].get(f,{}).get('granger',{}).get('causes') for r in results if 'corr' in r]
        gc = [x for x in gc if x is not None]
        if gc:
            print(f"  Granger SI->{f}: {sum(gc)}/{len(gc)} ({100*sum(gc)/len(gc):.0f}%)")
    confirmed = sum(1 for r in results if r.get('pred',{}).get('future_return_1d',{}).get('test_confirms'))
    train_sig = sum(1 for r in results if r.get('pred',{}).get('future_return_1d',{}).get('train_sig'))
    print(f"  Prediction: train_sig={train_sig}, test_confirms={confirmed}")
    validated = [r for r in results if r.get('strat',{}).get('validated')]
    beats = [r for r in results if r.get('strat',{}).get('beats_bh')]
    print(f"  Strategy: validated={len(validated)}/{len(results)}, beats_bh={len(beats)}/{len(results)}")
    if validated:
        print(f"  Avg OOS Sharpe: {np.mean([r['strat']['test_sharpe'] for r in validated]):.2f}")
    
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    level_ok = all(all(x>0 for x in [r['corr'].get(f,{}).get('level',{}).get('r',0) for r in results if 'corr' in r]) or 
                   all(x<0 for x in [r['corr'].get(f,{}).get('level',{}).get('r',0) for r in results if 'corr' in r]) 
                   for f in ['adx','volatility'])
    diff_ok = any(r['corr'].get(f,{}).get('diff',{}).get('p_adj',1) < 0.05 
                  for r in results for f in ['adx','volatility'] if 'corr' in r)
    granger_ok = any(sum(r['corr'].get(f,{}).get('granger',{}).get('causes',False) for r in results if 'corr' in r) > len(results)*0.3 
                     for f in ['adx','volatility','rsi'])
    pred_ok = confirmed > 0
    strat_ok = len(validated) > len(results) * 0.3
    score = sum([level_ok, diff_ok, granger_ok, pred_ok, strat_ok])
    print(f"  Level consistent: {level_ok}")
    print(f"  Diff significant: {diff_ok}")
    print(f"  Granger found: {granger_ok}")
    print(f"  Pred validated: {pred_ok}")
    print(f"  Strategy works: {strat_ok}")
    verdict = ["NONE","NONE","WEAK","MODERATE","STRONG","STRONG"][score]
    print(f"\n  VERDICT: {verdict} (Score: {score}/5)")
    
    out = Path('results/exploration_fixed/results.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'time': datetime.now().isoformat(), 'n': len(results), 
                   'score': score, 'verdict': verdict, 'results': results}, f, indent=2, default=str)
    print(f"\n  Saved: {out}")
    print("="*70 + "\n")
    return results

if __name__ == "__main__":
    main()
