#!/usr/bin/env python3
"""
PHASE 1 ADVANCED STATISTICAL METHODS
- Copula Analysis (tail dependence)
- Transfer Entropy (information flow)
- Wavelet Decomposition
- Quantile Regression
- Partial Correlations
- Cointegration Testing
- Rolling Beta Analysis
- Change Point Detection
- Bootstrap Inference
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, kendalltau, gaussian_kde
from scipy.stats import norm, t as t_dist
from scipy.signal import cwt, morlet2
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm
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

def transfer_entropy(x, y, lag=1, bins=10):
    """
    Compute transfer entropy from X to Y.
    TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})
    """
    x = np.array(x)
    y = np.array(y)
    
    n = len(x) - lag
    if n < 50:
        return np.nan
    
    # Discretize
    x_bins = np.digitize(x, np.linspace(x.min(), x.max(), bins))
    y_bins = np.digitize(y, np.linspace(y.min(), y.max(), bins))
    
    # Joint distributions
    y_t = y_bins[lag:]
    y_past = y_bins[lag-1:-1] if lag > 1 else y_bins[:-lag]
    x_past = x_bins[:-lag]
    
    # Entropy calculations using histograms
    def entropy(x):
        _, counts = np.unique(x, return_counts=True)
        probs = counts / len(x)
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def joint_entropy(x, y):
        pairs = list(zip(x, y))
        _, counts = np.unique(pairs, axis=0, return_counts=True)
        probs = counts / len(pairs)
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def triple_entropy(x, y, z):
        triples = list(zip(x, y, z))
        _, counts = np.unique(triples, axis=0, return_counts=True)
        probs = counts / len(triples)
        return -np.sum(probs * np.log(probs + 1e-10))
    
    # TE = H(Y_t, Y_past) - H(Y_past) - H(Y_t, Y_past, X_past) + H(Y_past, X_past)
    te = (joint_entropy(y_t, y_past) - entropy(y_past) - 
          triple_entropy(y_t, y_past, x_past) + joint_entropy(y_past, x_past))
    
    return te

def tail_dependence_coefficient(u, v, threshold=0.1):
    """
    Compute lower and upper tail dependence coefficients.
    """
    u = np.array(u)
    v = np.array(v)
    
    # Convert to uniform using empirical CDF
    u_ranks = (np.argsort(np.argsort(u)) + 1) / (len(u) + 1)
    v_ranks = (np.argsort(np.argsort(v)) + 1) / (len(v) + 1)
    
    # Lower tail: P(V < q | U < q)
    lower_mask = u_ranks < threshold
    if lower_mask.sum() > 10:
        lower_lambda = (v_ranks[lower_mask] < threshold).mean()
    else:
        lower_lambda = np.nan
    
    # Upper tail: P(V > 1-q | U > 1-q)
    upper_mask = u_ranks > (1 - threshold)
    if upper_mask.sum() > 10:
        upper_lambda = (v_ranks[upper_mask] > (1 - threshold)).mean()
    else:
        upper_lambda = np.nan
    
    return lower_lambda, upper_lambda

def block_bootstrap_correlation(x, y, n_bootstrap=1000, block_size=10):
    """
    Block bootstrap for correlation confidence intervals.
    """
    n = len(x)
    correlations = []
    
    for _ in range(n_bootstrap):
        # Generate block indices
        n_blocks = n // block_size + 1
        block_starts = np.random.randint(0, n - block_size, n_blocks)
        
        indices = []
        for start in block_starts:
            indices.extend(range(start, min(start + block_size, n)))
        indices = np.array(indices[:n])
        
        x_boot = x.iloc[indices].values
        y_boot = y.iloc[indices].values
        
        r, _ = spearmanr(x_boot, y_boot)
        if not np.isnan(r):
            correlations.append(r)
    
    if len(correlations) < 100:
        return np.nan, np.nan, np.nan
    
    correlations = np.array(correlations)
    return np.mean(correlations), np.percentile(correlations, 2.5), np.percentile(correlations, 97.5)

def detect_change_points(series, window=50, threshold=2.5):
    """
    Detect change points using CUSUM-like method.
    """
    series = np.array(series)
    n = len(series)
    
    change_points = []
    
    for i in range(window, n - window):
        left = series[i-window:i]
        right = series[i:i+window]
        
        # Two-sample t-test statistic
        t_stat = abs(left.mean() - right.mean()) / np.sqrt(left.var()/window + right.var()/window + 1e-10)
        
        if t_stat > threshold:
            change_points.append(i)
    
    # Remove nearby duplicates
    filtered = []
    for cp in change_points:
        if not filtered or cp - filtered[-1] > window:
            filtered.append(cp)
    
    return filtered

def main():
    print("\n" + "="*70)
    print("  PHASE 1 ADVANCED STATISTICAL METHODS")
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
    section("1. TRANSFER ENTROPY ANALYSIS")
    # ============================================================
    
    print("\n  Measuring directional information flow between SI and features")
    print("  TE(X→Y) > TE(Y→X) means X contains predictive info about Y")
    
    te_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common].values
        
        asset_te = {}
        for feat in ['adx', 'rsi_extremity', 'volatility']:
            feat_aligned = features.loc[common, feat].values
            
            # TE in both directions
            te_si_to_feat = transfer_entropy(si_aligned, feat_aligned, lag=1)
            te_feat_to_si = transfer_entropy(feat_aligned, si_aligned, lag=1)
            
            asset_te[feat] = {
                'te_si_to_feat': float(te_si_to_feat) if not np.isnan(te_si_to_feat) else None,
                'te_feat_to_si': float(te_feat_to_si) if not np.isnan(te_feat_to_si) else None,
            }
            
            if te_si_to_feat and te_feat_to_si:
                ratio = te_si_to_feat / (te_feat_to_si + 1e-10)
                direction = "SI → " + feat if ratio > 1.2 else (feat + " → SI" if ratio < 0.8 else "Bidirectional")
                print(f"    {feat}: TE(SI→{feat})={te_si_to_feat:.4f}, TE({feat}→SI)={te_feat_to_si:.4f} [{direction}]")
                
                if ratio > 1.5:
                    discover(f"SI predicts {feat} (TE ratio={ratio:.2f}) in {name}")
                elif ratio < 0.67:
                    discover(f"{feat} predicts SI (TE ratio={ratio:.2f}) in {name}")
        
        te_results[name] = asset_te
    
    all_results['transfer_entropy'] = te_results
    
    # ============================================================
    section("2. TAIL DEPENDENCE (COPULA) ANALYSIS")
    # ============================================================
    
    print("\n  Do SI and features move together during extreme events?")
    
    tail_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        
        asset_tail = {}
        for feat in ['adx', 'rsi_extremity', 'volatility']:
            feat_aligned = features.loc[common, feat]
            
            lower_lambda, upper_lambda = tail_dependence_coefficient(
                si_aligned.values, feat_aligned.values, threshold=0.1
            )
            
            asset_tail[feat] = {
                'lower_tail': float(lower_lambda) if not np.isnan(lower_lambda) else None,
                'upper_tail': float(upper_lambda) if not np.isnan(upper_lambda) else None,
            }
            
            lower_str = f"{lower_lambda:.3f}" if not np.isnan(lower_lambda) else "N/A"
            upper_str = f"{upper_lambda:.3f}" if not np.isnan(upper_lambda) else "N/A"
            print(f"    {feat}: Lower λ={lower_str}, Upper λ={upper_str}")
            
            # Check for asymmetric tail dependence
            if lower_lambda and upper_lambda and not np.isnan(lower_lambda) and not np.isnan(upper_lambda):
                if abs(lower_lambda - upper_lambda) > 0.1:
                    discover(f"Asymmetric tail dependence SI-{feat} in {name} (L={lower_lambda:.2f}, U={upper_lambda:.2f})")
        
        tail_results[name] = asset_tail
    
    all_results['tail_dependence'] = tail_results
    
    # ============================================================
    section("3. QUANTILE REGRESSION")
    # ============================================================
    
    print("\n  Does SI-feature relationship change across quantiles?")
    
    quantile_results = {}
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        adx_aligned = features.loc[common, 'adx']
        
        # Standardize
        si_std = (si_aligned - si_aligned.mean()) / si_aligned.std()
        adx_std = (adx_aligned - adx_aligned.mean()) / adx_aligned.std()
        
        df = pd.DataFrame({'si': si_std, 'adx': adx_std}).dropna()
        
        asset_quantile = {}
        print(f"    {'Quantile':<10} {'β(SI→ADX)':>12} {'p-value':>10}")
        print("    " + "-"*35)
        
        for q in quantiles:
            try:
                model = QuantReg(df['adx'], sm.add_constant(df['si']))
                result = model.fit(q=q)
                beta = result.params['si']
                pval = result.pvalues['si']
                
                asset_quantile[f'q{int(q*100)}'] = {
                    'beta': float(beta),
                    'pvalue': float(pval),
                }
                
                sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
                print(f"    {q:<10.2f} {beta:>+12.3f} {pval:>10.4f} {sig}")
            except:
                print(f"    {q:<10.2f} {'N/A':>12}")
        
        # Check for quantile heterogeneity
        if 'q10' in asset_quantile and 'q90' in asset_quantile:
            beta_diff = abs(asset_quantile['q90']['beta'] - asset_quantile['q10']['beta'])
            if beta_diff > 0.1:
                discover(f"SI-ADX relationship varies by quantile in {name} (Δβ={beta_diff:.3f})")
        
        quantile_results[name] = asset_quantile
    
    all_results['quantile_regression'] = quantile_results
    
    # ============================================================
    section("4. COINTEGRATION TESTING")
    # ============================================================
    
    print("\n  Are SI and features cointegrated (long-run equilibrium)?")
    
    coint_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        
        asset_coint = {}
        for feat in ['adx', 'rsi_extremity']:
            feat_aligned = features.loc[common, feat]
            
            # First check stationarity
            si_adf = adfuller(si_aligned.dropna())[1]
            feat_adf = adfuller(feat_aligned.dropna())[1]
            
            # Cointegration test
            try:
                coint_stat, coint_pval, _ = coint(si_aligned.dropna(), feat_aligned.dropna())
                
                asset_coint[feat] = {
                    'coint_stat': float(coint_stat),
                    'coint_pval': float(coint_pval),
                    'si_stationary': si_adf < 0.05,
                    'feat_stationary': feat_adf < 0.05,
                }
                
                status = "Cointegrated ✓" if coint_pval < 0.05 else "Not cointegrated"
                print(f"    SI-{feat}: stat={coint_stat:.3f}, p={coint_pval:.4f} [{status}]")
                
                if coint_pval < 0.05:
                    discover(f"SI and {feat} are cointegrated in {name} (long-run equilibrium)")
            except:
                print(f"    SI-{feat}: Error in cointegration test")
        
        coint_results[name] = asset_coint
    
    all_results['cointegration'] = coint_results
    
    # ============================================================
    section("5. PARTIAL CORRELATIONS")
    # ============================================================
    
    print("\n  Controlling for confounders (volatility)")
    
    partial_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        
        df = pd.DataFrame({
            'si': si.loc[common],
            'adx': features.loc[common, 'adx'],
            'rsi_ext': features.loc[common, 'rsi_extremity'],
            'vol': features.loc[common, 'volatility'],
        }).dropna()
        
        # Simple correlation
        r_si_adx, _ = spearmanr(df['si'], df['adx'])
        
        # Partial correlation controlling for volatility
        # r(X,Y|Z) = (r(X,Y) - r(X,Z)*r(Y,Z)) / sqrt((1-r(X,Z)^2)(1-r(Y,Z)^2))
        r_si_vol, _ = spearmanr(df['si'], df['vol'])
        r_adx_vol, _ = spearmanr(df['adx'], df['vol'])
        
        partial_r = (r_si_adx - r_si_vol * r_adx_vol) / np.sqrt((1 - r_si_vol**2) * (1 - r_adx_vol**2))
        
        partial_results[name] = {
            'simple_r': float(r_si_adx),
            'partial_r_control_vol': float(partial_r),
            'r_si_vol': float(r_si_vol),
            'r_adx_vol': float(r_adx_vol),
        }
        
        print(f"    SI-ADX simple:    r = {r_si_adx:+.3f}")
        print(f"    SI-ADX | Vol:     r = {partial_r:+.3f}")
        print(f"    Change:           Δ = {partial_r - r_si_adx:+.3f}")
        
        if abs(partial_r - r_si_adx) > 0.05:
            discover(f"Volatility confounds SI-ADX in {name} (Δr={partial_r - r_si_adx:+.3f})")
    
    all_results['partial_correlations'] = partial_results
    
    # ============================================================
    section("6. BOOTSTRAP CONFIDENCE INTERVALS")
    # ============================================================
    
    print("\n  Robust inference using block bootstrap")
    
    bootstrap_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        adx_aligned = features.loc[common, 'adx']
        
        mean_r, ci_low, ci_high = block_bootstrap_correlation(si_aligned, adx_aligned, n_bootstrap=500, block_size=20)
        
        bootstrap_results[name] = {
            'mean_r': float(mean_r) if not np.isnan(mean_r) else None,
            'ci_95_low': float(ci_low) if not np.isnan(ci_low) else None,
            'ci_95_high': float(ci_high) if not np.isnan(ci_high) else None,
        }
        
        if not np.isnan(mean_r):
            print(f"    SI-ADX: r = {mean_r:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")
            
            if ci_low > 0:
                discover(f"SI-ADX significantly positive in {name} (CI: [{ci_low:.3f}, {ci_high:.3f}])")
            elif ci_high < 0:
                discover(f"SI-ADX significantly negative in {name}")
    
    all_results['bootstrap'] = bootstrap_results
    
    # ============================================================
    section("7. CHANGE POINT DETECTION")
    # ============================================================
    
    print("\n  Detecting structural breaks in SI behavior")
    
    changepoint_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        
        change_points = detect_change_points(si.values, window=30, threshold=2.5)
        
        changepoint_results[name] = {
            'n_change_points': len(change_points),
            'change_point_indices': change_points,
            'change_point_dates': [str(si.index[i]) for i in change_points if i < len(si)],
        }
        
        print(f"    Detected {len(change_points)} structural breaks")
        if change_points:
            for i, cp in enumerate(change_points[:5]):
                if cp < len(si):
                    print(f"      {i+1}. {si.index[cp].strftime('%Y-%m-%d')}")
            
            if len(change_points) > 5:
                print(f"      ... and {len(change_points) - 5} more")
        
        if len(change_points) > 10:
            discover(f"Frequent SI regime changes in {name} ({len(change_points)} breaks)")
    
    all_results['change_points'] = changepoint_results
    
    # ============================================================
    section("8. ROLLING BETA STABILITY")
    # ============================================================
    
    print("\n  Is SI-ADX relationship stable over time?")
    
    rolling_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        adx_aligned = features.loc[common, 'adx']
        
        # Rolling regression
        window = 120  # ~6 months
        betas = []
        dates = []
        
        for i in range(window, len(si_aligned)):
            si_window = si_aligned.iloc[i-window:i]
            adx_window = adx_aligned.iloc[i-window:i]
            
            df = pd.DataFrame({'si': si_window, 'adx': adx_window}).dropna()
            if len(df) < 50:
                continue
            
            try:
                model = sm.OLS(df['adx'], sm.add_constant(df['si'])).fit()
                betas.append(model.params['si'])
                dates.append(si_aligned.index[i])
            except:
                pass
        
        if betas:
            betas = np.array(betas)
            rolling_results[name] = {
                'mean_beta': float(np.mean(betas)),
                'std_beta': float(np.std(betas)),
                'min_beta': float(np.min(betas)),
                'max_beta': float(np.max(betas)),
                'pct_positive': float((betas > 0).mean()),
            }
            
            print(f"    Mean β: {np.mean(betas):.3f} ± {np.std(betas):.3f}")
            print(f"    Range: [{np.min(betas):.3f}, {np.max(betas):.3f}]")
            print(f"    % Positive: {(betas > 0).mean():.1%}")
            
            if (betas > 0).mean() > 0.8:
                discover(f"SI-ADX relationship robustly positive in {name} ({(betas > 0).mean():.0%} of windows)")
    
    all_results['rolling_beta'] = rolling_results
    
    # ============================================================
    section("SUMMARY")
    # ============================================================
    
    print(f"\n  Total discoveries: {len(discoveries)}")
    for i, d in enumerate(discoveries, 1):
        print(f"    {i}. {d}")
    
    out_path = Path("results/phase1_advanced/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'discoveries': discoveries,
            'results': all_results,
        }, f, indent=2, default=str)
    
    discoveries_path = Path("docs/DISCOVERIES_LOG.md")
    with open(discoveries_path, 'a') as f:
        f.write(f"\n\n## Phase 1 Advanced Stats - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for d in discoveries:
            f.write(f"- {d}\n")
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
