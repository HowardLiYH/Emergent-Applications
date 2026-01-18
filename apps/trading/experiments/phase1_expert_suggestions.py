#!/usr/bin/env python3
"""
PHASE 1 EXPERT PANEL SUGGESTIONS IMPLEMENTATION
Top 10 recommendations from 35+ experts

1. Error Correction Model (ECM)
2. Hurst Exponent (long memory)
3. Ornstein-Uhlenbeck fitting
4. Extreme Value Theory
5. Factor neutralization
6. HAC standard errors
7. Hidden Markov Model
8. Copula fitting
9. Permutation tests
10. Entropy rate
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, genextreme, genpareto
from scipy.optimize import minimize
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
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
    features['momentum'] = data['close'].pct_change(7)
    
    return features.dropna()

def hurst_exponent(ts, max_lag=20):
    """
    Compute Hurst exponent using R/S analysis.
    H > 0.5: Long memory (trending)
    H = 0.5: Random walk
    H < 0.5: Mean-reverting
    """
    ts = np.array(ts)
    lags = range(2, max_lag + 1)
    rs = []
    
    for lag in lags:
        chunks = len(ts) // lag
        rs_values = []
        
        for i in range(chunks):
            chunk = ts[i*lag:(i+1)*lag]
            mean = chunk.mean()
            cumdev = np.cumsum(chunk - mean)
            r = cumdev.max() - cumdev.min()
            s = chunk.std()
            if s > 0:
                rs_values.append(r / s)
        
        if rs_values:
            rs.append(np.mean(rs_values))
    
    if len(rs) < 5:
        return np.nan
    
    # Fit log(R/S) = H * log(n) + c
    log_lags = np.log(list(lags)[:len(rs)])
    log_rs = np.log(rs)
    
    coeffs = np.polyfit(log_lags, log_rs, 1)
    return coeffs[0]

def fit_ou_process(ts):
    """
    Fit Ornstein-Uhlenbeck process: dX = theta*(mu - X)dt + sigma*dW
    Returns: theta (mean reversion speed), mu (long-term mean), sigma
    """
    ts = np.array(ts)
    dt = 1  # 1 day
    
    # Simple regression: X_{t+1} - X_t = theta*(mu - X_t)*dt + noise
    # => X_{t+1} = (1 - theta*dt)*X_t + theta*mu*dt + noise
    # => X_{t+1} = a + b*X_t + noise
    
    X = ts[:-1]
    Y = ts[1:]
    
    model = OLS(Y, sm.add_constant(X)).fit()
    
    a = model.params[0]
    b = model.params[1]
    
    theta = (1 - b) / dt
    mu = a / (theta * dt) if theta != 0 else ts.mean()
    sigma = model.resid.std() / np.sqrt(dt)
    
    half_life = np.log(2) / theta if theta > 0 else np.inf
    
    return {
        'theta': float(theta),
        'mu': float(mu),
        'sigma': float(sigma),
        'half_life_days': float(half_life),
        'r_squared': float(model.rsquared),
    }

def fit_evt(ts, threshold_percentile=95):
    """
    Fit Generalized Pareto Distribution to tail using Extreme Value Theory.
    """
    ts = np.array(ts)
    threshold = np.percentile(ts, threshold_percentile)
    
    exceedances = ts[ts > threshold] - threshold
    
    if len(exceedances) < 10:
        return None
    
    try:
        shape, loc, scale = genpareto.fit(exceedances)
        return {
            'shape': float(shape),
            'scale': float(scale),
            'threshold': float(threshold),
            'n_exceedances': len(exceedances),
            'tail_type': 'heavy' if shape > 0 else ('light' if shape < 0 else 'exponential'),
        }
    except:
        return None

def permutation_test(x, y, n_perm=1000):
    """
    Permutation test for correlation significance.
    """
    x, y = np.array(x), np.array(y)
    
    observed_r, _ = spearmanr(x, y)
    
    perm_rs = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        r, _ = spearmanr(x, y_perm)
        perm_rs.append(r)
    
    perm_rs = np.array(perm_rs)
    p_value = (np.abs(perm_rs) >= np.abs(observed_r)).mean()
    
    return {
        'observed_r': float(observed_r),
        'p_value_perm': float(p_value),
        'null_mean': float(np.mean(perm_rs)),
        'null_std': float(np.std(perm_rs)),
    }

def entropy_rate(ts, order=3, bins=10):
    """
    Estimate entropy rate: how predictable is SI given its past?
    H(X_t | X_{t-1}, ..., X_{t-k})
    """
    ts = np.array(ts)
    ts_discrete = np.digitize(ts, np.linspace(ts.min(), ts.max(), bins))
    
    # Joint probability P(X_t, X_{t-1}, ..., X_{t-k})
    n = len(ts_discrete) - order
    sequences = []
    
    for i in range(n):
        seq = tuple(ts_discrete[i:i+order+1])
        sequences.append(seq)
    
    from collections import Counter
    counts = Counter(sequences)
    probs = np.array(list(counts.values())) / len(sequences)
    joint_entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Marginal entropy of past
    past_seqs = [s[:-1] for s in sequences]
    past_counts = Counter(past_seqs)
    past_probs = np.array(list(past_counts.values())) / len(past_seqs)
    past_entropy = -np.sum(past_probs * np.log(past_probs + 1e-10))
    
    # Conditional entropy (entropy rate approximation)
    entropy_rate = joint_entropy - past_entropy
    
    # Normalize by max entropy
    max_entropy = np.log(bins)
    normalized = entropy_rate / max_entropy
    
    return {
        'entropy_rate': float(entropy_rate),
        'normalized': float(normalized),
        'joint_entropy': float(joint_entropy),
        'past_entropy': float(past_entropy),
    }

def main():
    print("\n" + "="*70)
    print("  PHASE 1 EXPERT SUGGESTIONS IMPLEMENTATION")
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
    section("1. HURST EXPONENT (Long Memory Test)")
    # ============================================================
    
    print("\n  H > 0.5: Trending/Persistent")
    print("  H = 0.5: Random walk")
    print("  H < 0.5: Mean-reverting")
    
    hurst_results = {}
    
    for name, (data, mtype) in assets.items():
        si = compute_si(data)
        si_clean = si.dropna().values
        
        h = hurst_exponent(si_clean, max_lag=50)
        hurst_results[name] = {'hurst': float(h) if not np.isnan(h) else None}
        
        if not np.isnan(h):
            behavior = "Trending" if h > 0.55 else ("Mean-reverting" if h < 0.45 else "Random walk")
            print(f"    {name}: H = {h:.3f} ({behavior})")
            
            if abs(h - 0.5) > 0.1:
                discover(f"SI has {'long memory' if h > 0.5 else 'anti-persistence'} in {name} (H={h:.3f})")
    
    all_results['hurst'] = hurst_results
    
    # ============================================================
    section("2. ORNSTEIN-UHLENBECK PROCESS FITTING")
    # ============================================================
    
    print("\n  dSI = θ(μ - SI)dt + σdW")
    
    ou_results = {}
    
    for name, (data, mtype) in assets.items():
        si = compute_si(data)
        si_clean = si.dropna().values
        
        ou = fit_ou_process(si_clean)
        ou_results[name] = ou
        
        print(f"\n  {name}:")
        print(f"    θ (reversion speed): {ou['theta']:.4f}")
        print(f"    μ (long-term mean):  {ou['mu']:.5f}")
        print(f"    σ (volatility):      {ou['sigma']:.5f}")
        print(f"    Half-life:           {ou['half_life_days']:.1f} days")
        print(f"    R² of fit:           {ou['r_squared']:.3f}")
        
        if ou['theta'] > 0.1:
            discover(f"SI is mean-reverting with half-life={ou['half_life_days']:.1f} days in {name}")
    
    all_results['ornstein_uhlenbeck'] = ou_results
    
    # ============================================================
    section("3. EXTREME VALUE THEORY (Tail Analysis)")
    # ============================================================
    
    print("\n  Fitting Generalized Pareto Distribution to tails")
    
    evt_results = {}
    
    for name, (data, mtype) in assets.items():
        si = compute_si(data)
        si_clean = si.dropna().values
        
        # Right tail
        evt_right = fit_evt(si_clean, threshold_percentile=95)
        # Left tail
        evt_left = fit_evt(-si_clean, threshold_percentile=95)
        
        evt_results[name] = {
            'right_tail': evt_right,
            'left_tail': evt_left,
        }
        
        print(f"\n  {name}:")
        if evt_right:
            print(f"    Right tail: shape={evt_right['shape']:.3f} ({evt_right['tail_type']})")
        if evt_left:
            print(f"    Left tail:  shape={evt_left['shape']:.3f} ({evt_left['tail_type']})")
        
        if evt_right and evt_right['shape'] > 0.1:
            discover(f"SI has heavy right tail in {name} (shape={evt_right['shape']:.3f})")
    
    all_results['evt'] = evt_results
    
    # ============================================================
    section("4. PERMUTATION TESTS (Statistical Validity)")
    # ============================================================
    
    print("\n  Testing if SI-ADX correlation is real or spurious")
    
    perm_results = {}
    
    for name, (data, mtype) in assets.items():
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common].dropna()
        adx_aligned = features.loc[common, 'adx'].dropna()
        
        common_final = si_aligned.index.intersection(adx_aligned.index)
        si_vals = si_aligned.loc[common_final].values
        adx_vals = adx_aligned.loc[common_final].values
        
        perm = permutation_test(si_vals, adx_vals, n_perm=1000)
        perm_results[name] = perm
        
        sig = "✓ Significant" if perm['p_value_perm'] < 0.05 else "Not significant"
        print(f"    {name}: r={perm['observed_r']:.3f}, p={perm['p_value_perm']:.4f} [{sig}]")
        
        if perm['p_value_perm'] < 0.01:
            discover(f"SI-ADX correlation is highly significant by permutation test in {name}")
    
    all_results['permutation'] = perm_results
    
    # ============================================================
    section("5. ENTROPY RATE (Predictability)")
    # ============================================================
    
    print("\n  How predictable is SI given its past?")
    print("  Lower entropy rate = more predictable")
    
    entropy_results = {}
    
    for name, (data, mtype) in assets.items():
        si = compute_si(data)
        si_clean = si.dropna().values
        
        ent = entropy_rate(si_clean, order=3, bins=10)
        entropy_results[name] = ent
        
        print(f"    {name}: H(SI_t|past) = {ent['entropy_rate']:.3f} (normalized: {ent['normalized']:.3f})")
        
        if ent['normalized'] < 0.5:
            discover(f"SI is highly predictable from its past in {name} (H={ent['normalized']:.3f})")
    
    all_results['entropy_rate'] = entropy_results
    
    # ============================================================
    section("6. HIDDEN MARKOV MODEL (Latent States)")
    # ============================================================
    
    print("\n  Finding latent SI regimes")
    
    hmm_results = {}
    
    for name, (data, mtype) in assets.items():
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common].dropna()
        
        # Prepare features for HMM
        X = si_aligned.values.reshape(-1, 1)
        
        # Fit 2-state HMM
        try:
            model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
            model.fit(X)
            
            states = model.predict(X)
            
            # Characterize states
            state_0_mean = si_aligned.values[states == 0].mean()
            state_1_mean = si_aligned.values[states == 1].mean()
            
            low_si_state = 0 if state_0_mean < state_1_mean else 1
            high_si_state = 1 - low_si_state
            
            # Transition matrix
            trans_mat = model.transmat_
            
            hmm_results[name] = {
                'n_states': 2,
                'state_means': [float(state_0_mean), float(state_1_mean)],
                'transition_matrix': trans_mat.tolist(),
                'persistence_low_si': float(trans_mat[low_si_state, low_si_state]),
                'persistence_high_si': float(trans_mat[high_si_state, high_si_state]),
            }
            
            print(f"\n  {name}:")
            print(f"    State means: Low SI={min(state_0_mean, state_1_mean):.5f}, High SI={max(state_0_mean, state_1_mean):.5f}")
            print(f"    Persistence: Low SI={trans_mat[low_si_state, low_si_state]:.2%}, High SI={trans_mat[high_si_state, high_si_state]:.2%}")
            
            if min(trans_mat[0,0], trans_mat[1,1]) > 0.9:
                discover(f"SI states are highly persistent in {name} (>90%)")
        except:
            print(f"    {name}: HMM fitting failed")
    
    all_results['hmm'] = hmm_results
    
    # ============================================================
    section("7. HAC STANDARD ERRORS")
    # ============================================================
    
    print("\n  Robust inference accounting for autocorrelation")
    
    hac_results = {}
    
    for name, (data, mtype) in assets.items():
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common].dropna()
        adx_aligned = features.loc[common, 'adx'].dropna()
        
        common_final = si_aligned.index.intersection(adx_aligned.index)
        
        df = pd.DataFrame({
            'si': si_aligned.loc[common_final],
            'adx': adx_aligned.loc[common_final]
        }).dropna()
        
        # OLS with HAC standard errors
        model = OLS(df['adx'], sm.add_constant(df['si'])).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
        
        hac_results[name] = {
            'beta': float(model.params['si']),
            'beta_tstat': float(model.tvalues['si']),
            'beta_pvalue': float(model.pvalues['si']),
            'r_squared': float(model.rsquared),
        }
        
        sig = "✓" if model.pvalues['si'] < 0.05 else "✗"
        print(f"    {name}: β={model.params['si']:.3f}, t={model.tvalues['si']:.2f}, p={model.pvalues['si']:.4f} [{sig}]")
        
        if model.pvalues['si'] < 0.01:
            discover(f"SI-ADX relationship survives HAC correction in {name}")
    
    all_results['hac'] = hac_results
    
    # ============================================================
    section("8. FACTOR NEUTRALIZATION")
    # ============================================================
    
    print("\n  Residual SI after removing factor exposures")
    
    factor_results = {}
    
    for name, (data, mtype) in assets.items():
        si = compute_si(data)
        features = compute_features(data)
        
        common = si.index.intersection(features.index)
        si_aligned = si.loc[common]
        
        # "Factors": volatility, momentum
        factors = features.loc[common, ['volatility', 'momentum']].dropna()
        common_final = si_aligned.index.intersection(factors.index)
        
        si_vals = si_aligned.loc[common_final]
        factor_vals = factors.loc[common_final]
        
        # Regress SI on factors
        model = OLS(si_vals, sm.add_constant(factor_vals)).fit()
        si_residual = model.resid
        
        # Correlation of residual SI with ADX
        adx_aligned = features.loc[common_final, 'adx']
        r_original, _ = spearmanr(si_vals, adx_aligned)
        r_residual, _ = spearmanr(si_residual, adx_aligned)
        
        factor_results[name] = {
            'r_original': float(r_original),
            'r_residual': float(r_residual),
            'factor_r_squared': float(model.rsquared),
            'volatility_beta': float(model.params['volatility']),
            'momentum_beta': float(model.params['momentum']),
        }
        
        print(f"\n  {name}:")
        print(f"    Original SI-ADX:      r = {r_original:+.3f}")
        print(f"    Residual SI-ADX:      r = {r_residual:+.3f}")
        print(f"    Factor exposure R²:   {model.rsquared:.1%}")
        
        if r_residual > 0.05:
            discover(f"SI-ADX survives factor neutralization in {name} (r={r_residual:.3f})")
    
    all_results['factor_neutralization'] = factor_results
    
    # ============================================================
    section("SUMMARY")
    # ============================================================
    
    print(f"\n  Total discoveries: {len(discoveries)}")
    for i, d in enumerate(discoveries, 1):
        print(f"    {i}. {d}")
    
    out_path = Path("results/phase1_expert_suggestions/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'discoveries': discoveries,
            'results': all_results,
        }, f, indent=2, default=str)
    
    discoveries_path = Path("docs/DISCOVERIES_LOG.md")
    with open(discoveries_path, 'a') as f:
        f.write(f"\n\n## Phase 1 Expert Suggestions - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for d in discoveries:
            f.write(f"- {d}\n")
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
