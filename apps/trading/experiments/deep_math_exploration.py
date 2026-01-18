#!/usr/bin/env python3
"""
DEEP MATHEMATICAL EXPLORATION
1. Check if MCI ≈ SI (is correlation trivial?)
2. Explore statistical distribution aspects
3. Find more mathematical connections
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, pearsonr, entropy, ks_2samp, normaltest
from scipy.special import rel_entr
import warnings
warnings.filterwarnings('ignore')

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2
from src.data.loader_v2 import DataLoaderV2, MarketType

MARKETS = {
    MarketType.CRYPTO: ['BTCUSDT'],
    MarketType.STOCKS: ['SPY'],
}

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def compute_si_with_details(data, window=7):
    """Compute SI and return detailed agent info."""
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    population.run(data)
    si = population.compute_si_timeseries(data, window=window)
    
    # Get agent affinities at each timestep
    agent_entropies = []
    for agent in population.agents:
        p = agent.niche_affinity + 1e-10
        h = -np.sum(p * np.log(p)) / np.log(3)
        agent_entropies.append(h)
    
    return si, population, np.mean(agent_entropies)

def compute_mci(data):
    """Compute Market Clarity Index from price data."""
    plus_dm = (data['high'] - data['high'].shift()).clip(lower=0)
    minus_dm = (data['low'].shift() - data['low']).clip(lower=0)
    
    plus_dm_ma = plus_dm.rolling(14).mean()
    minus_dm_ma = minus_dm.rolling(14).mean()
    
    total = plus_dm_ma + minus_dm_ma + 1e-10
    p_plus = plus_dm_ma / total
    p_minus = minus_dm_ma / total
    
    # Binary entropy
    h_market = -(p_plus * np.log(p_plus + 1e-10) + p_minus * np.log(p_minus + 1e-10))
    mci = 1 - (h_market / np.log(2))
    
    return mci

def main():
    print("\n" + "="*70)
    print("  DEEP MATHEMATICAL EXPLORATION")
    print("="*70)
    
    discoveries = []
    
    loader = DataLoaderV2()
    data = loader.load('BTCUSDT', MarketType.CRYPTO)
    
    section("1. ARE MCI AND SI THE SAME? (Triviality Check)")
    
    si, population, final_agent_entropy = compute_si_with_details(data)
    mci = compute_mci(data)
    
    # Align
    common = si.index.intersection(mci.dropna().index)
    si_aligned = si.loc[common]
    mci_aligned = mci.loc[common]
    
    # Correlation
    r_spearman, _ = spearmanr(si_aligned, mci_aligned)
    r_pearson, _ = pearsonr(si_aligned, mci_aligned)
    
    print(f"\n  Correlation SI vs MCI:")
    print(f"    Spearman r = {r_spearman:.3f}")
    print(f"    Pearson r  = {r_pearson:.3f}")
    
    # Check if they're linearly related
    from sklearn.linear_model import LinearRegression
    X = mci_aligned.values.reshape(-1, 1)
    y = si_aligned.values
    reg = LinearRegression().fit(X, y)
    r_squared = reg.score(X, y)
    
    print(f"    R² (MCI explains SI) = {r_squared:.3f}")
    
    if r_squared > 0.8:
        print("\n  ⚠️ WARNING: MCI and SI are nearly identical (trivial)!")
        discoveries.append("WARNING: MCI-SI correlation may be trivial (R²={:.3f})".format(r_squared))
    elif r_squared > 0.5:
        print("\n  📊 MCI partially explains SI - related but not identical")
        discoveries.append("MCI partially explains SI (R²={:.3f}) - related but distinct".format(r_squared))
    else:
        print("\n  ✅ MCI and SI are distinct concepts (low R²)")
        discoveries.append("MCI and SI are mathematically distinct (R²={:.3f})".format(r_squared))
    
    # Key difference analysis
    print("\n  Key Differences:")
    print("    • MCI: Computed from PRICE (+DM/-DM) - market structure")
    print("    • SI:  Computed from AGENT DYNAMICS - emergent behavior")
    print("    • Correlation shows: market structure → agent specialization")
    
    section("2. STATISTICAL DISTRIBUTION ANALYSIS")
    
    returns = data['close'].pct_change().dropna()
    
    # SI distribution properties
    print("\n  A. SI Distribution:")
    print(f"    Mean: {si.mean():.4f}")
    print(f"    Std:  {si.std():.4f}")
    print(f"    Skewness: {si.skew():.4f}")
    print(f"    Kurtosis: {si.kurtosis():.4f}")
    
    # Test for normality
    stat, p_normal = normaltest(si.dropna())
    print(f"    Normal? p={p_normal:.4f} ({'YES' if p_normal > 0.05 else 'NO'})")
    
    if si.skew() > 0.5:
        discoveries.append("SI is right-skewed ({:.2f}) - agents tend toward specialization".format(si.skew()))
    elif si.skew() < -0.5:
        discoveries.append("SI is left-skewed ({:.2f}) - agents tend toward generalization".format(si.skew()))
    
    # Return distribution in high vs low SI regimes
    print("\n  B. Return Distribution by SI Regime:")
    si_median = si.median()
    
    # Align returns with SI
    common_ret = si.index.intersection(returns.index)
    si_for_ret = si.loc[common_ret]
    ret_aligned = returns.loc[common_ret]
    
    high_si_mask = si_for_ret > si_median
    low_si_mask = si_for_ret <= si_median
    
    ret_high_si = ret_aligned[high_si_mask]
    ret_low_si = ret_aligned[low_si_mask]
    
    print(f"    High SI periods: mean={ret_high_si.mean()*100:.3f}%, std={ret_high_si.std()*100:.3f}%")
    print(f"    Low SI periods:  mean={ret_low_si.mean()*100:.3f}%, std={ret_low_si.std()*100:.3f}%")
    
    # KS test - are distributions different?
    ks_stat, ks_p = ks_2samp(ret_high_si.dropna(), ret_low_si.dropna())
    print(f"    KS test p={ks_p:.4f} ({'DIFFERENT' if ks_p < 0.05 else 'SIMILAR'})")
    
    if ks_p < 0.05:
        discoveries.append("Return distributions differ between high/low SI regimes (KS p={:.4f})".format(ks_p))
    
    # Tail behavior
    print("\n  C. Tail Risk Analysis:")
    ret_5th = ret_aligned.quantile(0.05)
    ret_95th = ret_aligned.quantile(0.95)
    
    extreme_down = ret_aligned < ret_5th
    extreme_up = ret_aligned > ret_95th
    
    # SI before extreme moves
    si_before_extreme_down = si_for_ret.shift(1)[extreme_down].mean()
    si_before_extreme_up = si_for_ret.shift(1)[extreme_up].mean()
    si_before_normal = si_for_ret.shift(1)[~extreme_down & ~extreme_up].mean()
    
    print(f"    SI before extreme DOWN: {si_before_extreme_down:.4f}")
    print(f"    SI before extreme UP:   {si_before_extreme_up:.4f}")
    print(f"    SI before normal:       {si_before_normal:.4f}")
    
    if abs(si_before_extreme_down - si_before_normal) > 0.02:
        discoveries.append("SI differs before extreme down moves ({:.4f} vs {:.4f})".format(
            si_before_extreme_down, si_before_normal))
    
    section("3. ENTROPY DECOMPOSITION")
    
    print("\n  SI = 1 - mean(H_agent)")
    print("  Let's decompose what drives agent entropy:")
    
    # Get agent affinities from final state
    print("\n  Final Agent Affinities (after full competition):")
    for i, agent in enumerate(population.agents[:6]):  # Show first 6
        p = agent.niche_affinity
        h = -np.sum(p * np.log(p + 1e-10)) / np.log(3)
        print(f"    Agent {i}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] H={h:.3f}")
    
    section("4. HIGHER-ORDER CORRELATIONS")
    
    print("\n  Testing SI correlations with higher-order statistics:")
    
    # Realized volatility (different from simple std)
    realized_vol = np.sqrt((returns**2).rolling(14).sum())
    
    # Realized skewness
    realized_skew = returns.rolling(14).skew()
    
    # Realized kurtosis
    realized_kurt = returns.rolling(14).kurt()
    
    # Hurst exponent proxy (variance ratio)
    var_1 = returns.rolling(7).var()
    var_2 = returns.rolling(14).var()
    hurst_proxy = np.log(var_2 / (2 * var_1 + 1e-10)) / np.log(2)
    
    # Autocorrelation of returns
    autocorr = returns.rolling(14).apply(lambda x: x.autocorr(lag=1), raw=False)
    
    features = {
        'realized_vol': realized_vol,
        'realized_skew': realized_skew,
        'realized_kurt': realized_kurt,
        'hurst_proxy': hurst_proxy,
        'autocorr': autocorr,
    }
    
    print(f"\n  {'Feature':<20} {'Corr with SI':>15} {'Consistent':>12}")
    print("  " + "-"*50)
    
    for name, feat in features.items():
        common = si.index.intersection(feat.dropna().index)
        if len(common) > 100:
            r, p = spearmanr(si.loc[common], feat.loc[common])
            sig = "✅" if p < 0.05 else ""
            print(f"  {name:<20} {r:>+15.3f} {sig:>12}")
            if abs(r) > 0.1 and p < 0.05:
                discoveries.append(f"SI correlates with {name} (r={r:.3f})")
    
    section("5. INFORMATION-THEORETIC MEASURES")
    
    print("\n  Computing mutual information between SI and features:")
    
    # Discretize for MI calculation
    si_discrete = pd.qcut(si.dropna(), q=5, labels=False, duplicates='drop')
    
    # Returns discretized
    ret_common = returns.loc[si_discrete.index].dropna()
    si_discrete_common = si_discrete.loc[ret_common.index]
    ret_discrete = pd.qcut(ret_common, q=5, labels=False, duplicates='drop')
    
    # Joint distribution
    if len(si_discrete_common) > 100 and len(ret_discrete) > 100:
        # Align indices
        common_idx = si_discrete_common.index.intersection(ret_discrete.index)
        si_d = si_discrete_common.loc[common_idx]
        ret_d = ret_discrete.loc[common_idx]
        
        joint = pd.crosstab(si_d, ret_d, normalize=True)
        
        # Marginals
        p_si = joint.sum(axis=1)
        p_ret = joint.sum(axis=0)
        
        # Mutual information
        mi = 0
        for i in joint.index:
            for j in joint.columns:
                if joint.loc[i, j] > 0:
                    mi += joint.loc[i, j] * np.log(joint.loc[i, j] / (p_si[i] * p_ret[j] + 1e-10) + 1e-10)
        
        print(f"    Mutual Information I(SI; Returns) = {mi:.4f} nats")
        
        # Normalized
        h_si = -np.sum(p_si * np.log(p_si + 1e-10))
        h_ret = -np.sum(p_ret * np.log(p_ret + 1e-10))
        nmi = mi / min(h_si, h_ret) if min(h_si, h_ret) > 0 else 0
        
        print(f"    Normalized MI = {nmi:.4f}")
        
        if nmi > 0.05:
            discoveries.append(f"SI and returns share mutual information (NMI={nmi:.4f})")
    
    section("6. LEAD-LAG STRUCTURE")
    
    print("\n  Cross-correlation SI(t) vs Returns(t+k):")
    
    for lag in [-5, -3, -1, 0, 1, 3, 5]:
        ret_shifted = returns.shift(-lag)
        common = si.index.intersection(ret_shifted.dropna().index)
        if len(common) > 100:
            r, p = spearmanr(si.loc[common], ret_shifted.loc[common])
            arrow = "←" if lag < 0 else ("→" if lag > 0 else "=")
            print(f"    SI(t) vs Returns(t{lag:+d}): r={r:+.3f} {'*' if p < 0.05 else ''}")
    
    section("7. NOVEL MATHEMATICAL CONNECTIONS")
    
    print("\n  A. SI as Information Ratio:")
    print("    If SI measures 'clarity', then 1/SI might relate to noise")
    
    noise_proxy = 1 / (si + 0.1)  # Add small constant to avoid div by zero
    r_noise_vol, _ = spearmanr(noise_proxy.dropna(), 
                                realized_vol.loc[noise_proxy.dropna().index].dropna())
    print(f"    Corr(1/SI, realized_vol) = {r_noise_vol:.3f}")
    
    if abs(r_noise_vol) > 0.1:
        discoveries.append(f"1/SI correlates with realized volatility (r={r_noise_vol:.3f}) - SI as signal/noise measure")
    
    print("\n  B. SI Derivative (momentum of specialization):")
    si_diff = si.diff()
    
    # Does change in SI predict future returns?
    ret_future = returns.shift(-1)
    common = si_diff.dropna().index.intersection(ret_future.dropna().index)
    if len(common) > 100:
        r_si_diff, p = spearmanr(si_diff.loc[common], ret_future.loc[common])
        print(f"    Corr(dSI/dt, future_return) = {r_si_diff:.3f} (p={p:.4f})")
        
        if abs(r_si_diff) > 0.05 and p < 0.05:
            discoveries.append(f"SI momentum predicts returns (r={r_si_diff:.3f})")
    
    print("\n  C. SI Acceleration:")
    si_accel = si_diff.diff()
    common = si_accel.dropna().index.intersection(ret_future.dropna().index)
    if len(common) > 100:
        r_si_accel, p = spearmanr(si_accel.loc[common], ret_future.loc[common])
        print(f"    Corr(d²SI/dt², future_return) = {r_si_accel:.3f} (p={p:.4f})")
    
    section("DISCOVERIES SUMMARY")
    
    print("\n  Key findings from deep exploration:")
    for i, d in enumerate(discoveries, 1):
        print(f"    {i}. {d}")
    
    # Save discoveries
    discoveries_path = Path("docs/DISCOVERIES_LOG.md")
    with open(discoveries_path, 'a') as f:
        f.write(f"\n\n## Deep Math Exploration - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for d in discoveries:
            f.write(f"- {d}\n")
    
    print(f"\n  Discoveries logged to {discoveries_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
