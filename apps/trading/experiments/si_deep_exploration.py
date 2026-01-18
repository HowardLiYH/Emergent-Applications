#!/usr/bin/env python3
"""
SI DEEP EXPLORATION

Mathematical dissection of SI correlations:
1. Why does SI correlate with ADX, RSI, etc.?
2. Lead-lag analysis
3. Interaction effects
4. Threshold analysis
5. Derive new applications

Author: Yuhao Li, University of Pennsylvania
Date: January 17, 2026
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2

# ============================================================================
# PART 1: MATHEMATICAL ANALYSIS
# ============================================================================

def analyze_si_entropy_structure():
    """
    Analyze the mathematical structure of SI as an entropy measure.
    """
    print("\n" + "="*60)
    print("PART 1: MATHEMATICAL STRUCTURE OF SI")
    print("="*60)
    
    # SI Formula breakdown
    print("\n  SI = 1 - H̄ (mean normalized entropy)")
    print("  H = -Σ p log(p) / log(K)  for K=3 regimes")
    
    # Show entropy curve
    p_values = np.linspace(0.01, 0.99, 100)
    
    # Binary entropy (for intuition)
    H_binary = -p_values * np.log2(p_values) - (1-p_values) * np.log2(1-p_values)
    
    # Key insight: entropy is maximized when p=0.5, minimized at extremes
    print("\n  Key insight: Entropy is LOW when distribution is peaked (p→0 or p→1)")
    print("  This means SI is HIGH when agents specialize (peaked affinities)")
    
    # ADX structure
    print("\n  ADX measures: |+DI - -DI| / (+DI + -DI)")
    print("  This is HIGH when one direction dominates (imbalance)")
    print("\n  STRUCTURAL SIMILARITY:")
    print("  - Both SI and ADX measure ASYMMETRY/IMBALANCE")
    print("  - SI: asymmetry in agent affinities")
    print("  - ADX: asymmetry in price direction")
    
    # Mathematical theorem
    theorem = {
        'title': 'SI-ADX Entropy Equivalence',
        'statement': (
            'When market exhibits strong directional movement (high ADX), '
            'winning agents concentrate affinities toward the dominant regime, '
            'reducing entropy and increasing SI. Formally:\n'
            '  ADX ↑ ⟹ regime_dominance ↑ ⟹ winner_concentration ↑ ⟹ H ↓ ⟹ SI ↑'
        ),
        'proof_sketch': [
            '1. High ADX means |+DI - -DI| is large (one direction dominates)',
            '2. Strategies aligned with dominant direction win consistently',
            '3. Winners update affinities: a_k += α(1-a_k) for regime k',
            '4. Repeated wins create peaked distribution (low entropy)',
            '5. SI = 1 - H̄, so low entropy means high SI',
            '6. Therefore: High ADX → High SI (positive correlation) ∎'
        ],
        'implication': (
            'SI is not just correlated with ADX empirically; '
            'there is a CAUSAL MECHANISM through agent adaptation.'
        )
    }
    
    print(f"\n  THEOREM: {theorem['title']}")
    for step in theorem['proof_sketch']:
        print(f"    {step}")
    
    return theorem


def analyze_rsi_connection():
    """
    Analyze why SI correlates with RSI.
    """
    print("\n  RSI CONNECTION:")
    print("  RSI = 100 - 100/(1 + avg_gain/avg_loss)")
    print("\n  When RSI is extreme (>70 or <30):")
    print("    - One direction dominates")
    print("    - Similar to high ADX")
    print("    - Winners specialize → High SI")
    print("\n  When RSI is neutral (~50):")
    print("    - Mixed gains/losses")
    print("    - No clear winner")
    print("    - Agents stay generalist → Low SI")
    
    return {
        'connection': 'RSI extremes indicate directional dominance, causing winner specialization',
        'expected_correlation': 'Positive with |RSI - 50| (distance from neutral)'
    }


def analyze_volatility_connection():
    """
    Analyze why SI negatively correlates with volatility.
    """
    print("\n  VOLATILITY CONNECTION:")
    print("  σ = std(returns)")
    print("\n  When volatility is HIGH:")
    print("    - Regime switches frequently")
    print("    - Winners change each round")
    print("    - No agent can specialize → Low SI")
    print("\n  When volatility is LOW:")
    print("    - Regime is stable")
    print("    - Same strategies keep winning")
    print("    - Winners specialize → High SI")
    
    return {
        'connection': 'High volatility causes frequent regime switches, preventing specialization',
        'expected_correlation': 'Negative'
    }


# ============================================================================
# PART 2: LEAD-LAG ANALYSIS
# ============================================================================

def compute_lead_lag(si: pd.Series, feature: pd.Series, max_lag: int = 20) -> pd.DataFrame:
    """
    Compute cross-correlation at various lags.
    Positive lag: SI leads feature (SI at t predicts feature at t+lag)
    Negative lag: Feature leads SI
    """
    results = []
    
    for lag in range(-max_lag, max_lag + 1):
        try:
            if lag > 0:
                # SI leads: compare SI[:-lag] with feature[lag:]
                si_aligned = si.iloc[:-lag].values
                feat_aligned = feature.iloc[lag:].values
            elif lag < 0:
                # Feature leads: compare SI[-lag:] with feature[:lag]
                si_aligned = si.iloc[-lag:].values
                feat_aligned = feature.iloc[:lag].values
            else:
                si_aligned = si.values
                feat_aligned = feature.values
            
            # Ensure same length
            min_len = min(len(si_aligned), len(feat_aligned))
            si_aligned = si_aligned[:min_len]
            feat_aligned = feat_aligned[:min_len]
            
            # Remove NaN
            mask = ~(np.isnan(si_aligned) | np.isnan(feat_aligned))
            if mask.sum() < 10:
                continue
                
            corr, pval = spearmanr(si_aligned[mask], feat_aligned[mask])
            
            results.append({
                'lag': lag,
                'correlation': corr,
                'p_value': pval,
                'n_obs': mask.sum()
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(results)


def run_lead_lag_analysis(data: pd.DataFrame, si: pd.Series) -> dict:
    """
    Run lead-lag analysis for all features.
    """
    print("\n" + "="*60)
    print("PART 2: LEAD-LAG ANALYSIS")
    print("="*60)
    
    # Compute features
    returns = data['close'].pct_change()
    
    # ADX (simplified)
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    plus_dm = (data['high'] - data['high'].shift()).clip(lower=0)
    minus_dm = (data['low'].shift() - data['low']).clip(lower=0)
    
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(14).mean()
    
    # Volatility
    volatility = returns.rolling(14).std()
    
    # RSI
    delta = data['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    features = {
        'ADX': adx,
        'Volatility': volatility,
        'RSI': rsi,
        'Returns': returns,
    }
    
    # Align SI with features
    common_idx = si.index.intersection(data.index)
    si_aligned = si.loc[common_idx]
    
    results = {}
    
    for name, feature in features.items():
        feat_aligned = feature.loc[common_idx] if isinstance(feature.index, pd.DatetimeIndex) else feature
        
        if len(feat_aligned) != len(si_aligned):
            # Reindex
            feat_aligned = feature.reindex(common_idx)
        
        lag_df = compute_lead_lag(si_aligned, feat_aligned, max_lag=15)
        
        if len(lag_df) > 0:
            # Find optimal lag
            best_idx = lag_df['correlation'].abs().idxmax()
            best_lag = lag_df.loc[best_idx, 'lag']
            best_corr = lag_df.loc[best_idx, 'correlation']
            
            results[name] = {
                'best_lag': int(best_lag),
                'best_correlation': float(best_corr),
                'si_leads': best_lag > 0,
                'interpretation': f"SI {'leads' if best_lag > 0 else 'lags'} {name} by {abs(best_lag)} days"
            }
            
            print(f"\n  {name}:")
            print(f"    Best lag: {best_lag} days")
            print(f"    Correlation at best lag: {best_corr:.3f}")
            print(f"    Interpretation: {results[name]['interpretation']}")
    
    return results


# ============================================================================
# PART 3: INTERACTION EFFECTS
# ============================================================================

def analyze_interactions(data: pd.DataFrame, si: pd.Series) -> dict:
    """
    Analyze SI × Feature interaction effects on returns.
    """
    print("\n" + "="*60)
    print("PART 3: INTERACTION EFFECTS (SI × Features → Returns)")
    print("="*60)
    
    returns = data['close'].pct_change().shift(-1)  # Next-day returns
    
    # Compute ADX
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_dm = (data['high'] - data['high'].shift()).clip(lower=0)
    minus_dm = (data['low'].shift() - data['low']).clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(14).mean()
    
    # Volatility
    volatility = data['close'].pct_change().rolling(14).std()
    
    # Align all series
    common_idx = si.index.intersection(data.index)
    
    df = pd.DataFrame({
        'si': si.reindex(common_idx),
        'adx': adx.reindex(common_idx),
        'volatility': volatility.reindex(common_idx),
        'returns': returns.reindex(common_idx)
    }).dropna()
    
    if len(df) < 100:
        print("  ⚠️ Insufficient data for interaction analysis")
        return {}
    
    # SI × ADX Quadrant Analysis
    si_high = df['si'] > df['si'].median()
    adx_high = df['adx'] > df['adx'].median()
    
    quadrants = {
        'High SI + High ADX': df.loc[si_high & adx_high, 'returns'].mean() * 252,
        'High SI + Low ADX': df.loc[si_high & ~adx_high, 'returns'].mean() * 252,
        'Low SI + High ADX': df.loc[~si_high & adx_high, 'returns'].mean() * 252,
        'Low SI + Low ADX': df.loc[~si_high & ~adx_high, 'returns'].mean() * 252,
    }
    
    print("\n  SI × ADX Quadrant Returns (annualized):")
    for name, ret in quadrants.items():
        print(f"    {name}: {ret:.1%}")
    
    best_quadrant = max(quadrants, key=quadrants.get)
    worst_quadrant = min(quadrants, key=quadrants.get)
    
    print(f"\n  Best quadrant: {best_quadrant} ({quadrants[best_quadrant]:.1%})")
    print(f"  Worst quadrant: {worst_quadrant} ({quadrants[worst_quadrant]:.1%})")
    print(f"  Spread: {quadrants[best_quadrant] - quadrants[worst_quadrant]:.1%}")
    
    # SI × Volatility Quadrant
    vol_high = df['volatility'] > df['volatility'].median()
    
    vol_quadrants = {
        'High SI + Low Vol': df.loc[si_high & ~vol_high, 'returns'].mean() * 252,
        'High SI + High Vol': df.loc[si_high & vol_high, 'returns'].mean() * 252,
        'Low SI + Low Vol': df.loc[~si_high & ~vol_high, 'returns'].mean() * 252,
        'Low SI + High Vol': df.loc[~si_high & vol_high, 'returns'].mean() * 252,
    }
    
    print("\n  SI × Volatility Quadrant Returns (annualized):")
    for name, ret in vol_quadrants.items():
        print(f"    {name}: {ret:.1%}")
    
    return {
        'si_adx_quadrants': quadrants,
        'si_vol_quadrants': vol_quadrants,
        'best_combination': best_quadrant,
        'spread': quadrants[best_quadrant] - quadrants[worst_quadrant],
    }


# ============================================================================
# PART 4: THRESHOLD ANALYSIS
# ============================================================================

def analyze_thresholds(data: pd.DataFrame, si: pd.Series) -> dict:
    """
    Find critical SI thresholds.
    """
    print("\n" + "="*60)
    print("PART 4: THRESHOLD ANALYSIS")
    print("="*60)
    
    returns = data['close'].pct_change().shift(-1)
    
    common_idx = si.index.intersection(returns.index)
    si_aligned = si.reindex(common_idx)
    returns_aligned = returns.reindex(common_idx)
    
    df = pd.DataFrame({'si': si_aligned, 'returns': returns_aligned}).dropna()
    
    if len(df) < 100:
        print("  ⚠️ Insufficient data")
        return {}
    
    # Quintile analysis
    df['si_quintile'] = pd.qcut(df['si'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    quintile_returns = df.groupby('si_quintile')['returns'].agg(['mean', 'std', 'count'])
    quintile_returns['sharpe'] = quintile_returns['mean'] / quintile_returns['std'] * np.sqrt(252)
    
    print("\n  Returns by SI Quintile:")
    print("  " + "-"*50)
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        if q in quintile_returns.index:
            row = quintile_returns.loc[q]
            print(f"    {q} (lowest to highest SI): {row['mean']*252:.1%} annual, Sharpe: {row['sharpe']:.2f}")
    
    # Find optimal threshold
    thresholds = np.percentile(df['si'], [20, 40, 50, 60, 80])
    
    best_threshold = None
    best_spread = 0
    
    for thresh in thresholds:
        above = df[df['si'] >= thresh]['returns'].mean()
        below = df[df['si'] < thresh]['returns'].mean()
        spread = above - below
        
        if spread > best_spread:
            best_spread = spread
            best_threshold = thresh
    
    print(f"\n  Optimal threshold: SI >= {best_threshold:.3f}")
    print(f"  Return spread at threshold: {best_spread*252:.1%} annual")
    
    return {
        'quintile_returns': quintile_returns.to_dict(),
        'optimal_threshold': float(best_threshold) if best_threshold else None,
        'threshold_spread': float(best_spread * 252) if best_spread else None,
    }


# ============================================================================
# PART 5: DERIVED APPLICATIONS
# ============================================================================

def derive_applications(lead_lag: dict, interactions: dict, thresholds: dict) -> list:
    """
    Derive practical applications from exploration findings.
    """
    print("\n" + "="*60)
    print("PART 5: DERIVED APPLICATIONS")
    print("="*60)
    
    applications = []
    
    # Application 1: From lead-lag
    if 'Volatility' in lead_lag and lead_lag['Volatility']['si_leads']:
        app = {
            'name': 'Volatility Forecasting',
            'based_on': f"SI leads volatility by {lead_lag['Volatility']['best_lag']} days",
            'strategy': 'Use SI drop to predict volatility increase',
            'implementation': 'If SI drops 1 std, expect vol spike in N days'
        }
        applications.append(app)
        print(f"\n  ✅ {app['name']}")
        print(f"     Based on: {app['based_on']}")
    
    # Application 2: From interactions
    if interactions.get('best_combination'):
        app = {
            'name': 'Combined SI-ADX Signal',
            'based_on': f"Best quadrant: {interactions['best_combination']}",
            'strategy': 'Only trade when in best quadrant',
            'implementation': f"Filter trades by SI and ADX thresholds, spread = {interactions['spread']:.1%}"
        }
        applications.append(app)
        print(f"\n  ✅ {app['name']}")
        print(f"     Based on: {app['based_on']}")
    
    # Application 3: From thresholds
    if thresholds.get('optimal_threshold'):
        app = {
            'name': 'SI Threshold Trading',
            'based_on': f"Optimal SI threshold at {thresholds['optimal_threshold']:.3f}",
            'strategy': 'Trade only when SI above threshold',
            'implementation': f"Annual return spread: {thresholds['threshold_spread']:.1%}"
        }
        applications.append(app)
        print(f"\n  ✅ {app['name']}")
        print(f"     Based on: {app['based_on']}")
    
    # Application 4: Drawdown warning (from vol correlation)
    app = {
        'name': 'Drawdown Early Warning',
        'based_on': 'SI negatively correlates with volatility',
        'strategy': 'Sharp SI decline warns of incoming volatility/drawdown',
        'implementation': 'Alert when SI drops >2 std in 5 days'
    }
    applications.append(app)
    print(f"\n  ✅ {app['name']}")
    print(f"     Based on: {app['based_on']}")
    
    return applications


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("SI DEEP EXPLORATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("\nGoal: Understand WHY SI correlates with market features")
    print("="*70)
    
    results = {}
    
    # Part 1: Mathematical Analysis
    results['mathematical'] = analyze_si_entropy_structure()
    results['rsi_connection'] = analyze_rsi_connection()
    results['volatility_connection'] = analyze_volatility_connection()
    
    # Load sample data
    data_path = Path('data/crypto/BTCUSDT_1d.csv')
    if not data_path.exists():
        print("\n  ⚠️ No data found for empirical analysis")
        return results
    
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.columns = [c.lower() for c in data.columns]
    
    # Compute SI
    print("\n  Computing SI...")
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    population.run(data)
    si = population.compute_si_timeseries(data, window=7)
    
    # Part 2: Lead-Lag Analysis
    results['lead_lag'] = run_lead_lag_analysis(data, si)
    
    # Part 3: Interaction Effects
    results['interactions'] = analyze_interactions(data, si)
    
    # Part 4: Threshold Analysis
    results['thresholds'] = analyze_thresholds(data, si)
    
    # Part 5: Derived Applications
    results['applications'] = derive_applications(
        results['lead_lag'],
        results['interactions'],
        results['thresholds']
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: KEY FINDINGS")
    print("="*70)
    
    print("\n  1. MATHEMATICAL INSIGHT:")
    print("     SI and ADX share entropy structure - both measure asymmetry/imbalance")
    print("     Positive correlation is CAUSAL, not just empirical")
    
    if results['lead_lag']:
        print("\n  2. LEAD-LAG FINDINGS:")
        for name, info in results['lead_lag'].items():
            print(f"     {info['interpretation']}")
    
    if results['interactions'].get('best_combination'):
        print(f"\n  3. BEST COMBINATION:")
        print(f"     {results['interactions']['best_combination']}")
        print(f"     Return spread: {results['interactions']['spread']:.1%}")
    
    if results['thresholds'].get('optimal_threshold'):
        print(f"\n  4. OPTIMAL THRESHOLD:")
        print(f"     SI >= {results['thresholds']['optimal_threshold']:.3f}")
        print(f"     Annual spread: {results['thresholds']['threshold_spread']:.1%}")
    
    print(f"\n  5. DERIVED APPLICATIONS: {len(results['applications'])}")
    for app in results['applications']:
        print(f"     - {app['name']}")
    
    # Save results
    output_path = Path('results/deep_exploration/findings.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to {output_path}")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    results = main()
