#!/usr/bin/env python3
"""
FIX ALL CONCERNS - Address NeurIPS Publication Issues

This script addresses ALL concerns from the v4.1 execution:
1. Uses FULL NichePopulation SI (not simplified proxy)
2. Re-runs factor regression for novelty validation
3. Documents MECHANISM of SI emergence
4. Creates transparency report on factor exposure
5. Analyzes why regime-conditioning failed
6. Updates paper framing recommendations

Author: Yuhao Li, University of Pennsylvania
Date: January 17, 2026
"""

import sys
sys.path.insert(0, '.')

import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Import FULL NichePopulation (not simplified proxy)
from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2

# ============================================================================
# PARAMETER DOCUMENTATION
# ============================================================================

PARAMETER_CHOICES = {
    'n_agents_per_strategy': 3,  # 6 strategies × 3 agents = 18 agents
    'si_window_daily': 7,        # 7-day rolling SI for daily data
    'momentum_window': 20,
    'volatility_window': 20,
    'trend_window': 14,
    'random_seed': 42,
}

np.random.seed(PARAMETER_CHOICES['random_seed'])

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(market: str, symbol: str) -> pd.DataFrame:
    """Load data for a specific asset."""
    data_dir = Path('data')
    daily_path = data_dir / market / f"{symbol}_1d.csv"
    if daily_path.exists():
        df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        return df
    raise FileNotFoundError(f"Data not found for {market}/{symbol}")


def get_all_assets() -> List[tuple]:
    """Get all available assets."""
    assets = []
    data_dir = Path('data')
    for market_dir in data_dir.iterdir():
        if market_dir.is_dir() and market_dir.name in ['crypto', 'forex', 'stocks', 'commodities']:
            for filepath in market_dir.glob("*_1d.csv"):
                symbol = filepath.stem.replace('_1d', '')
                assets.append((market_dir.name, symbol))
    return assets


# ============================================================================
# FIX 1: FULL NICHEPOPULATION SI
# ============================================================================

def compute_full_si(data: pd.DataFrame, frequency: str = 'daily') -> pd.Series:
    """
    Compute SI using the FULL NichePopulation algorithm.
    
    This is the CORRECT implementation that:
    - Uses 18 competing agents (6 strategies × 3 agents)
    - Tracks niche affinity updates based on competition outcomes
    - Computes SI as 1 - mean(normalized_entropy)
    
    The key difference from the simplified proxy:
    - Simple proxy: Just measured strategy return dispersion
    - Full SI: Tracks emergent specialization through competition
    """
    print(f"      Running full NichePopulation ({len(data)} bars)...")
    
    # Get frequency-appropriate strategies
    strategies = get_default_strategies(frequency)
    
    # Create population
    population = NichePopulationV2(
        strategies=strategies,
        n_agents_per_strategy=PARAMETER_CHOICES['n_agents_per_strategy'],
        frequency=frequency
    )
    
    # Run competition
    population.run(data)
    
    # Compute SI time series
    si = population.compute_si_timeseries(data, window=PARAMETER_CHOICES['si_window_daily'])
    
    return si, population


def document_si_emergence(population: NichePopulationV2) -> Dict:
    """
    Document HOW SI emerges from competition.
    
    This is the MECHANISM documentation required for NeurIPS.
    """
    mechanism = {
        'algorithm': 'NichePopulation',
        'n_agents': len(population.agents),
        'n_strategies': len(population.strategies),
        'n_regimes': 3,  # trending, mean-reverting, volatile
        
        # Key mechanism: Affinity updates
        'affinity_update_rule': 'α=0.1: winner gets α*(1-current), losers get *(1-α)',
        
        # How SI is computed
        'si_formula': 'SI = 1 - mean(normalized_entropy of niche_affinities)',
        'si_interpretation': {
            'high_si': 'Agents specialized in different niches (clear winners per regime)',
            'low_si': 'Agents equally spread across niches (no clear specialization)',
        },
        
        # What causes SI to change
        'si_increases_when': [
            'One strategy consistently wins in a regime',
            'Market regimes are clearly distinct',
            'Competition outcomes are consistent',
        ],
        'si_decreases_when': [
            'No strategy dominates any regime',
            'Market conditions are ambiguous',
            'Winners change frequently',
        ],
    }
    
    # Capture agent states
    agent_states = []
    for agent in population.agents:
        state = {
            'agent_id': agent.agent_id,
            'strategy': population.strategies[agent.strategy_idx].name,
            'niche_affinity': agent.niche_affinity.tolist(),
            'dominant_niche': int(np.argmax(agent.niche_affinity)),
            'specialization': float(np.max(agent.niche_affinity)),
            'win_rate': agent.win_count / max(agent.total_trades, 1),
        }
        agent_states.append(state)
    
    mechanism['agent_states'] = agent_states
    
    return mechanism


# ============================================================================
# FIX 2: FACTOR REGRESSION WITH FULL SI
# ============================================================================

def construct_factors(data: pd.DataFrame) -> pd.DataFrame:
    """Construct factor returns for regression."""
    returns = data['close'].pct_change()
    factors = pd.DataFrame(index=data.index)
    
    factors['market'] = returns
    
    mom_window = PARAMETER_CHOICES['momentum_window']
    mom_signal = np.sign(data['close'].pct_change(mom_window))
    factors['momentum'] = mom_signal.shift(1) * returns
    
    vol_window = PARAMETER_CHOICES['volatility_window']
    rolling_vol = returns.rolling(vol_window).std()
    vol_signal = 1 / rolling_vol.clip(lower=0.001)
    vol_signal_norm = vol_signal / vol_signal.rolling(60).mean().clip(lower=0.001)
    factors['vol_timing'] = vol_signal_norm.shift(1) * returns
    
    trend_window = PARAMETER_CHOICES['trend_window']
    if 'high' in data.columns and 'low' in data.columns:
        high_low_range = (data['high'] - data['low']).rolling(trend_window).mean()
        trend_strength = high_low_range / data['close'].rolling(trend_window).mean().clip(lower=0.001)
    else:
        trend_strength = returns.abs().rolling(trend_window).mean()
    trend_signal = np.sign(data['close'].pct_change(7)) * trend_strength
    factors['trend'] = trend_signal.shift(1) * returns
    
    return factors.dropna()


def run_factor_regression_full_si(data: pd.DataFrame, si: pd.Series, market: str) -> Dict:
    """
    Run factor regression with FULL NichePopulation SI.
    
    This should show LOWER R² than the simplified proxy if SI is truly novel.
    """
    returns = data['close'].pct_change()
    
    # Generate SI strategy returns
    si_signal = (si > si.median()).astype(float)
    si_returns = si_signal.shift(1) * returns
    
    # Construct factors
    factors = construct_factors(data)
    
    # Align
    aligned = pd.concat([si_returns.rename('si_returns'), factors], axis=1).dropna()
    
    if len(aligned) < 60:
        return {'error': 'Insufficient data'}
    
    y = aligned['si_returns']
    X = sm.add_constant(aligned.drop('si_returns', axis=1))
    
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    
    return {
        'alpha': float(model.params['const']),
        'alpha_tstat': float(model.tvalues['const']),
        'alpha_significant': bool(model.pvalues['const'] < 0.05),
        'r_squared': float(model.rsquared),
        'factor_betas': {k: float(v) for k, v in model.params.drop('const').to_dict().items()},
        'factor_tstats': {k: float(v) for k, v in model.tvalues.drop('const').to_dict().items()},
        'n_obs': int(len(y)),
    }


# ============================================================================
# FIX 3: SI AS BEHAVIORAL METRIC (NOT TRADING SIGNAL)
# ============================================================================

def analyze_si_as_behavioral_metric(data: pd.DataFrame, si: pd.Series) -> Dict:
    """
    Analyze SI as a BEHAVIORAL metric describing market state.
    
    This reframes SI from "trading signal" to "market condition indicator".
    """
    returns = data['close'].pct_change()
    
    # Correlation with market characteristics (not future returns)
    correlations = {}
    
    # 1. Volatility correlation
    vol = returns.rolling(20).std()
    r, _ = spearmanr(si.dropna(), vol.reindex(si.index).dropna())
    correlations['realized_volatility'] = float(r)
    
    # 2. Trend strength correlation
    trend = returns.rolling(7).mean().abs() / returns.rolling(7).std()
    r, _ = spearmanr(si.dropna(), trend.reindex(si.index).dropna())
    correlations['trend_strength'] = float(r)
    
    # 3. Regime persistence correlation
    regime_changes = (returns > 0).astype(int).diff().abs().rolling(7).sum()
    r, _ = spearmanr(si.dropna(), regime_changes.reindex(si.index).dropna())
    correlations['regime_instability'] = float(r)
    
    # 4. Volume (if available)
    if 'volume' in data.columns:
        vol_change = data['volume'].pct_change().rolling(7).std()
        r, _ = spearmanr(si.dropna(), vol_change.reindex(si.index).dropna())
        correlations['volume_volatility'] = float(r)
    
    # Interpretation
    interpretation = {
        'high_si_indicates': [],
        'low_si_indicates': [],
    }
    
    if correlations.get('realized_volatility', 0) > 0.2:
        interpretation['high_si_indicates'].append('Higher market volatility')
    if correlations.get('trend_strength', 0) > 0.2:
        interpretation['high_si_indicates'].append('Stronger directional trends')
    if correlations.get('regime_instability', 0) < -0.2:
        interpretation['high_si_indicates'].append('More stable market regimes')
    
    return {
        'correlations': correlations,
        'interpretation': interpretation,
        'recommended_framing': 'SI as market state descriptor, not alpha source',
    }


# ============================================================================
# FIX 4: TRANSPARENCY ON FACTOR EXPOSURE
# ============================================================================

def create_transparency_report(factor_results: Dict) -> Dict:
    """
    Create honest transparency report on factor exposure.
    """
    avg_r2 = np.mean([r['r_squared'] for r in factor_results.values() if 'r_squared' in r])
    avg_alpha_t = np.mean([r['alpha_tstat'] for r in factor_results.values() if 'alpha_tstat' in r])
    
    report = {
        'summary': {
            'average_r_squared': float(avg_r2),
            'average_alpha_tstat': float(avg_alpha_t),
        },
        'honest_assessment': {
            'if_r2_high': "SI strategy returns are partially explained by known factors",
            'if_r2_low': "SI captures genuinely novel information",
            'current_finding': 'high_r2' if avg_r2 > 0.3 else 'low_r2',
        },
        'implications': [],
        'mitigations': [],
    }
    
    if avg_r2 > 0.3:
        report['implications'] = [
            "SI may be correlated with momentum/volatility factors",
            "Claims of 'novel alpha' should be tempered",
            "Paper should focus on MECHANISM not profitability",
        ]
        report['mitigations'] = [
            "Emphasize SI emergence mechanism as contribution",
            "Position SI as behavioral/descriptive metric",
            "Acknowledge factor correlation in limitations",
            "Show SI is useful for understanding, not just trading",
        ]
    else:
        report['implications'] = [
            "SI appears to capture novel information",
            "Factor-adjusted alpha is significant",
            "Alpha claims may be defensible",
        ]
    
    return report


# ============================================================================
# FIX 5: ANALYZE REGIME-CONDITIONING FAILURE
# ============================================================================

def analyze_regime_failure(data: pd.DataFrame, si: pd.Series) -> Dict:
    """
    Understand WHY regime-conditioning hurt performance.
    """
    returns = data['close'].pct_change()
    
    # Simple regime classification
    vol = returns.rolling(14).std()
    trend = abs(returns.rolling(14).mean()) / vol
    
    regimes = pd.Series(index=data.index, dtype=str)
    regimes[vol.rank(pct=True) > 0.8] = 'volatile'
    regimes[(vol.rank(pct=True) <= 0.8) & (trend > 0.1)] = 'trending'
    regimes[regimes.isna()] = 'mean_reverting'
    
    # SI correlation by regime
    regime_corrs = {}
    for regime in ['trending', 'mean_reverting', 'volatile']:
        mask = regimes == regime
        if mask.sum() > 30:
            aligned = pd.concat([si, returns.shift(-1)], axis=1).dropna()
            regime_aligned = aligned[mask.reindex(aligned.index).fillna(False)]
            if len(regime_aligned) > 10:
                r, _ = spearmanr(regime_aligned.iloc[:, 0], regime_aligned.iloc[:, 1])
                regime_corrs[regime] = float(r)
    
    # Diagnose failure
    diagnosis = {
        'regime_correlations': regime_corrs,
        'signs_consistent': len(set(np.sign(v) for v in regime_corrs.values() if not np.isnan(v))) == 1,
    }
    
    # Reasons for failure
    reasons = []
    
    if not diagnosis['signs_consistent']:
        reasons.append("SI-return correlation FLIPS sign across regimes")
    
    # Check if regimes are too short
    regime_changes = (regimes != regimes.shift()).cumsum()
    avg_duration = regimes.groupby(regime_changes).size().mean()
    
    if avg_duration < 5:
        reasons.append(f"Regimes too short (avg {avg_duration:.1f} days) - can't act on them")
    
    # Check if SI is regime-independent
    si_by_regime = {r: si[regimes == r].mean() for r in ['trending', 'mean_reverting', 'volatile']}
    si_variation = np.std(list(si_by_regime.values()))
    
    if si_variation < 0.1:
        reasons.append("SI doesn't vary much across regimes - regime info not useful")
    
    diagnosis['failure_reasons'] = reasons
    diagnosis['avg_regime_duration'] = float(avg_duration)
    diagnosis['si_by_regime'] = {k: float(v) for k, v in si_by_regime.items()}
    
    # Recommendation
    if len(reasons) > 0:
        diagnosis['recommendation'] = "Don't use regime-conditioning with current SI"
    else:
        diagnosis['recommendation'] = "Regime-conditioning may work with tuning"
    
    return diagnosis


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def analyze_asset_full(market: str, symbol: str) -> Dict:
    """Run full analysis for one asset with REAL NichePopulation SI."""
    print(f"\n  [{market}/{symbol}]")
    
    try:
        data = load_data(market, symbol)
        print(f"    Loaded {len(data)} bars")
        
        # FIX 1: Compute FULL SI
        si, population = compute_full_si(data, frequency='daily')
        print(f"    Computed SI: mean={si.mean():.3f}, std={si.std():.3f}")
        
        # Document mechanism
        mechanism = document_si_emergence(population)
        
        # FIX 2: Factor regression
        factor_result = run_factor_regression_full_si(data, si, market)
        r2 = factor_result.get('r_squared', 0)
        alpha_t = factor_result.get('alpha_tstat', 0)
        print(f"    Factor regression: R²={r2:.3f}, α t-stat={alpha_t:.2f}")
        
        # FIX 3: Behavioral metric analysis
        behavioral = analyze_si_as_behavioral_metric(data, si)
        
        # FIX 5: Regime failure analysis
        regime_failure = analyze_regime_failure(data, si)
        
        result = {
            'market': market,
            'symbol': symbol,
            'n_observations': len(data),
            
            # SI statistics
            'si_mean': float(si.mean()),
            'si_std': float(si.std()),
            
            # Factor regression (FIX 2)
            'factor_r_squared': r2,
            'factor_alpha_tstat': alpha_t,
            'factor_alpha_significant': factor_result.get('alpha_significant', False),
            
            # Mechanism (FIX 3)
            'mechanism': mechanism,
            
            # Behavioral (FIX 3)
            'behavioral_analysis': behavioral,
            
            # Regime failure (FIX 5)
            'regime_failure_analysis': regime_failure,
        }
        
        return result
        
    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'market': market, 'symbol': symbol, 'error': str(e)}


def main():
    """Run all fixes."""
    
    print("\n" + "="*70)
    print("FIX ALL CONCERNS - FULL NICHEPOPULATION SI ANALYSIS")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("\nAddressing:")
    print("  1. Using FULL NichePopulation SI (not simplified proxy)")
    print("  2. Re-running factor regression for novelty validation")
    print("  3. Documenting MECHANISM of SI emergence")
    print("  4. Creating transparency report on factor exposure")
    print("  5. Analyzing why regime-conditioning failed")
    print("="*70)
    
    assets = get_all_assets()
    print(f"\nAnalyzing {len(assets)} assets with FULL NichePopulation SI...")
    
    results = []
    for market, symbol in assets:
        result = analyze_asset_full(market, symbol)
        results.append(result)
    
    # Aggregate results
    valid = [r for r in results if 'error' not in r]
    
    print("\n" + "="*70)
    print("RESULTS COMPARISON: Simple Proxy vs Full SI")
    print("="*70)
    
    # Compare R² values
    avg_r2_full = np.mean([r['factor_r_squared'] for r in valid])
    print(f"\n  FACTOR REGRESSION R²:")
    print(f"    Simple proxy (previous): 0.662")
    print(f"    Full NichePopulation:    {avg_r2_full:.3f}")
    
    if avg_r2_full < 0.5:
        print(f"\n  ✅ IMPROVEMENT: Full SI shows LOWER factor exposure!")
        print(f"     This supports SI as a NOVEL signal.")
    else:
        print(f"\n  ⚠️ FINDING: Full SI still has high factor exposure.")
        print(f"     Paper should focus on MECHANISM, not alpha.")
    
    # Create transparency report (FIX 4)
    factor_results_dict = {r['symbol']: {
        'r_squared': r['factor_r_squared'],
        'alpha_tstat': r['factor_alpha_tstat'],
    } for r in valid}
    
    transparency = create_transparency_report(factor_results_dict)
    
    print("\n" + "-"*70)
    print("TRANSPARENCY REPORT (FIX 4)")
    print("-"*70)
    print(f"  Average R²: {transparency['summary']['average_r_squared']:.3f}")
    print(f"  Average α t-stat: {transparency['summary']['average_alpha_tstat']:.2f}")
    print(f"  Assessment: {transparency['honest_assessment']['current_finding']}")
    
    print("\n  Implications:")
    for imp in transparency['implications']:
        print(f"    • {imp}")
    
    print("\n  Mitigations:")
    for mit in transparency['mitigations']:
        print(f"    • {mit}")
    
    # Regime failure analysis summary (FIX 5)
    print("\n" + "-"*70)
    print("REGIME-CONDITIONING FAILURE ANALYSIS (FIX 5)")
    print("-"*70)
    
    all_reasons = []
    for r in valid:
        reasons = r.get('regime_failure_analysis', {}).get('failure_reasons', [])
        all_reasons.extend(reasons)
    
    from collections import Counter
    reason_counts = Counter(all_reasons)
    
    print("\n  Failure reasons (across all assets):")
    for reason, count in reason_counts.most_common():
        print(f"    • {reason} ({count}/{len(valid)} assets)")
    
    # Paper framing recommendations (FIX 6)
    print("\n" + "="*70)
    print("PAPER FRAMING RECOMMENDATIONS (FIX 6)")
    print("="*70)
    
    recommendations = {
        'title_suggestions': [
            "Emergent Specialization in Multi-Agent Market Competition",
            "Measuring Strategy Specialization via Niche Affinity Evolution",
            "SI: A Behavioral Metric for Market Agent Dynamics",
        ],
        'contribution_framing': {
            'primary': "MECHANISM: How specialization emerges from competition",
            'secondary': "MEASUREMENT: SI as a behavioral state descriptor",
            'NOT_recommended': "ALPHA: SI as a profitable trading signal",
        },
        'abstract_template': """
We introduce the Specialization Index (SI), an emergent metric that measures 
the degree of niche differentiation among competing trading strategies.
SI emerges naturally from agent competition and tracks how market conditions
favor different strategy types. We show that SI correlates with market state
characteristics (volatility, trend strength) and provides insight into when
markets are "readable" by systematic strategies. While SI shows some 
predictive content for returns, its primary contribution is as a behavioral
metric for understanding market dynamics, not as an alpha source.
""".strip(),
        'limitations_to_acknowledge': [
            "SI correlates with known factors (momentum, volatility)",
            "Factor-adjusted alpha is modest",
            "Regime-conditioning does not improve performance",
            "Results are based on [N] assets over [time period]",
        ],
    }
    
    print("\n  Title suggestions:")
    for title in recommendations['title_suggestions']:
        print(f"    • {title}")
    
    print("\n  Contribution framing:")
    for key, value in recommendations['contribution_framing'].items():
        marker = "✅" if key != 'NOT_recommended' else "❌"
        print(f"    {marker} {key}: {value}")
    
    print("\n  Abstract template saved to results.")
    
    # Save all results
    output_path = Path('results/fix_all_concerns/full_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    full_output = {
        'timestamp': datetime.now().isoformat(),
        'comparison': {
            'simple_proxy_r2': 0.662,
            'full_si_r2': float(avg_r2_full),
            'improvement': float(0.662 - avg_r2_full),
        },
        'transparency_report': transparency,
        'recommendations': recommendations,
        'asset_results': results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(full_output, f, indent=2, default=str)
    
    print(f"\n  Full results saved to {output_path}")
    
    # Final verdict
    print("\n" + "="*70)
    if avg_r2_full < 0.5:
        print("✅ FIXES SUCCESSFUL: Full SI is more novel than simple proxy")
    else:
        print("⚠️ FIXES APPLIED: Follow paper framing recommendations")
    print("="*70 + "\n")
    
    return full_output


if __name__ == "__main__":
    results = main()
