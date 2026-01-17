#!/usr/bin/env python3
"""
SI as RISK INDICATOR - Deep Analysis

Tests SI as a risk management tool rather than directional signal:
1. Drawdown prediction - Does low SI precede drawdowns?
2. Volatility timing - Does SI predict future volatility?
3. Position scaling - Scale position size by SI (not direction)
4. Tail risk warning - Does low SI indicate higher tail risk?

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
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

warnings.filterwarnings('ignore')

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2

# ============================================================================
# PARAMETERS
# ============================================================================

PARAMS = {
    'n_agents_per_strategy': 3,
    'si_window': 7,
    'vol_forecast_horizons': [1, 5, 10, 20],  # Days ahead
    'dd_lookback': 20,
    'random_seed': 42,
}

np.random.seed(PARAMS['random_seed'])

# ============================================================================
# DATA
# ============================================================================

def load_data(market: str, symbol: str) -> pd.DataFrame:
    data_dir = Path('data')
    daily_path = data_dir / market / f"{symbol}_1d.csv"
    if daily_path.exists():
        df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        return df
    raise FileNotFoundError(f"Data not found for {market}/{symbol}")


def get_all_assets() -> List[tuple]:
    assets = []
    data_dir = Path('data')
    for market_dir in data_dir.iterdir():
        if market_dir.is_dir() and market_dir.name in ['crypto', 'forex', 'stocks', 'commodities']:
            for filepath in market_dir.glob("*_1d.csv"):
                symbol = filepath.stem.replace('_1d', '')
                assets.append((market_dir.name, symbol))
    return assets


def compute_full_si(data: pd.DataFrame) -> pd.Series:
    """Compute SI using full NichePopulation."""
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(
        strategies=strategies,
        n_agents_per_strategy=PARAMS['n_agents_per_strategy'],
        frequency='daily'
    )
    population.run(data)
    si = population.compute_si_timeseries(data, window=PARAMS['si_window'])
    return si


# ============================================================================
# TEST 1: DRAWDOWN PREDICTION
# ============================================================================

def test_drawdown_prediction(data: pd.DataFrame, si: pd.Series) -> Dict:
    """
    Test if LOW SI precedes drawdowns.
    
    Hypothesis: When SI is low, markets are "unreadable" and more likely
    to produce losses for systematic strategies.
    """
    returns = data['close'].pct_change()
    
    # Compute rolling drawdown (max loss in next N days)
    future_dd = []
    for i in range(len(returns) - PARAMS['dd_lookback']):
        future_returns = returns.iloc[i+1:i+1+PARAMS['dd_lookback']]
        cumulative = (1 + future_returns).cumprod()
        max_dd = (cumulative / cumulative.cummax() - 1).min()
        future_dd.append({'date': returns.index[i], 'future_dd': max_dd})
    
    future_dd_df = pd.DataFrame(future_dd).set_index('date')['future_dd']
    
    # Align SI with future drawdown
    aligned = pd.concat([si, future_dd_df], axis=1).dropna()
    aligned.columns = ['si', 'future_dd']
    
    if len(aligned) < 50:
        return {'error': 'Insufficient data'}
    
    # Correlation: Does LOW SI predict WORSE (more negative) drawdowns?
    corr, pval = spearmanr(aligned['si'], aligned['future_dd'])
    
    # Quantile analysis
    low_si = aligned[aligned['si'] < aligned['si'].quantile(0.25)]
    high_si = aligned[aligned['si'] > aligned['si'].quantile(0.75)]
    
    avg_dd_low_si = low_si['future_dd'].mean()
    avg_dd_high_si = high_si['future_dd'].mean()
    
    # Does low SI lead to worse drawdowns?
    low_si_worse = avg_dd_low_si < avg_dd_high_si  # More negative = worse
    
    return {
        'correlation_si_dd': float(corr),
        'correlation_pval': float(pval),
        'significant': bool(pval < 0.05),
        'avg_dd_when_low_si': float(avg_dd_low_si),
        'avg_dd_when_high_si': float(avg_dd_high_si),
        'low_si_predicts_worse_dd': bool(low_si_worse),
        'dd_difference': float(avg_dd_high_si - avg_dd_low_si),  # Positive = high SI is better
    }


# ============================================================================
# TEST 2: VOLATILITY FORECASTING
# ============================================================================

def test_volatility_forecasting(data: pd.DataFrame, si: pd.Series) -> Dict:
    """
    Test if SI predicts future volatility.
    
    Hypothesis: SI may be related to how "structured" the market is,
    which could predict volatility regimes.
    """
    returns = data['close'].pct_change()
    
    results = {}
    
    for horizon in PARAMS['vol_forecast_horizons']:
        # Future realized volatility
        future_vol = returns.rolling(horizon).std().shift(-horizon)
        
        # Align
        aligned = pd.concat([si, future_vol], axis=1).dropna()
        if len(aligned) < 50:
            continue
        
        aligned.columns = ['si', 'future_vol']
        
        # Correlation
        corr, pval = spearmanr(aligned['si'], aligned['future_vol'])
        
        # R² from simple regression
        from sklearn.linear_model import LinearRegression
        X = aligned['si'].values.reshape(-1, 1)
        y = aligned['future_vol'].values
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        
        results[f'{horizon}d'] = {
            'correlation': float(corr),
            'pval': float(pval),
            'significant': bool(pval < 0.05),
            'r_squared': float(r2),
            'predictive': bool(pval < 0.05 and r2 > 0.01),
        }
    
    # Summary
    predictive_horizons = [h for h, r in results.items() if r.get('predictive', False)]
    
    return {
        'by_horizon': results,
        'predictive_horizons': predictive_horizons,
        'any_predictive': len(predictive_horizons) > 0,
    }


# ============================================================================
# TEST 3: POSITION SCALING
# ============================================================================

def test_position_scaling(data: pd.DataFrame, si: pd.Series, market: str) -> Dict:
    """
    Test SI for position sizing (not direction).
    
    Strategy: Always long, but scale position by SI.
    Compare to: Constant sizing, Inverse vol sizing.
    """
    returns = data['close'].pct_change()
    
    costs = {'crypto': 0.0006, 'forex': 0.0001, 'stocks': 0.0002, 'commodities': 0.0003}
    cost = costs.get(market, 0.0002)
    
    # Align SI with returns
    aligned_si = si.reindex(returns.index).fillna(si.median())
    
    # Strategy 1: Constant (baseline)
    const_returns = returns.copy()
    
    # Strategy 2: SI scaling (high SI = larger position)
    si_norm = (aligned_si - aligned_si.min()) / (aligned_si.max() - aligned_si.min() + 1e-10)
    si_scale = 0.5 + si_norm  # Range [0.5, 1.5]
    si_returns = returns * si_scale.shift(1)
    
    # Apply costs
    si_trades = si_scale.diff().abs().fillna(0)
    si_net = si_returns - si_trades * cost
    
    # Strategy 3: Inverse SI (low SI = larger position, contrarian)
    inv_si_scale = 1.5 - si_norm  # Range [0.5, 1.5] but inverted
    inv_si_returns = returns * inv_si_scale.shift(1)
    inv_si_trades = inv_si_scale.diff().abs().fillna(0)
    inv_si_net = inv_si_returns - inv_si_trades * cost
    
    # Metrics
    def compute_metrics(rets):
        rets = rets.dropna()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        cum = (1 + rets).cumprod()
        max_dd = ((cum / cum.cummax()) - 1).min()
        return {'sharpe': float(sharpe), 'max_dd': float(max_dd), 'total_return': float(cum.iloc[-1] - 1)}
    
    const_metrics = compute_metrics(const_returns)
    si_metrics = compute_metrics(si_net)
    inv_si_metrics = compute_metrics(inv_si_net)
    
    # Which is best?
    best = max([
        ('constant', const_metrics['sharpe']),
        ('si_scale', si_metrics['sharpe']),
        ('inv_si_scale', inv_si_metrics['sharpe']),
    ], key=lambda x: x[1])
    
    return {
        'constant': const_metrics,
        'si_scale': si_metrics,
        'inv_si_scale': inv_si_metrics,
        'best_method': best[0],
        'si_improves_sharpe': si_metrics['sharpe'] > const_metrics['sharpe'],
        'si_reduces_dd': si_metrics['max_dd'] > const_metrics['max_dd'],  # Less negative = better
    }


# ============================================================================
# TEST 4: TAIL RISK WARNING
# ============================================================================

def test_tail_risk(data: pd.DataFrame, si: pd.Series) -> Dict:
    """
    Test if LOW SI indicates higher tail risk (extreme losses).
    
    Hypothesis: Low SI = market is chaotic = higher probability of
    extreme moves.
    """
    returns = data['close'].pct_change()
    
    # Define tail events (worst 5% of days)
    tail_threshold = returns.quantile(0.05)
    is_tail = returns < tail_threshold
    
    # Align SI
    aligned = pd.concat([si, is_tail.rename('is_tail')], axis=1).dropna()
    
    if len(aligned) < 100:
        return {'error': 'Insufficient data'}
    
    # Probability of tail event given SI quintile
    aligned['si_quintile'] = pd.qcut(aligned.iloc[:, 0], 5, labels=['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high'])
    
    tail_prob_by_quintile = aligned.groupby('si_quintile')['is_tail'].mean()
    
    # Does low SI have higher tail probability?
    q1_prob = tail_prob_by_quintile.get('Q1_low', 0)
    q5_prob = tail_prob_by_quintile.get('Q5_high', 0)
    
    low_si_more_risky = q1_prob > q5_prob
    
    # Logistic regression: SI predicting tail events
    from sklearn.linear_model import LogisticRegression
    X = aligned.iloc[:, 0].values.reshape(-1, 1)
    y = aligned['is_tail'].astype(int).values
    
    try:
        model = LogisticRegression().fit(X, y)
        coef = model.coef_[0][0]
        # Negative coef means higher SI = lower tail probability
        si_reduces_tail_risk = coef < 0
    except:
        coef = 0
        si_reduces_tail_risk = False
    
    return {
        'tail_prob_by_quintile': tail_prob_by_quintile.to_dict(),
        'q1_low_si_tail_prob': float(q1_prob),
        'q5_high_si_tail_prob': float(q5_prob),
        'low_si_more_risky': bool(low_si_more_risky),
        'logistic_coef': float(coef),
        'si_reduces_tail_risk': bool(si_reduces_tail_risk),
        'risk_ratio': float(q1_prob / (q5_prob + 0.001)),
    }


# ============================================================================
# MAIN
# ============================================================================

def analyze_asset(market: str, symbol: str) -> Dict:
    """Run all risk indicator tests for one asset."""
    print(f"\n  [{market}/{symbol}]")
    
    try:
        data = load_data(market, symbol)
        print(f"    Loading data... {len(data)} bars")
        
        print(f"    Computing full SI...")
        si = compute_full_si(data)
        print(f"    SI computed: mean={si.mean():.3f}")
        
        # Run all tests
        print(f"    Test 1: Drawdown prediction...")
        dd_result = test_drawdown_prediction(data, si)
        
        print(f"    Test 2: Volatility forecasting...")
        vol_result = test_volatility_forecasting(data, si)
        
        print(f"    Test 3: Position scaling...")
        scale_result = test_position_scaling(data, si, market)
        
        print(f"    Test 4: Tail risk...")
        tail_result = test_tail_risk(data, si)
        
        # Summary
        result = {
            'market': market,
            'symbol': symbol,
            'drawdown_prediction': dd_result,
            'volatility_forecasting': vol_result,
            'position_scaling': scale_result,
            'tail_risk': tail_result,
            
            # Key findings
            'si_predicts_drawdowns': dd_result.get('low_si_predicts_worse_dd', False),
            'si_predicts_volatility': vol_result.get('any_predictive', False),
            'si_improves_sizing': scale_result.get('si_improves_sharpe', False),
            'si_warns_tail_risk': tail_result.get('low_si_more_risky', False),
        }
        
        # Print summary
        checks = [
            result['si_predicts_drawdowns'],
            result['si_predicts_volatility'],
            result['si_improves_sizing'],
            result['si_warns_tail_risk'],
        ]
        n_passed = sum(checks)
        print(f"    ✅ {n_passed}/4 risk indicator tests passed")
        
        return result
        
    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'market': market, 'symbol': symbol, 'error': str(e)}


def main():
    print("\n" + "="*70)
    print("SI AS RISK INDICATOR - DEEP ANALYSIS")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("\nTests:")
    print("  1. Drawdown prediction (does low SI precede losses?)")
    print("  2. Volatility forecasting (does SI predict future vol?)")
    print("  3. Position scaling (scale size by SI)")
    print("  4. Tail risk warning (does low SI = higher tail risk?)")
    print("="*70)
    
    assets = get_all_assets()
    print(f"\nAnalyzing {len(assets)} assets...")
    
    results = []
    for market, symbol in assets:
        result = analyze_asset(market, symbol)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: SI AS RISK INDICATOR")
    print("="*70)
    
    valid = [r for r in results if 'error' not in r]
    
    # Count passes for each test
    dd_pass = sum(1 for r in valid if r.get('si_predicts_drawdowns', False))
    vol_pass = sum(1 for r in valid if r.get('si_predicts_volatility', False))
    scale_pass = sum(1 for r in valid if r.get('si_improves_sizing', False))
    tail_pass = sum(1 for r in valid if r.get('si_warns_tail_risk', False))
    
    print(f"\n  Assets analyzed: {len(valid)}")
    print(f"\n  Test Results:")
    print(f"    1. SI predicts drawdowns:     {dd_pass}/{len(valid)} ({dd_pass/len(valid)*100:.0f}%)")
    print(f"    2. SI predicts volatility:    {vol_pass}/{len(valid)} ({vol_pass/len(valid)*100:.0f}%)")
    print(f"    3. SI improves sizing:        {scale_pass}/{len(valid)} ({scale_pass/len(valid)*100:.0f}%)")
    print(f"    4. Low SI = higher tail risk: {tail_pass}/{len(valid)} ({tail_pass/len(valid)*100:.0f}%)")
    
    # Best use case
    best_use = max([
        ('drawdown_warning', dd_pass),
        ('volatility_forecast', vol_pass),
        ('position_scaling', scale_pass),
        ('tail_risk_warning', tail_pass),
    ], key=lambda x: x[1])
    
    print(f"\n  Best use case: {best_use[0]} ({best_use[1]}/{len(valid)} assets)")
    
    # Recommendation
    print("\n  " + "-"*50)
    print("  RECOMMENDATION:")
    
    if best_use[1] >= len(valid) * 0.5:
        print(f"  ✅ SI is useful as a {best_use[0].replace('_', ' ')}")
        print(f"     This supports framing SI as a RISK INDICATOR")
    else:
        print(f"  ⚠️ SI shows weak risk indicator properties")
        print(f"     Focus paper on MECHANISM, not practical utility")
    
    # Save results
    output_path = Path('results/si_risk_indicator/full_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_assets': len(valid),
        'test_results': {
            'drawdown_prediction': {'passed': dd_pass, 'total': len(valid)},
            'volatility_forecasting': {'passed': vol_pass, 'total': len(valid)},
            'position_scaling': {'passed': scale_pass, 'total': len(valid)},
            'tail_risk_warning': {'passed': tail_pass, 'total': len(valid)},
        },
        'best_use_case': best_use[0],
        'asset_results': results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n  Results saved to {output_path}")
    print("="*70 + "\n")
    
    return summary


if __name__ == "__main__":
    results = main()
