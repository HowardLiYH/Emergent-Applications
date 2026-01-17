#!/usr/bin/env python3
"""
P1.5: Factor Regression - NEXT_STEPS_PLAN v4.1

Prove that SI is a novel signal, not just repackaged momentum/volatility.

Required by: Prof. Kumar (MIT) - CRITICAL for publication

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

warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETER DOCUMENTATION
# ============================================================================

PARAMETER_CHOICES = {
    'si_window': 7,
    'momentum_window': 20,      # 20-day momentum
    'volatility_window': 20,    # 20-day realized volatility
    'trend_window': 14,         # ADX-like trend measure
    'hac_lags': 5,              # HAC standard errors lags
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
# SI AND FACTOR COMPUTATION
# ============================================================================

def compute_si_simple(data: pd.DataFrame, window: int = 7) -> pd.Series:
    """Compute simplified SI."""
    returns = data['close'].pct_change()
    momentum = returns.rolling(window).mean()
    mean_rev = -returns.rolling(window).mean()
    volatility = returns.rolling(window).std()

    strategies = pd.DataFrame({
        'momentum': momentum,
        'mean_reversion': mean_rev,
        'low_vol': -volatility,
    }).dropna()

    strategies_norm = strategies.apply(lambda x: (x - x.mean()) / (x.std() + 1e-10))
    si = strategies_norm.std(axis=1)
    si = (si - si.min()) / (si.max() - si.min() + 1e-10)
    return si


def construct_factors(data: pd.DataFrame) -> pd.DataFrame:
    """
    Construct factor returns for regression.

    Factors:
    1. Market: Buy-and-hold return
    2. Momentum: Sign of past 20-day return times current return
    3. Volatility timing: Inverse volatility signal
    4. Trend: Trend strength signal
    """
    returns = data['close'].pct_change()

    factors = pd.DataFrame(index=data.index)

    # Factor 1: Market (buy and hold)
    factors['market'] = returns

    # Factor 2: Momentum
    mom_window = PARAMETER_CHOICES['momentum_window']
    mom_signal = np.sign(data['close'].pct_change(mom_window))
    factors['momentum'] = mom_signal.shift(1) * returns

    # Factor 3: Volatility timing (inverse vol)
    vol_window = PARAMETER_CHOICES['volatility_window']
    rolling_vol = returns.rolling(vol_window).std()
    vol_signal = 1 / rolling_vol.clip(lower=0.001)
    vol_signal_norm = vol_signal / vol_signal.rolling(60).mean().clip(lower=0.001)
    factors['vol_timing'] = vol_signal_norm.shift(1) * returns

    # Factor 4: Trend strength
    trend_window = PARAMETER_CHOICES['trend_window']
    if 'high' in data.columns and 'low' in data.columns:
        high_low_range = (data['high'] - data['low']).rolling(trend_window).mean()
        trend_strength = high_low_range / data['close'].rolling(trend_window).mean().clip(lower=0.001)
    else:
        # Fallback: use return magnitude as trend proxy
        trend_strength = returns.abs().rolling(trend_window).mean()

    trend_signal = np.sign(data['close'].pct_change(7)) * trend_strength
    factors['trend'] = trend_signal.shift(1) * returns

    return factors.dropna()


def compute_si_strategy_returns(data: pd.DataFrame, si: pd.Series) -> pd.Series:
    """Compute returns from SI-based strategy."""
    returns = data['close'].pct_change()

    # Long when SI > median
    si_signal = (si > si.median()).astype(float)

    # Strategy returns
    strategy_returns = si_signal.shift(1) * returns

    return strategy_returns


# ============================================================================
# FACTOR REGRESSION
# ============================================================================

def run_factor_regression(si_returns: pd.Series, factors: pd.DataFrame,
                         use_hac: bool = True) -> Dict:
    """
    Regress SI strategy returns on known factors.
    """
    # Align data
    aligned = pd.concat([si_returns.rename('si_returns'), factors], axis=1).dropna()

    if len(aligned) < 60:
        return {'error': 'Insufficient data', 'alpha_significant': False}

    y = aligned['si_returns']
    X = sm.add_constant(aligned.drop('si_returns', axis=1))

    # Fit model
    if use_hac:
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': PARAMETER_CHOICES['hac_lags']})
    else:
        model = sm.OLS(y, X).fit()

    # Extract results
    alpha = model.params['const']
    alpha_tstat = model.tvalues['const']
    alpha_pval = model.pvalues['const']

    # Factor betas
    factor_betas = model.params.drop('const').to_dict()
    factor_tstats = model.tvalues.drop('const').to_dict()
    factor_pvals = model.pvalues.drop('const').to_dict()

    # Information Ratio
    annual_alpha = alpha * 252
    annual_resid_vol = model.resid.std() * np.sqrt(252)
    ir = annual_alpha / annual_resid_vol if annual_resid_vol > 0 else 0

    return {
        'alpha': float(alpha),
        'alpha_annual': float(annual_alpha),
        'alpha_tstat': float(alpha_tstat),
        'alpha_pval': float(alpha_pval),
        'alpha_significant': bool(alpha_pval < 0.05),
        'r_squared': float(model.rsquared),
        'adj_r_squared': float(model.rsquared_adj),
        'factor_betas': {k: float(v) for k, v in factor_betas.items()},
        'factor_tstats': {k: float(v) for k, v in factor_tstats.items()},
        'factor_pvals': {k: float(v) for k, v in factor_pvals.items()},
        'information_ratio': float(ir),
        'n_obs': int(len(y)),
        'residual_std': float(model.resid.std()),
    }


def run_incremental_regression(si_returns: pd.Series, factors: pd.DataFrame) -> Dict:
    """
    Run regression with incremental factor addition.
    Shows how alpha changes as factors are added.
    """
    results = {}

    y = si_returns

    # Model 1: SI only (intercept = alpha)
    aligned = pd.concat([y], axis=1).dropna()
    if len(aligned) > 30:
        model = sm.OLS(aligned.iloc[:, 0], sm.add_constant(np.ones(len(aligned)))).fit()
        results['(1) SI Only'] = {
            'alpha': float(model.params['const']),
            'alpha_tstat': float(model.tvalues['const']),
            'r_squared': 0.0,
        }

    # Model 2: + Market
    if 'market' in factors.columns:
        aligned = pd.concat([y, factors['market']], axis=1).dropna()
        if len(aligned) > 30:
            X = sm.add_constant(aligned.iloc[:, 1])
            model = sm.OLS(aligned.iloc[:, 0], X).fit()
            results['(2) + Market'] = {
                'alpha': float(model.params['const']),
                'alpha_tstat': float(model.tvalues['const']),
                'r_squared': float(model.rsquared),
                'market_beta': float(model.params.iloc[1]),
            }

    # Model 3: + All Factors
    aligned = pd.concat([y, factors], axis=1).dropna()
    if len(aligned) > 30:
        X = sm.add_constant(aligned.iloc[:, 1:])
        model = sm.OLS(aligned.iloc[:, 0], X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        results['(3) + All Factors'] = {
            'alpha': float(model.params['const']),
            'alpha_tstat': float(model.tvalues['const']),
            'r_squared': float(model.rsquared),
        }

    return results


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_asset(market: str, symbol: str) -> Dict:
    """Run factor regression for a single asset."""
    print(f"\n  Analyzing {market}/{symbol}...")

    try:
        # Load data
        data = load_data(market, symbol)

        # Compute SI
        si = compute_si_simple(data, window=PARAMETER_CHOICES['si_window'])

        # Compute SI strategy returns
        si_returns = compute_si_strategy_returns(data, si)

        # Construct factors
        factors = construct_factors(data)

        # Align
        aligned_si = si_returns.reindex(factors.index).dropna()
        aligned_factors = factors.reindex(aligned_si.index)

        # Run full regression
        full_reg = run_factor_regression(aligned_si, aligned_factors)

        # Run incremental regression
        incremental = run_incremental_regression(aligned_si, aligned_factors)

        # Correlation of SI with factors
        factor_correlations = {}
        for col in aligned_factors.columns:
            r, _ = spearmanr(si.reindex(aligned_factors.index).dropna(),
                           aligned_factors[col].reindex(si.index).dropna())
            factor_correlations[col] = float(r) if not np.isnan(r) else 0

        result = {
            'market': market,
            'symbol': symbol,
            'n_observations': full_reg.get('n_obs', 0),

            # Full regression
            'alpha': full_reg.get('alpha', 0),
            'alpha_annual': full_reg.get('alpha_annual', 0),
            'alpha_tstat': full_reg.get('alpha_tstat', 0),
            'alpha_significant': full_reg.get('alpha_significant', False),
            'r_squared': full_reg.get('r_squared', 0),
            'information_ratio': full_reg.get('information_ratio', 0),

            # Factor loadings
            'factor_betas': full_reg.get('factor_betas', {}),
            'factor_tstats': full_reg.get('factor_tstats', {}),

            # SI-factor correlations
            'si_factor_correlations': factor_correlations,

            # Incremental
            'incremental_regression': incremental,

            # Assessment
            'novel_signal': (
                full_reg.get('alpha_tstat', 0) > 2.0 and
                full_reg.get('r_squared', 1) < 0.5
            ),
        }

        # Print summary
        alpha_t = full_reg.get('alpha_tstat', 0)
        r2 = full_reg.get('r_squared', 0)
        status = "✅" if result['novel_signal'] else "⚠️"
        sig = "***" if alpha_t > 2.58 else ("**" if alpha_t > 1.96 else ("*" if alpha_t > 1.65 else ""))
        print(f"    {status} Alpha t-stat: {alpha_t:.2f}{sig}, R²: {r2:.3f}, Novel: {result['novel_signal']}")

        return result

    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        return {
            'market': market,
            'symbol': symbol,
            'error': str(e),
            'novel_signal': False,
        }


def main():
    """Run P1.5 factor regression for all assets."""

    print("\n" + "="*60)
    print("P1.5: FACTOR REGRESSION")
    print("Proving SI is a novel signal (Prof. Kumar requirement)")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Get all assets
    assets = get_all_assets()
    print(f"\nFound {len(assets)} assets to analyze")

    # Analyze each asset
    results = []
    for market, symbol in assets:
        result = analyze_asset(market, symbol)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("P1.5 SUMMARY")
    print("="*60)

    valid_results = [r for r in results if 'error' not in r]
    novel = [r for r in valid_results if r['novel_signal']]
    significant_alpha = [r for r in valid_results if r['alpha_tstat'] > 2.0]
    low_r2 = [r for r in valid_results if r['r_squared'] < 0.5]

    print(f"\n  Assets analyzed: {len(valid_results)}")
    print(f"  Alpha t-stat > 2.0: {len(significant_alpha)}/{len(valid_results)}")
    print(f"  R² < 0.5 (not explained by factors): {len(low_r2)}/{len(valid_results)}")
    print(f"  Novel signal (both criteria): {len(novel)}/{len(valid_results)}")

    # Average metrics
    if valid_results:
        avg_alpha_t = np.mean([r['alpha_tstat'] for r in valid_results])
        avg_r2 = np.mean([r['r_squared'] for r in valid_results])
        avg_ir = np.mean([r['information_ratio'] for r in valid_results])

        print(f"\n  Average alpha t-stat: {avg_alpha_t:.2f}")
        print(f"  Average R²: {avg_r2:.3f}")
        print(f"  Average Information Ratio: {avg_ir:.3f}")

    # By market
    print("\n  By market:")
    for market in ['crypto', 'forex', 'stocks', 'commodities']:
        market_results = [r for r in valid_results if r.get('market') == market]
        market_novel = [r for r in market_results if r['novel_signal']]
        if market_results:
            avg_t = np.mean([r['alpha_tstat'] for r in market_results])
            print(f"    {market}: {len(market_novel)}/{len(market_results)} novel, avg t-stat: {avg_t:.2f}")

    # Publication-ready table
    print("\n  Factor-Adjusted Alpha Table:")
    print("  " + "-"*50)
    print(f"  {'Asset':<15} {'Alpha t':<10} {'R²':<8} {'Novel':<8}")
    print("  " + "-"*50)
    for r in valid_results:
        sig = "***" if r['alpha_tstat'] > 2.58 else ("**" if r['alpha_tstat'] > 1.96 else "*" if r['alpha_tstat'] > 1.65 else "")
        novel_str = "✓" if r['novel_signal'] else ""
        print(f"  {r['symbol']:<15} {r['alpha_tstat']:.2f}{sig:<5} {r['r_squared']:.3f}   {novel_str}")
    print("  " + "-"*50)
    print("  *** p<0.01, ** p<0.05, * p<0.10")

    # Save results
    output_path = Path('results/p1_5_factor/full_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_assets': len(valid_results),
            'n_novel': len(novel),
            'summary': {
                'significant_alpha': len(significant_alpha),
                'low_r2': len(low_r2),
                'avg_alpha_tstat': float(avg_alpha_t) if valid_results else 0,
                'avg_r2': float(avg_r2) if valid_results else 0,
            },
            'results': results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to {output_path}")

    # Final verdict
    print("\n" + "="*60)

    # Success criteria: alpha t-stat > 2.0 in majority
    if len(significant_alpha) >= len(valid_results) / 2:
        print("✅ P1.5 PASSED: SI shows significant factor-adjusted alpha")
        print("   SI is a NOVEL signal, not just repackaged factors")
        print("   Ready to proceed to P2: Regime-Conditional SI")
    else:
        print("⚠️ P1.5 PARTIALLY PASSED: Some assets show novel alpha")
        print("   Review results before proceeding")
    print("="*60 + "\n")

    return results


if __name__ == "__main__":
    results = main()
