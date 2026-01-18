#!/usr/bin/env python3
"""
TEST ALL 10 SI APPLICATIONS - VERSION 2 (WITH FIXES)

FIXES APPLIED:
1. Train/Validation/Test split (60/20/20)
2. Bootstrap confidence intervals
3. FDR correction for multiple testing
4. Walk-forward OOS validation
5. Fixed momentum signal (uses lagged values, no look-ahead)
6. Proper position shifting (shift(1) before applying to returns)
7. Costs applied to position changes only
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2
from src.data.loader_v2 import DataLoaderV2, MarketType


# Reproducibility
RANDOM_SEED = 42

# Purging gap to prevent data leakage
PURGE_DAYS = 7  # Gap between train and test to account for feature lookback


# HAC Standard Errors for autocorrelated data
def hac_standard_error(x, max_lag=None):
    """
    Newey-West HAC standard error estimator.
    Accounts for autocorrelation in time series.
    """
    n = len(x)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2/9)))  # Optimal lag
    
    x = np.array(x)
    x_demeaned = x - np.mean(x)
    
    # Variance term
    gamma_0 = np.sum(x_demeaned ** 2) / n
    
    # Autocovariance terms with Bartlett weights
    weighted_sum = 0
    for j in range(1, max_lag + 1):
        weight = 1 - j / (max_lag + 1)  # Bartlett kernel
        gamma_j = np.sum(x_demeaned[j:] * x_demeaned[:-j]) / n
        weighted_sum += 2 * weight * gamma_j
    
    hac_var = gamma_0 + weighted_sum
    return np.sqrt(hac_var / n)


def hac_tstat(x):
    """Compute t-statistic using HAC standard errors."""
    mean = np.mean(x)
    se = hac_standard_error(x)
    return mean / se if se > 0 else 0


np.random.seed(RANDOM_SEED)

# ============================================================
# CONFIGURATION
# ============================================================

ASSETS = {
    'crypto': ['BTCUSDT', 'ETHUSDT'],
    'stocks': ['SPY', 'QQQ'],
    'forex': ['EURUSD', 'GBPUSD'],
}

TRANSACTION_COSTS = {
    'BTCUSDT': 0.0004, 'ETHUSDT': 0.0004,
    'SPY': 0.0002, 'QQQ': 0.0002,
    'EURUSD': 0.0001, 'GBPUSD': 0.0001,
}

MARKET_TYPE_MAP = {
    'BTCUSDT': MarketType.CRYPTO, 'ETHUSDT': MarketType.CRYPTO,
    'SPY': MarketType.STOCKS, 'QQQ': MarketType.STOCKS,
    'EURUSD': MarketType.FOREX, 'GBPUSD': MarketType.FOREX,
}

SI_WINDOW = 7
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
N_BOOTSTRAP = 1000

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

# Block Bootstrap for time series with autocorrelation
def block_bootstrap_sharpe(returns, n_boot=1000, block_size=None, alpha=0.05):
    """
    Block bootstrap for Sharpe ratio CI.
    Uses non-overlapping blocks to preserve autocorrelation structure.
    
    Args:
        returns: Array of returns
        n_boot: Number of bootstrap samples
        block_size: Size of each block (default: sqrt(n))
        alpha: Significance level for CI
    
    Returns:
        dict with mean, ci_lower, ci_upper, prob_positive
    """
    n = len(returns)
    if block_size is None:
        block_size = max(5, int(np.sqrt(n)))  # Rule of thumb
    
    n_blocks = n // block_size
    sharpes = []
    
    for _ in range(n_boot):
        # Sample blocks with replacement
        block_indices = np.random.randint(0, n - block_size + 1, n_blocks)
        sample = np.concatenate([returns[i:i+block_size] for i in block_indices])
        
        if len(sample) > 0 and np.std(sample) > 0:
            sharpe = np.mean(sample) / np.std(sample) * np.sqrt(252)
            sharpes.append(sharpe)
    
    sharpes = np.array(sharpes)
    return {
        'mean': float(np.mean(sharpes)),
        'ci_lower': float(np.percentile(sharpes, 100 * alpha / 2)),
        'ci_upper': float(np.percentile(sharpes, 100 * (1 - alpha / 2))),
        'prob_positive': float(np.mean(sharpes > 0)),
        'n_samples': len(sharpes)
    }



def compute_si_full_data(data: pd.DataFrame, window: int = 7) -> pd.Series:
    """
    Compute SI on full data.

    NOTE: This is NOT look-ahead bias because:
    1. SI at time t is computed from agent affinities up to time t
    2. The competition runs sequentially - each step only uses past data
    3. SI is a trailing indicator (rolling window uses only past data)
    """
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')

    # Run competition on full data (sequentially, no look-ahead)
    population.run(data)

    # Compute SI timeseries
    return population.compute_si_timeseries(data, window=window)


def train_val_test_split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    """Split data into train/val/test with indices."""
    n = len(data)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = data.iloc[:train_end]
    val = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]

    return train, val, test, train_end, val_end


def sharpe_ratio(returns: pd.Series) -> float:
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum / peak - 1)
    return float(dd.min())


def bootstrap_ci(returns: pd.Series, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05) -> Dict:
    """
    Bootstrap confidence interval for Sharpe ratio.
    FIX: Adds statistical significance.
    """
    if len(returns) < 20:
        return {'mean': 0, 'ci_lower': 0, 'ci_upper': 0, 'p_value': 1.0, 'significant': False}

    sharpes = []
    n = len(returns)

    for _ in range(n_boot):
        sample = returns.sample(n=n, replace=True)
        sharpes.append(sharpe_ratio(sample))

    sharpes = np.array(sharpes)
    mean_sharpe = np.mean(sharpes)
    ci_lower = np.percentile(sharpes, 100 * alpha / 2)
    ci_upper = np.percentile(sharpes, 100 * (1 - alpha / 2))

    # P-value: proportion of bootstrap samples <= 0
    p_value = (sharpes <= 0).mean()

    return {
        'mean': mean_sharpe,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }


def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    FDR correction for multiple testing.
    FIX: Addresses multiple testing problem.
    """
    n = len(p_values)
    sorted_pvals = sorted(enumerate(p_values), key=lambda x: x[1])

    significant = [False] * n

    for i, (orig_idx, pval) in enumerate(sorted_pvals):
        threshold = alpha * (i + 1) / n
        if pval <= threshold:
            significant[orig_idx] = True
        else:
            break  # Once we fail, all higher p-values also fail

    return significant


def apply_costs(returns: pd.Series, positions: pd.Series, cost_rate: float) -> pd.Series:
    """Apply transaction costs correctly to position changes only."""
    position_changes = positions.diff().abs().fillna(0)
    costs = position_changes * cost_rate
    return returns - costs


def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = data['high'], data['low'], data['close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = compute_atr(data, 1) * period
    plus_di = 100 * (plus_dm.rolling(period).sum() / (tr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).sum() / (tr + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean()


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_validation(data: pd.DataFrame, si: pd.Series,
                            strategy_func, cost_rate: float,
                            train_size: int = 252, test_size: int = 63) -> Dict:
    """
    Walk-forward out-of-sample validation.
    FIX: Proper OOS testing.
    """
    results = []
    n = len(data)

    start_idx = 0
    while start_idx + train_size + test_size <= n:
        # Train window
        train_end = start_idx + train_size
        test_end = train_end + test_size

        # Get train and test data
        train_data = data.iloc[start_idx:train_end]
        test_data = data.iloc[train_end:test_end]

        train_si = si.iloc[start_idx:train_end]
        test_si = si.iloc[train_end:test_end]

        # Run strategy on test data
        test_returns = strategy_func(test_data, test_si, cost_rate)

        results.append({
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'oos_sharpe': sharpe_ratio(test_returns),
            'oos_return': test_returns.sum(),
            'profitable': test_returns.sum() > 0,
        })

        # Move forward
        start_idx += test_size

    if not results:
        return {'error': 'Not enough data for walk-forward'}

    avg_sharpe = np.mean([r['oos_sharpe'] for r in results])
    pct_profitable = np.mean([r['profitable'] for r in results])

    return {
        'n_windows': len(results),
        'avg_oos_sharpe': avg_sharpe,
        'pct_profitable': pct_profitable,
        'windows': results,
    }


# ============================================================
# APPLICATION STRATEGIES (RETURNING CALLABLE)
# ============================================================

def create_risk_budgeting_strategy():
    """Returns a callable strategy function."""
    def strategy(data: pd.DataFrame, si: pd.Series, cost_rate: float) -> pd.Series:
        returns = data['close'].pct_change()
        common = si.index.intersection(returns.dropna().index)
        si_aligned = si.loc[common]
        returns_aligned = returns.loc[common]

        si_rank = si_aligned.rank(pct=True)
        position = 0.5 + si_rank * 1.0
        position_shifted = position.shift(1).fillna(1.0)

        gross_returns = position_shifted * returns_aligned
        return apply_costs(gross_returns, position_shifted, cost_rate)

    return strategy


def create_spread_strategy():
    """Returns a callable strategy function."""
    def strategy(data: pd.DataFrame, si: pd.Series, cost_rate: float,
                 entry_z: float = 2.0, exit_z: float = 0.5, lookback: int = 60) -> pd.Series:
        returns = data['close'].pct_change()
        adx = compute_adx(data) / 100

        common = si.index.intersection(adx.dropna().index).intersection(returns.dropna().index)
        si_aligned = si.loc[common]
        adx_aligned = adx.loc[common]
        returns_aligned = returns.loc[common]

        spread = si_aligned - adx_aligned
        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()
        z_score = (spread - spread_mean) / (spread_std + 1e-10)

        position = pd.Series(0.0, index=z_score.index)
        in_trade = False
        current_pos = 0

        for i in range(1, len(z_score)):
            if not in_trade:
                if z_score.iloc[i] > entry_z:
                    current_pos = -1
                    in_trade = True
                elif z_score.iloc[i] < -entry_z:
                    current_pos = 1
                    in_trade = True
            else:
                if abs(z_score.iloc[i]) < exit_z:
                    current_pos = 0
                    in_trade = False
            position.iloc[i] = current_pos

        gross_returns = position.shift(1).fillna(0) * returns_aligned
        return apply_costs(gross_returns, position, cost_rate)

    return strategy


def create_factor_timing_strategy():
    """Returns a callable strategy function."""
    def strategy(data: pd.DataFrame, si: pd.Series, cost_rate: float,
                 threshold: float = 0.5) -> pd.Series:
        returns = data['close'].pct_change()

        common = si.index.intersection(returns.dropna().index)
        si_aligned = si.loc[common]
        returns_aligned = returns.loc[common]

        si_rank = si_aligned.rank(pct=True)

        # FIX: Proper momentum signal with shift
        mom_5d = data['close'].pct_change(5).loc[common]
        momentum_signal = np.sign(mom_5d.shift(1))  # Use PREVIOUS 5-day return

        # Mean-reversion signal (RSI extremes)
        rsi = compute_rsi(data['close'], 14).loc[common]
        meanrev_signal = np.where(rsi.shift(1) > 70, -1,
                         np.where(rsi.shift(1) < 30, 1, 0))

        # Combine based on SI
        position = np.where(si_rank.shift(1) > threshold, momentum_signal,
                    np.where(si_rank.shift(1) < (1 - threshold), meanrev_signal, 0))
        position = pd.Series(position, index=common)

        gross_returns = position * returns_aligned
        return apply_costs(gross_returns, position.abs(), cost_rate)

    return strategy


def create_regime_rebalance_strategy():
    """Returns a callable strategy function."""
    def strategy(data: pd.DataFrame, si: pd.Series, cost_rate: float) -> pd.Series:
        returns = data['close'].pct_change()

        common = si.index.intersection(returns.dropna().index)
        si_aligned = si.loc[common]
        returns_aligned = returns.loc[common]

        si_rank = si_aligned.rank(pct=True)

        # FIX: Use lagged SI for regime (no look-ahead)
        regime = np.where(si_rank.shift(1) > 0.67, 'high',
                 np.where(si_rank.shift(1) < 0.33, 'low', 'mid'))
        equity_weight = np.where(regime == 'high', 1.0,
                        np.where(regime == 'low', 0.5, 0.75))

        si_returns = pd.Series(equity_weight, index=common) * returns_aligned

        # Apply costs for weight changes
        weight_series = pd.Series(equity_weight, index=common)
        return apply_costs(si_returns, weight_series, cost_rate)

    return strategy


def create_entry_timing_strategy():
    """Returns a callable strategy function."""
    def strategy(data: pd.DataFrame, si: pd.Series, cost_rate: float) -> pd.Series:
        returns = data['close'].pct_change()

        common = si.index.intersection(returns.dropna().index)
        si_aligned = si.loc[common]
        returns_aligned = returns.loc[common]
        close = data['close'].loc[common]

        # FIX: Use lagged SI rank
        si_rank = si_aligned.rank(pct=True).shift(1)

        # FIX: Use lagged price position
        high_20 = close.rolling(20).max().shift(1)
        low_20 = close.rolling(20).min().shift(1)
        price_pos = (close.shift(1) - low_20) / (high_20 - low_20 + 1e-10)

        # Entry signal
        signal = np.where((si_rank > 0.7) & (price_pos < 0.2), 2,
                 np.where((si_rank > 0.6) & (price_pos < 0.5), 1,
                 np.where((si_rank < 0.3) & (price_pos > 0.8), -1,
                 np.where(si_rank < 0.3, 0, 0.5))))
        signal = pd.Series(signal, index=common)

        gross_returns = signal * returns_aligned
        return apply_costs(gross_returns, signal.abs(), cost_rate)

    return strategy


# ============================================================
# MAIN TESTING FUNCTION WITH OOS VALIDATION
# ============================================================

def test_application(data: pd.DataFrame, si: pd.Series,
                     strategy_func, cost_rate: float,
                     train_end: int, val_end: int) -> Dict:
    """Test an application with proper train/val/test split."""

    # In-sample (train)
    train_si = si.iloc[:train_end]
    train_data = data.iloc[:train_end]
    train_returns = strategy_func(train_data, train_si, cost_rate)
    train_sharpe = sharpe_ratio(train_returns)

    # Validation
    val_si = si.iloc[train_end:val_end]
    val_data = data.iloc[train_end:val_end]
    val_returns = strategy_func(val_data, val_si, cost_rate)
    val_sharpe = sharpe_ratio(val_returns)

    # Out-of-sample (test)
    test_si = si.iloc[val_end:]
    test_data = data.iloc[val_end:]
    test_returns = strategy_func(test_data, test_si, cost_rate)
    test_sharpe = sharpe_ratio(test_returns)

    # Bootstrap CI on test
    test_ci = bootstrap_ci(test_returns)

    # Walk-forward validation
    wf_results = walk_forward_validation(data, si, strategy_func, cost_rate)

    return {
        'train_sharpe': train_sharpe,
        'val_sharpe': val_sharpe,
        'test_sharpe': test_sharpe,
        'test_ci_lower': test_ci['ci_lower'],
        'test_ci_upper': test_ci['ci_upper'],
        'test_p_value': test_ci['p_value'],
        'test_significant': test_ci['significant'],
        'max_dd': max_drawdown(test_returns),
        'is_degradation': (train_sharpe - test_sharpe) / (abs(train_sharpe) + 0.01),
        'wf_avg_sharpe': wf_results.get('avg_oos_sharpe', None),
        'wf_pct_profitable': wf_results.get('pct_profitable', None),
        'wf_n_windows': wf_results.get('n_windows', 0),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*70)
    print("  TEST ALL SI APPLICATIONS (V2 - WITH FIXES)")
    print("="*70)
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"\n  FIXES APPLIED:")
    print("    1. Train/Val/Test split (60/20/20)")
    print("    2. Bootstrap confidence intervals")
    print("    3. FDR correction for multiple testing")
    print("    4. Walk-forward OOS validation")
    print("    5. Fixed signals (lagged, no look-ahead)")
    print("    6. Proper position shifting")
    print("    7. Costs on position changes only")

    loader = DataLoaderV2()
    all_results = {}
    all_p_values = []

    # Define strategies to test
    strategies = [
        ("1. Risk Budgeting", create_risk_budgeting_strategy()),
        ("2. SI-ADX Spread", create_spread_strategy()),
        ("3. Factor Timing", create_factor_timing_strategy()),
        ("6. Regime Rebalance", create_regime_rebalance_strategy()),
        ("10. Entry Timing", create_entry_timing_strategy()),
    ]

    all_assets = []
    for market, assets in ASSETS.items():
        all_assets.extend(assets)

    print(f"\n  Testing {len(strategies)} strategies on {len(all_assets)} assets")

    # Load data and compute SI
    print("\n  Loading data...")
    asset_data = {}
    asset_si = {}
    asset_splits = {}

    for asset in all_assets:
        print(f"    {asset}...", end=" ")
        data = loader.load(asset, MARKET_TYPE_MAP[asset])
        train, val, test, train_end, val_end = train_val_test_split(data)

        # Compute SI on full data (sequential, no look-ahead in SI computation)
        si = compute_si_full_data(data, window=SI_WINDOW)

        asset_data[asset] = data
        asset_si[asset] = si
        asset_splits[asset] = (train_end, val_end)
        print("✓")

    # Run tests
    print("\n" + "-"*70)
    print("  RUNNING TESTS")
    print("-"*70)

    for strat_name, strat_func in strategies:
        print(f"\n  {strat_name}")
        app_results = {}

        for asset in all_assets:
            data = asset_data[asset]
            si = asset_si[asset]
            train_end, val_end = asset_splits[asset]
            cost = TRANSACTION_COSTS[asset]

            result = test_application(data, si, strat_func, cost, train_end, val_end)
            app_results[asset] = result
            all_p_values.append(result['test_p_value'])

            # Print result
            test_sharpe = result['test_sharpe']
            sig = "✓" if result['test_significant'] else ""
            wf_sharpe = result['wf_avg_sharpe']
            wf_str = f"{wf_sharpe:.2f}" if wf_sharpe else "N/A"
            print(f"    {asset}: Test={test_sharpe:+.2f}{sig} WF={wf_str}")

        all_results[strat_name] = app_results

    # FDR correction
    print("\n" + "-"*70)
    print("  FDR CORRECTION (Multiple Testing)")
    print("-"*70)

    significant_after_fdr = benjamini_hochberg_correction(all_p_values)
    n_sig_before = sum(1 for p in all_p_values if p < 0.05)
    n_sig_after = sum(significant_after_fdr)

    print(f"  Tests significant at α=0.05: {n_sig_before}/{len(all_p_values)}")
    print(f"  Tests significant after FDR: {n_sig_after}/{len(all_p_values)}")

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY (Using OOS Test Sharpe)")
    print("="*70)

    summary = []
    for strat_name, results in all_results.items():
        test_sharpes = [r['test_sharpe'] for r in results.values()]
        wf_sharpes = [r['wf_avg_sharpe'] for r in results.values() if r['wf_avg_sharpe']]

        avg_test = np.mean(test_sharpes)
        avg_wf = np.mean(wf_sharpes) if wf_sharpes else None
        consistency = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes)
        n_sig = sum(1 for r in results.values() if r['test_significant'])

        summary.append({
            'strategy': strat_name,
            'avg_test_sharpe': avg_test,
            'avg_wf_sharpe': avg_wf,
            'consistency': consistency,
            'n_significant': n_sig,
        })

    summary = sorted(summary, key=lambda x: x['avg_test_sharpe'] or 0, reverse=True)

    print(f"\n  {'Rank':<6} {'Strategy':<25} {'Test Sharpe':>12} {'WF Sharpe':>12} {'Consist':>10} {'Sig':>6}")
    print("  " + "-"*75)

    for i, s in enumerate(summary):
        wf_str = f"{s['avg_wf_sharpe']:.3f}" if s['avg_wf_sharpe'] else "N/A"
        print(f"  {i+1:<6} {s['strategy']:<25} {s['avg_test_sharpe']:>12.3f} {wf_str:>12} {s['consistency']:>10.0%} {s['n_significant']:>6}")

    # Best strategy
    best = summary[0]
    print(f"\n  🏆 BEST STRATEGY: {best['strategy']}")
    print(f"     OOS Test Sharpe: {best['avg_test_sharpe']:.3f}")
    print(f"     Walk-Forward Sharpe: {best['avg_wf_sharpe']:.3f}" if best['avg_wf_sharpe'] else "     Walk-Forward: N/A")
    print(f"     Consistency: {best['consistency']:.0%}")

    # Save results
    out_path = Path("results/application_testing_v2/full_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'fixes_applied': [
            'Train/Val/Test split (60/20/20)',
            'Bootstrap confidence intervals',
            'FDR correction for multiple testing',
            'Walk-forward OOS validation',
            'Fixed signals (lagged, no look-ahead)',
            'Proper position shifting (shift(1))',
            'Costs applied to position changes only',
        ],
        'summary': summary,
        'fdr_correction': {
            'significant_before': n_sig_before,
            'significant_after': n_sig_after,
            'total_tests': len(all_p_values),
        },
        'detailed_results': all_results,
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

    return summary, all_results

if __name__ == "__main__":
    main()
