#!/usr/bin/env python3
"""
TEST ALL 10 SI APPLICATIONS

Comprehensive testing of all SI applications to select the best for thesis.
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2
from src.data.loader_v2 import DataLoaderV2, MarketType

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

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def compute_si(data: pd.DataFrame, window: int = 7) -> pd.Series:
    """Compute SI time series."""
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    population.run(data)
    return population.compute_si_timeseries(data, window=window)

def sharpe_ratio(returns: pd.Series) -> float:
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))

def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum / peak - 1)
    return float(dd.min())

def calmar_ratio(returns: pd.Series) -> float:
    ann_ret = returns.mean() * 252
    mdd = abs(max_drawdown(returns))
    return float(ann_ret / mdd) if mdd > 0 else 0.0

def apply_costs(returns: pd.Series, positions: pd.Series, cost_rate: float) -> pd.Series:
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
    plus_di = 100 * (plus_dm.rolling(period).sum() / tr)
    minus_di = 100 * (minus_dm.rolling(period).sum() / tr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean()

# ============================================================
# APPLICATION 1: SI RISK BUDGETING
# ============================================================

def app1_risk_budgeting(data: pd.DataFrame, si: pd.Series, cost_rate: float) -> Dict:
    """Test SI Risk Budgeting strategy."""
    returns = data['close'].pct_change()

    # Align data
    common = si.index.intersection(returns.dropna().index)
    si_aligned = si.loc[common]
    returns_aligned = returns.loc[common]

    si_rank = si_aligned.rank(pct=True)

    # Baseline: constant position
    baseline_returns = returns_aligned

    # Test: linear scaling
    position = 0.5 + si_rank * 1.0  # [0.5, 1.5]
    position_shifted = position.shift(1).fillna(1.0)

    gross_returns = position_shifted * returns_aligned
    net_returns = apply_costs(gross_returns, position_shifted, cost_rate)

    return {
        'baseline_sharpe': sharpe_ratio(baseline_returns),
        'strategy_sharpe': sharpe_ratio(net_returns),
        'improvement': sharpe_ratio(net_returns) - sharpe_ratio(baseline_returns),
        'max_dd': max_drawdown(net_returns),
        'calmar': calmar_ratio(net_returns),
    }

# ============================================================
# APPLICATION 2: SI-ADX SPREAD TRADING
# ============================================================

def app2_spread_trading(data: pd.DataFrame, si: pd.Series, cost_rate: float,
                        entry_z: float = 2.0, exit_z: float = 0.5, lookback: int = 60) -> Dict:
    """Test SI-ADX spread trading."""
    returns = data['close'].pct_change()
    adx = compute_adx(data) / 100  # Normalize to 0-1

    # Align
    common = si.index.intersection(adx.dropna().index).intersection(returns.dropna().index)
    si_aligned = si.loc[common]
    adx_aligned = adx.loc[common]
    returns_aligned = returns.loc[common]

    # Spread
    spread = si_aligned - adx_aligned
    spread_mean = spread.rolling(lookback).mean()
    spread_std = spread.rolling(lookback).std()
    z_score = (spread - spread_mean) / (spread_std + 1e-10)

    # Generate positions
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
    net_returns = apply_costs(gross_returns, position, cost_rate)

    n_trades = (position.diff() != 0).sum()

    return {
        'strategy_sharpe': sharpe_ratio(net_returns),
        'max_dd': max_drawdown(net_returns),
        'n_trades': int(n_trades),
        'win_rate': float((net_returns > 0).mean()),
    }

# ============================================================
# APPLICATION 3: FACTOR TIMING
# ============================================================

def app3_factor_timing(data: pd.DataFrame, si: pd.Series, cost_rate: float,
                       threshold: float = 0.5) -> Dict:
    """Test factor timing (momentum vs mean-reversion)."""
    returns = data['close'].pct_change()

    # Align
    common = si.index.intersection(returns.dropna().index)
    si_aligned = si.loc[common]
    returns_aligned = returns.loc[common]

    si_rank = si_aligned.rank(pct=True)

    # Momentum signal
    mom_signal = np.sign(data['close'].pct_change(5).loc[common])

    # Mean-reversion signal (RSI)
    rsi = compute_rsi(data['close'], 14).loc[common]
    meanrev_signal = np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0))

    # Factor timing
    position = np.where(si_rank > threshold, mom_signal,
                np.where(si_rank < (1 - threshold), meanrev_signal, 0))
    position = pd.Series(position, index=common)

    # Baseline: always momentum
    baseline_returns = mom_signal.shift(1) * returns_aligned

    # Strategy
    gross_returns = position.shift(1).fillna(0) * returns_aligned
    net_returns = apply_costs(gross_returns, position, cost_rate)

    return {
        'baseline_sharpe': sharpe_ratio(baseline_returns),
        'strategy_sharpe': sharpe_ratio(net_returns),
        'improvement': sharpe_ratio(net_returns) - sharpe_ratio(baseline_returns),
        'max_dd': max_drawdown(net_returns),
    }

# ============================================================
# APPLICATION 4: VOLATILITY FORECASTING
# ============================================================

def app4_vol_forecasting(data: pd.DataFrame, si: pd.Series) -> Dict:
    """Test SI for volatility forecasting."""
    returns = data['close'].pct_change()
    realized_vol = returns.rolling(5).std()
    future_vol = realized_vol.shift(-1)

    # Align
    common = si.index.intersection(realized_vol.dropna().index).intersection(future_vol.dropna().index)
    si_aligned = si.loc[common]
    vol_aligned = realized_vol.loc[common]
    future_vol_aligned = future_vol.loc[common]

    # Split
    split = int(len(common) * 0.7)

    # Baseline: EWMA
    ewma_pred = vol_aligned
    rmse_baseline = np.sqrt(((ewma_pred.iloc[split:] - future_vol_aligned.iloc[split:])**2).mean())

    # SI-enhanced (simple regression)
    from sklearn.linear_model import LinearRegression
    X = pd.DataFrame({'si': si_aligned, 'vol': vol_aligned})
    y = future_vol_aligned

    model = LinearRegression()
    model.fit(X.iloc[:split], y.iloc[:split])
    pred = model.predict(X.iloc[split:])
    rmse_si = np.sqrt(((pred - y.iloc[split:])**2).mean())

    return {
        'rmse_baseline': float(rmse_baseline),
        'rmse_si': float(rmse_si),
        'improvement_pct': float((rmse_baseline - rmse_si) / rmse_baseline * 100),
        'si_coef': float(model.coef_[0]),
    }

# ============================================================
# APPLICATION 5: DYNAMIC STOP-LOSS
# ============================================================

def app5_dynamic_stop(data: pd.DataFrame, si: pd.Series, cost_rate: float) -> Dict:
    """Test dynamic stop-loss based on SI."""
    returns = data['close'].pct_change()
    atr = compute_atr(data, 14)

    # Align
    common = si.index.intersection(atr.dropna().index).intersection(returns.dropna().index)
    si_aligned = si.loc[common]
    atr_aligned = atr.loc[common]
    returns_aligned = returns.loc[common]

    si_rank = si_aligned.rank(pct=True)

    # Base strategy: always long
    base_position = pd.Series(1.0, index=common)

    # Stop multiplier
    multiplier = np.where(si_rank > 0.7, 1.5,
                 np.where(si_rank < 0.3, 3.0, 2.0))
    stop_distance = atr_aligned * multiplier

    # Apply stops
    position = base_position.copy()
    for i in range(1, len(common)):
        if position.iloc[i-1] != 0:
            price_change = returns_aligned.iloc[i]
            if price_change < -stop_distance.iloc[i] / data['close'].loc[common].iloc[i]:
                position.iloc[i] = 0

    # Fixed stop baseline
    fixed_stop = atr_aligned * 2.0
    baseline_position = base_position.copy()
    for i in range(1, len(common)):
        if baseline_position.iloc[i-1] != 0:
            price_change = returns_aligned.iloc[i]
            if price_change < -fixed_stop.iloc[i] / data['close'].loc[common].iloc[i]:
                baseline_position.iloc[i] = 0

    baseline_returns = baseline_position.shift(1).fillna(1) * returns_aligned
    strategy_returns = position.shift(1).fillna(1) * returns_aligned

    return {
        'baseline_sharpe': sharpe_ratio(baseline_returns),
        'strategy_sharpe': sharpe_ratio(strategy_returns),
        'improvement': sharpe_ratio(strategy_returns) - sharpe_ratio(baseline_returns),
        'stops_triggered_baseline': int((baseline_position == 0).sum()),
        'stops_triggered_dynamic': int((position == 0).sum()),
    }

# ============================================================
# APPLICATION 6: REGIME REBALANCING
# ============================================================

def app6_regime_rebalancing(data: pd.DataFrame, si: pd.Series, cost_rate: float) -> Dict:
    """Test regime-based rebalancing."""
    returns = data['close'].pct_change()

    # Align
    common = si.index.intersection(returns.dropna().index)
    si_aligned = si.loc[common]
    returns_aligned = returns.loc[common]

    si_rank = si_aligned.rank(pct=True)

    # Regime-based allocation (simulating equity/cash)
    regime = np.where(si_rank > 0.67, 'high',
             np.where(si_rank < 0.33, 'low', 'mid'))
    equity_weight = np.where(regime == 'high', 1.0,
                    np.where(regime == 'low', 0.5, 0.75))

    # SI-based returns (equity weight * returns)
    si_returns = pd.Series(equity_weight, index=common).shift(1).fillna(1) * returns_aligned

    # Baseline: constant 100%
    baseline_returns = returns_aligned

    # Count rebalances
    rebalances = (pd.Series(regime) != pd.Series(regime).shift()).sum()

    return {
        'baseline_sharpe': sharpe_ratio(baseline_returns),
        'strategy_sharpe': sharpe_ratio(si_returns),
        'improvement': sharpe_ratio(si_returns) - sharpe_ratio(baseline_returns),
        'max_dd_baseline': max_drawdown(baseline_returns),
        'max_dd_strategy': max_drawdown(si_returns),
        'n_rebalances': int(rebalances),
    }

# ============================================================
# APPLICATION 7: TAIL RISK HEDGE
# ============================================================

def app7_tail_hedge(data: pd.DataFrame, si: pd.Series) -> Dict:
    """Test tail risk hedging based on SI."""
    returns = data['close'].pct_change()

    # Align
    common = si.index.intersection(returns.dropna().index)
    si_aligned = si.loc[common]
    returns_aligned = returns.loc[common]

    si_rank = si_aligned.rank(pct=True)

    # Hedge multiplier
    hedge_mult = np.where(si_rank < 0.1, 2.5,
                 np.where(si_rank < 0.25, 1.5,
                 np.where(si_rank > 0.75, 0.5, 1.0)))

    base_hedge = 0.05
    hedge_weight = base_hedge * hedge_mult
    equity_weight = 1 - hedge_weight

    # Hedge returns (assume -1 * equity returns for simplicity)
    hedge_returns = -returns_aligned * 0.5  # Imperfect hedge

    portfolio = pd.Series(equity_weight, index=common) * returns_aligned + \
                pd.Series(hedge_weight, index=common) * hedge_returns

    # Tail events (5th percentile)
    threshold = returns_aligned.quantile(0.05)
    tail_mask = returns_aligned < threshold

    tail_loss_unhedged = returns_aligned[tail_mask].mean()
    tail_loss_hedged = portfolio[tail_mask].mean()

    return {
        'unhedged_sharpe': sharpe_ratio(returns_aligned),
        'hedged_sharpe': sharpe_ratio(portfolio),
        'tail_loss_unhedged': float(tail_loss_unhedged),
        'tail_loss_hedged': float(tail_loss_hedged),
        'tail_protection_pct': float((tail_loss_unhedged - tail_loss_hedged) / abs(tail_loss_unhedged) * 100),
    }

# ============================================================
# APPLICATION 8: CROSS-ASSET MOMENTUM
# ============================================================

def app8_cross_asset(data1: pd.DataFrame, data2: pd.DataFrame,
                     si1: pd.Series, si2: pd.Series, cost_rate: float) -> Dict:
    """Test cross-asset SI momentum."""
    returns1 = data1['close'].pct_change()
    returns2 = data2['close'].pct_change()

    # Align all
    common = si1.index.intersection(si2.index).intersection(returns1.dropna().index).intersection(returns2.dropna().index)
    si1_aligned = si1.loc[common]
    si2_aligned = si2.loc[common]
    returns1_aligned = returns1.loc[common]
    returns2_aligned = returns2.loc[common]

    # SI spread
    si_spread = si1_aligned - si2_aligned
    spread_mean = si_spread.rolling(60).mean()
    spread_std = si_spread.rolling(60).std()
    z_score = (si_spread - spread_mean) / (spread_std + 1e-10)

    # Position: long asset2, short asset1 when z > 2
    position = np.where(z_score > 2, -1,
               np.where(z_score < -2, 1, 0))
    position = pd.Series(position, index=common)

    # Relative returns
    relative_returns = returns1_aligned - returns2_aligned
    strategy_returns = position.shift(1).fillna(0) * relative_returns
    net_returns = apply_costs(strategy_returns, position, cost_rate)

    return {
        'strategy_sharpe': sharpe_ratio(net_returns),
        'max_dd': max_drawdown(net_returns),
        'n_trades': int((position.diff() != 0).sum()),
    }

# ============================================================
# APPLICATION 9: ENSEMBLE STRATEGY
# ============================================================

def app9_ensemble(data: pd.DataFrame, si: pd.Series, cost_rate: float) -> Dict:
    """Test ensemble of SI strategies."""
    returns = data['close'].pct_change()
    adx = compute_adx(data) / 100

    # Align
    common = si.index.intersection(adx.dropna().index).intersection(returns.dropna().index)
    si_aligned = si.loc[common]
    returns_aligned = returns.loc[common]

    si_rank = si_aligned.rank(pct=True)

    # Signal 1: Risk budgeting
    sig1 = 0.5 + si_rank * 1.0

    # Signal 2: Momentum when high SI
    mom = np.sign(data['close'].pct_change(5).loc[common])
    sig2 = np.where(si_rank > 0.5, mom, 0)

    # Signal 3: Mean reversion when low SI
    rsi = compute_rsi(data['close'], 14).loc[common]
    sig3 = np.where(si_rank < 0.5, np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0)), 0)

    signals = pd.DataFrame({'rb': sig1, 'mom': sig2, 'mr': sig3}, index=common)

    # Equal weight ensemble
    ensemble = signals.mean(axis=1)

    # Calculate returns
    ensemble_returns = ensemble.shift(1).fillna(0) * returns_aligned
    net_returns = apply_costs(ensemble_returns, ensemble.abs(), cost_rate)

    # SI-only baseline
    si_only_returns = sig1.shift(1).fillna(1) * returns_aligned

    return {
        'si_only_sharpe': sharpe_ratio(si_only_returns),
        'ensemble_sharpe': sharpe_ratio(net_returns),
        'improvement': sharpe_ratio(net_returns) - sharpe_ratio(si_only_returns),
        'max_dd': max_drawdown(net_returns),
    }

# ============================================================
# APPLICATION 10: ENTRY TIMING
# ============================================================

def app10_entry_timing(data: pd.DataFrame, si: pd.Series, cost_rate: float) -> Dict:
    """Test SI for entry timing."""
    returns = data['close'].pct_change()

    # Align
    common = si.index.intersection(returns.dropna().index)
    si_aligned = si.loc[common]
    returns_aligned = returns.loc[common]
    close = data['close'].loc[common]

    si_rank = si_aligned.rank(pct=True)

    # Price position
    high_20 = close.rolling(20).max()
    low_20 = close.rolling(20).min()
    price_pos = (close - low_20) / (high_20 - low_20 + 1e-10)

    # Entry signal
    signal = np.where((si_rank > 0.7) & (price_pos < 0.2), 2,    # Strong buy
             np.where((si_rank > 0.6) & (price_pos < 0.5), 1,     # Buy
             np.where((si_rank < 0.3) & (price_pos > 0.8), -1,    # Avoid
             np.where(si_rank < 0.3, 0, 0.5))))                   # Neutral
    signal = pd.Series(signal, index=common)

    # 5-day forward returns
    fwd_5d = returns_aligned.rolling(5).sum().shift(-5)

    # Calculate edge by signal type
    strong_buy_edge = fwd_5d[signal == 2].mean() if (signal == 2).sum() > 0 else np.nan
    buy_edge = fwd_5d[signal == 1].mean() if (signal == 1).sum() > 0 else np.nan
    avoid_edge = fwd_5d[signal == -1].mean() if (signal == -1).sum() > 0 else np.nan

    # Strategy returns
    strategy_returns = signal.shift(1).fillna(0) * returns_aligned
    net_returns = apply_costs(strategy_returns, signal.abs(), cost_rate)

    return {
        'strategy_sharpe': sharpe_ratio(net_returns),
        'strong_buy_edge': float(strong_buy_edge) if not np.isnan(strong_buy_edge) else None,
        'buy_edge': float(buy_edge) if not np.isnan(buy_edge) else None,
        'avoid_edge': float(avoid_edge) if not np.isnan(avoid_edge) else None,
        'max_dd': max_drawdown(net_returns),
    }

# ============================================================
# MAIN TESTING LOOP
# ============================================================

def main():
    print("\n" + "="*70)
    print("  TEST ALL 10 SI APPLICATIONS")
    print("="*70)
    print(f"  Time: {datetime.now().isoformat()}")

    loader = DataLoaderV2()
    all_results = {}

    # Load all data and compute SI
    print("\n  Loading data and computing SI...")
    asset_data = {}
    asset_si = {}

    all_assets = []
    for market, assets in ASSETS.items():
        all_assets.extend(assets)

    for asset in all_assets:
        print(f"    Loading {asset}...")
        data = loader.load(asset, MARKET_TYPE_MAP[asset])
        si = compute_si(data, window=SI_WINDOW)
        asset_data[asset] = data
        asset_si[asset] = si

    print(f"\n  Loaded {len(all_assets)} assets")

    # ============================================================
    # RUN ALL APPLICATIONS
    # ============================================================

    applications = [
        ("1. Risk Budgeting", app1_risk_budgeting, ['data', 'si', 'cost']),
        ("2. SI-ADX Spread", app2_spread_trading, ['data', 'si', 'cost']),
        ("3. Factor Timing", app3_factor_timing, ['data', 'si', 'cost']),
        ("4. Vol Forecasting", app4_vol_forecasting, ['data', 'si']),
        ("5. Dynamic Stop", app5_dynamic_stop, ['data', 'si', 'cost']),
        ("6. Regime Rebalance", app6_regime_rebalancing, ['data', 'si', 'cost']),
        ("7. Tail Hedge", app7_tail_hedge, ['data', 'si']),
        ("8. Cross-Asset", None, ['special']),  # Special handling
        ("9. Ensemble", app9_ensemble, ['data', 'si', 'cost']),
        ("10. Entry Timing", app10_entry_timing, ['data', 'si', 'cost']),
    ]

    for app_name, app_func, args in applications:
        print(f"\n{'='*70}")
        print(f"  {app_name}")
        print('='*70)

        app_results = {}

        if app_name == "8. Cross-Asset":
            # Special handling for cross-asset
            pairs = [('BTCUSDT', 'ETHUSDT'), ('SPY', 'QQQ'), ('EURUSD', 'GBPUSD')]
            for a1, a2 in pairs:
                pair_name = f"{a1}-{a2}"
                print(f"    Testing pair: {pair_name}")
                cost = (TRANSACTION_COSTS[a1] + TRANSACTION_COSTS[a2]) / 2
                result = app8_cross_asset(
                    asset_data[a1], asset_data[a2],
                    asset_si[a1], asset_si[a2], cost
                )
                app_results[pair_name] = result
                print(f"      Sharpe: {result['strategy_sharpe']:.3f}")
        else:
            for asset in all_assets:
                print(f"    Testing: {asset}")
                data = asset_data[asset]
                si = asset_si[asset]
                cost = TRANSACTION_COSTS[asset]

                try:
                    if 'cost' in args:
                        result = app_func(data, si, cost)
                    else:
                        result = app_func(data, si)
                    app_results[asset] = result

                    # Print key metric
                    if 'strategy_sharpe' in result:
                        print(f"      Sharpe: {result['strategy_sharpe']:.3f}")
                    elif 'improvement_pct' in result:
                        print(f"      Improvement: {result['improvement_pct']:.1f}%")
                except Exception as e:
                    print(f"      Error: {e}")
                    app_results[asset] = {'error': str(e)}

        all_results[app_name] = app_results

    # ============================================================
    # SUMMARY AND RANKING
    # ============================================================

    print("\n" + "="*70)
    print("  SUMMARY AND RANKING")
    print("="*70)

    summary = []

    for app_name, results in all_results.items():
        sharpes = []
        improvements = []

        for asset, res in results.items():
            if isinstance(res, dict) and 'error' not in res:
                if 'strategy_sharpe' in res:
                    sharpes.append(res['strategy_sharpe'])
                if 'improvement' in res:
                    improvements.append(res['improvement'])
                elif 'improvement_pct' in res:
                    improvements.append(res['improvement_pct'] / 100)

        avg_sharpe = np.mean(sharpes) if sharpes else 0
        avg_improvement = np.mean(improvements) if improvements else 0
        consistency = sum(1 for s in sharpes if s > 0) / len(sharpes) if sharpes else 0

        summary.append({
            'application': app_name,
            'avg_sharpe': avg_sharpe,
            'avg_improvement': avg_improvement,
            'consistency': consistency,
            'n_assets': len(sharpes),
        })

    # Sort by average Sharpe
    summary = sorted(summary, key=lambda x: x['avg_sharpe'], reverse=True)

    print(f"\n  {'Rank':<6} {'Application':<25} {'Avg Sharpe':>12} {'Improvement':>12} {'Consistency':>12}")
    print("  " + "-"*70)

    for i, s in enumerate(summary):
        print(f"  {i+1:<6} {s['application']:<25} {s['avg_sharpe']:>12.3f} {s['avg_improvement']:>+12.3f} {s['consistency']:>12.0%}")

    # Best application
    best = summary[0]
    print(f"\n  🏆 BEST APPLICATION: {best['application']}")
    print(f"     Average Sharpe: {best['avg_sharpe']:.3f}")
    print(f"     Consistency: {best['consistency']:.0%}")

    # Save results
    out_path = Path("results/application_testing/full_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'detailed_results': all_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

    return summary, all_results

if __name__ == "__main__":
    main()
