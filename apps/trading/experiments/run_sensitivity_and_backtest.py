#!/usr/bin/env python3
"""
SI Sensitivity Analysis + Backtest with Transaction Costs

Tests:
1. SI window sensitivity: 72h (3d), 168h (7d), 336h (14d)
2. Agent count sensitivity: 3, 5, 9 per strategy
3. Backtest with realistic transaction costs
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

from src.agents.strategies import DEFAULT_STRATEGIES
from src.competition.niche_population import NichePopulation
from src.data.loader_v2 import DataLoaderV2, MarketType, TRANSACTION_COSTS


# Test configurations
SI_WINDOWS = [72, 168, 336]  # 3d, 7d, 14d for daily data -> 72, 168, 336 hours equivalent
SI_AGENTS = [3, 5, 9]

# For daily data, we need to adjust windows
DAILY_WINDOWS = [14, 30, 60]  # 2wk, 1mo, 2mo for daily data


def run_sensitivity_test(symbol: str, market_type: MarketType, loader: DataLoaderV2):
    """Test SI sensitivity to window and agent count."""
    print(f"\n{'='*60}")
    print(f"SENSITIVITY: {symbol}")
    print("=" * 60)

    try:
        data = loader.load(symbol, market_type)
        train, val, test = loader.temporal_split_with_purging(data, frequency='daily')

        if len(train) < 200:
            return None

        results = []

        for window in DAILY_WINDOWS:
            for n_agents in SI_AGENTS:
                print(f"   Testing window={window}d, agents={n_agents}...", end=" ")

                try:
                    population = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=n_agents)
                    start_idx = min(50, len(train) // 3)
                    population.run(train, start_idx=start_idx)
                    si = population.compute_si_timeseries(train, window=window)

                    # Compute correlation with returns
                    returns = train['close'].pct_change()
                    common = si.index.intersection(returns.index)
                    if len(common) < 50:
                        print("insufficient data")
                        continue

                    # Correlation with abs returns (volatility proxy)
                    r_vol, p_vol = spearmanr(si.loc[common], abs(returns.loc[common]))

                    # Correlation with trend
                    trend = returns.rolling(window).mean().loc[common]
                    r_trend, p_trend = spearmanr(si.loc[common].dropna(), trend.dropna())

                    results.append({
                        'window': window,
                        'n_agents': n_agents,
                        'si_mean': float(si.mean()),
                        'si_std': float(si.std()),
                        'r_volatility': float(r_vol) if not np.isnan(r_vol) else 0,
                        'p_volatility': float(p_vol) if not np.isnan(p_vol) else 1,
                        'r_trend': float(r_trend) if not np.isnan(r_trend) else 0,
                        'p_trend': float(p_trend) if not np.isnan(p_trend) else 1,
                    })
                    print(f"SI={si.mean():.3f}, r_vol={r_vol:.3f}")

                except Exception as e:
                    print(f"error: {e}")

        return results

    except Exception as e:
        print(f"   Error: {e}")
        return None


def backtest_with_costs(symbol: str, market_type: MarketType, loader: DataLoaderV2):
    """Backtest with realistic transaction costs."""
    print(f"\n{'='*60}")
    print(f"BACKTEST: {symbol}")
    print(f"Transaction cost: {TRANSACTION_COSTS[market_type]*100:.3f}%")
    print("=" * 60)

    try:
        data = loader.load(symbol, market_type)
        train, val, test = loader.temporal_split_with_purging(data, frequency='daily')

        # Use test set for final backtest
        test_data = test.copy()

        if len(test_data) < 50:
            return None

        # Compute SI on test
        population = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=3)
        start_idx = min(30, len(test_data) // 3)
        population.run(test_data, start_idx=start_idx)
        si = population.compute_si_timeseries(test_data, window=30)  # 30-day window

        # Align
        common = si.index.intersection(test_data.index)
        if len(common) < 30:
            return None

        si = si.loc[common]
        prices = test_data.loc[common, 'close']
        returns = prices.pct_change()

        # SI percentile
        si_pct = si.rolling(60).rank(pct=True)

        # Strategy: Long when SI > 60th percentile
        position = (si_pct > 0.6).astype(float).shift(1).fillna(0)

        # Returns
        strategy_returns_gross = position * returns

        # Transaction costs
        cost = TRANSACTION_COSTS[market_type]
        position_changes = position.diff().abs()
        costs = position_changes * cost * 2  # 2x for round-trip

        strategy_returns_net = strategy_returns_gross - costs

        # Drop NaN
        strategy_returns_net = strategy_returns_net.dropna()
        strategy_returns_gross = strategy_returns_gross.dropna()

        if len(strategy_returns_net) < 10:
            return None

        # Metrics
        def calc_metrics(rets, label):
            total = (1 + rets).prod() - 1
            ann = (1 + total) ** (252 / len(rets)) - 1
            vol = rets.std() * np.sqrt(252)
            sharpe = ann / vol if vol > 0 else 0

            # Drawdown
            cum = (1 + rets).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            max_dd = dd.min()

            return {
                'total_return': float(total),
                'annual_return': float(ann),
                'volatility': float(vol),
                'sharpe': float(sharpe),
                'max_drawdown': float(max_dd),
                'n_periods': len(rets)
            }

        gross = calc_metrics(strategy_returns_gross, 'gross')
        net = calc_metrics(strategy_returns_net, 'net')

        # Trade stats
        n_trades = int(position_changes.sum())
        total_cost = float(costs.sum())

        print(f"   Gross Return: {gross['total_return']*100:+.2f}%")
        print(f"   Net Return:   {net['total_return']*100:+.2f}%")
        print(f"   Gross Sharpe: {gross['sharpe']:.3f}")
        print(f"   Net Sharpe:   {net['sharpe']:.3f}")
        print(f"   Total Costs:  {total_cost*100:.3f}%")
        print(f"   # Trades:     {n_trades}")

        return {
            'symbol': symbol,
            'market': market_type.value,
            'cost_bps': TRANSACTION_COSTS[market_type] * 10000,
            'gross': gross,
            'net': net,
            'n_trades': n_trades,
            'total_cost': total_cost,
            'cost_impact': (gross['total_return'] - net['total_return']) if gross['total_return'] != 0 else 0
        }

    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 70)
    print("SI SENSITIVITY ANALYSIS + BACKTEST WITH COSTS")
    print("=" * 70)

    loader = DataLoaderV2(purge_days=7, embargo_days=1)

    MARKETS = {
        MarketType.CRYPTO: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        MarketType.FOREX: ['EURUSD', 'GBPUSD', 'USDJPY'],
        MarketType.STOCKS: ['SPY', 'QQQ', 'AAPL'],
        MarketType.COMMODITIES: ['GOLD', 'OIL'],
    }

    # ============================================================
    # SENSITIVITY ANALYSIS
    # ============================================================
    print("\n" + "#" * 70)
    print("# PART 1: SENSITIVITY ANALYSIS")
    print("#" * 70)

    all_sensitivity = {}

    # Test on one asset per market for efficiency
    test_assets = {
        MarketType.CRYPTO: 'BTCUSDT',
        MarketType.FOREX: 'EURUSD',
        MarketType.STOCKS: 'SPY',
        MarketType.COMMODITIES: 'GOLD',
    }

    for market_type, symbol in test_assets.items():
        result = run_sensitivity_test(symbol, market_type, loader)
        if result:
            all_sensitivity[symbol] = result

    # Summary
    print("\n" + "=" * 70)
    print("SENSITIVITY SUMMARY")
    print("=" * 70)

    print("\n| Symbol | Window | Agents | SI Mean | SI Std | r(vol) | r(trend) |")
    print("|--------|--------|--------|---------|--------|--------|----------|")

    for symbol, results in all_sensitivity.items():
        for r in results:
            print(f"| {symbol:6} | {r['window']:>6}d | {r['n_agents']:>6} | {r['si_mean']:>7.3f} | {r['si_std']:>6.3f} | {r['r_volatility']:>+6.3f} | {r['r_trend']:>+8.3f} |")

    # ============================================================
    # BACKTEST WITH TRANSACTION COSTS
    # ============================================================
    print("\n" + "#" * 70)
    print("# PART 2: BACKTEST WITH TRANSACTION COSTS")
    print("#" * 70)

    all_backtest = {}

    for market_type, symbols in MARKETS.items():
        for symbol in symbols:
            result = backtest_with_costs(symbol, market_type, loader)
            if result:
                all_backtest[symbol] = result

    # Summary
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY (With Transaction Costs)")
    print("=" * 70)

    print("\n| Symbol | Cost (bps) | Gross Ret | Net Ret | Gross Sharpe | Net Sharpe | Cost Impact |")
    print("|--------|------------|-----------|---------|--------------|------------|-------------|")

    for symbol, r in all_backtest.items():
        print(f"| {symbol:6} | {r['cost_bps']:>10.1f} | {r['gross']['total_return']*100:>+8.2f}% | {r['net']['total_return']*100:>+6.2f}% | {r['gross']['sharpe']:>+12.3f} | {r['net']['sharpe']:>+10.3f} | {r['cost_impact']*100:>+10.2f}% |")

    # Save results
    output_dir = Path("results/sensitivity_backtest")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump({
            'sensitivity': all_sensitivity,
            'backtest': all_backtest,
            'transaction_costs': {k.value: v for k, v in TRANSACTION_COSTS.items()}
        }, f, indent=2, default=float)

    print(f"\n✅ Results saved to: {output_dir / 'results.json'}")

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    # Calculate averages
    avg_net_return = np.mean([r['net']['total_return'] for r in all_backtest.values()])
    avg_net_sharpe = np.mean([r['net']['sharpe'] for r in all_backtest.values()])
    avg_cost_impact = np.mean([r['cost_impact'] for r in all_backtest.values()])

    print(f"\nAverage Net Return: {avg_net_return*100:+.2f}%")
    print(f"Average Net Sharpe: {avg_net_sharpe:.3f}")
    print(f"Average Cost Impact: {avg_cost_impact*100:.2f}% of returns")

    if avg_net_sharpe > 0.3:
        print("\n✅ STRATEGY PROFITABLE AFTER COSTS")
    elif avg_net_sharpe > 0:
        print("\n⚠️  STRATEGY MARGINALLY PROFITABLE")
    else:
        print("\n❌ STRATEGY NOT PROFITABLE AFTER COSTS")


if __name__ == "__main__":
    main()
