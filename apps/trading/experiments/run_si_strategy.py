#!/usr/bin/env python3
"""
Run SI-based trading strategy backtest.
Uses the 3 confirmed SI correlates to generate trading signals.

ALWAYS multi-asset - never single coin!
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.data.loader import DataLoader
from src.agents.strategies import DEFAULT_STRATEGIES
from src.competition.niche_population import NichePopulation
from src.backtest.si_strategy import SITradingStrategy


def run_single_asset(symbol: str, output_dir: Path) -> dict:
    """Run strategy on a single asset."""
    print(f"\n{'='*60}")
    print(f"BACKTESTING: {symbol}")
    print("=" * 60)

    try:
        # 1. Load data
        loader = DataLoader()
        data = loader.load(symbol)
        print(f"   Loaded {len(data)} rows")

        # 2. Split data
        train, val, test = loader.temporal_split(data)

        # 3. Run competition on TRAIN to compute SI
        print("\n1. Computing SI on TRAIN set...")
        population = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=3)
        population.run(train, start_idx=200)
        si_train = population.compute_si_timeseries(train, window=168)
        print(f"   SI computed: mean={si_train.mean():.3f}, std={si_train.std():.3f}")

        # 4. Run competition on VAL
        print("\n2. Computing SI on VAL set...")
        population_val = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=3)
        population_val.run(val, start_idx=200)
        si_val = population_val.compute_si_timeseries(val, window=168)
        print(f"   SI computed: mean={si_val.mean():.3f}, std={si_val.std():.3f}")

        # 5. Run competition on TEST
        print("\n3. Computing SI on TEST set...")
        population_test = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=3)
        population_test.run(test, start_idx=200)
        si_test = population_test.compute_si_timeseries(test, window=168)
        print(f"   SI computed: mean={si_test.mean():.3f}, std={si_test.std():.3f}")

        # 6. Initialize strategy
        strategy = SITradingStrategy(
            si_high_threshold=0.7,
            si_low_threshold=0.3,
            trend_threshold=0.05,
            lookback=168
        )

        # 7. Backtest on TRAIN (for calibration)
        print("\n4. Backtesting on TRAIN...")
        train_results = strategy.backtest(train, si_train)
        print(f"   Sharpe: {train_results.get('sharpe_ratio', 'N/A'):.3f}")
        print(f"   Return: {train_results.get('total_return', 0)*100:.2f}%")

        # 8. Backtest on VAL (out-of-sample validation)
        print("\n5. Backtesting on VAL...")
        val_results = strategy.backtest(val, si_val)
        print(f"   Sharpe: {val_results.get('sharpe_ratio', 'N/A'):.3f}")
        print(f"   Return: {val_results.get('total_return', 0)*100:.2f}%")

        # 9. Backtest on TEST (final holdout)
        print("\n6. Backtesting on TEST (final)...")
        test_results = strategy.backtest(test, si_test)
        print(f"   Sharpe: {test_results.get('sharpe_ratio', 'N/A'):.3f}")
        print(f"   Return: {test_results.get('total_return', 0)*100:.2f}%")
        print(f"   Max Drawdown: {test_results.get('max_drawdown', 0)*100:.2f}%")

        return {
            'symbol': symbol,
            'status': 'SUCCESS',
            'train': train_results,
            'val': val_results,
            'test': test_results
        }

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            'symbol': symbol,
            'status': 'ERROR',
            'error': str(e)
        }


def main():
    print("=" * 60)
    print("SI-BASED TRADING STRATEGY BACKTEST")
    print("⚠️  MULTI-ASSET MODE - Testing on BTC, ETH, SOL")
    print("=" * 60)

    # ALWAYS multi-asset [[memory:12438146]]
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

    output_dir = Path("results/si_strategy")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for symbol in symbols:
        result = run_single_asset(symbol, output_dir)
        all_results[symbol] = result

    # Summary
    print("\n" + "=" * 60)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("=" * 60)

    print("\n| Asset | TRAIN Sharpe | VAL Sharpe | TEST Sharpe | TEST Return |")
    print("|-------|--------------|------------|-------------|-------------|")

    for symbol, result in all_results.items():
        if result['status'] == 'SUCCESS':
            train_sharpe = result['train'].get('sharpe_ratio', 0)
            val_sharpe = result['val'].get('sharpe_ratio', 0)
            test_sharpe = result['test'].get('sharpe_ratio', 0)
            test_ret = result['test'].get('total_return', 0) * 100
            print(f"| {symbol:5} | {train_sharpe:12.3f} | {val_sharpe:10.3f} | {test_sharpe:11.3f} | {test_ret:10.2f}% |")
        else:
            print(f"| {symbol:5} | ERROR |")

    # Regime analysis
    print("\n" + "=" * 60)
    print("REGIME PERFORMANCE (TEST SET)")
    print("=" * 60)

    for symbol, result in all_results.items():
        if result['status'] == 'SUCCESS':
            print(f"\n{symbol}:")
            regime_perf = result['test'].get('regime_performance', {})
            for regime, perf in regime_perf.items():
                count = perf.get('count', 0)
                mean_ret = perf.get('mean_return', 0) * 100
                total_ret = perf.get('total_return', 0) * 100
                print(f"  {regime:12}: {count:5} trades, mean={mean_ret:+.4f}%, total={total_ret:+.2f}%")

    # Save results
    with open(output_dir / "strategy_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    # Check if strategy is profitable
    test_sharpes = []
    for symbol, result in all_results.items():
        if result['status'] == 'SUCCESS':
            test_sharpes.append(result['test'].get('sharpe_ratio', 0))

    avg_sharpe = np.mean(test_sharpes) if test_sharpes else 0

    if avg_sharpe > 0.5:
        print(f"✅ STRATEGY PROFITABLE: Average TEST Sharpe = {avg_sharpe:.3f}")
        print("   Consider paper trading before live deployment.")
    elif avg_sharpe > 0:
        print(f"⚠️  STRATEGY MARGINALLY PROFITABLE: Average TEST Sharpe = {avg_sharpe:.3f}")
        print("   Needs optimization before deployment.")
    else:
        print(f"❌ STRATEGY NOT PROFITABLE: Average TEST Sharpe = {avg_sharpe:.3f}")
        print("   SI correlations may not translate to trading edge.")

    print(f"\nResults saved to: {output_dir / 'strategy_results.json'}")

    return avg_sharpe > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
