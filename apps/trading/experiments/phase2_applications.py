#!/usr/bin/env python3
"""
PHASE 2: SI APPLICATIONS

Based on 125 Phase 1 discoveries, implement practical applications:
1. Regime Detection System
2. Position Sizing Strategy
3. Crisis Early Warning
4. Volatility Forecasting
5. Factor Timing
6. SI-ADX Pairs Trading
7. Drawdown Prediction
8. Enhanced Technical Signals
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2
from src.data.loader_v2 import DataLoaderV2, MarketType

applications = []

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def result(msg):
    applications.append(msg)
    print(f"  ✅ {msg}")

def compute_si(data, window=7):
    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
    population.run(data)
    return population.compute_si_timeseries(data, window=window)

def sharpe_ratio(returns):
    if returns.std() == 0:
        return 0
    return returns.mean() / returns.std() * np.sqrt(252)

def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum / peak - 1)
    return dd.min()

def main():
    print("\n" + "="*70)
    print("  PHASE 2: SI APPLICATIONS")
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
    section("APP 1: REGIME DETECTION SYSTEM")
    # ============================================================
    
    print("\n  Using SI sticky extremes (80% persistence) for regime detection")
    
    regime_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        returns = data['close'].pct_change()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        # Define regimes based on SI percentiles
        low_thresh = np.percentile(si_aligned, 33)
        high_thresh = np.percentile(si_aligned, 67)
        
        si_regime = pd.Series(index=si_aligned.index, dtype=str)
        si_regime[si_aligned <= low_thresh] = 'Low_SI'
        si_regime[(si_aligned > low_thresh) & (si_aligned <= high_thresh)] = 'Mid_SI'
        si_regime[si_aligned > high_thresh] = 'High_SI'
        
        # Validate regimes: check if volatility differs across regimes
        vol = returns_aligned.rolling(14).std()
        
        common_final = si_regime.index.intersection(vol.dropna().index)
        
        vol_by_regime = {}
        for regime in ['Low_SI', 'Mid_SI', 'High_SI']:
            mask = si_regime.loc[common_final] == regime
            if mask.sum() > 10:
                vol_by_regime[regime] = vol.loc[common_final][mask].mean()
        
        # Persistence check
        regime_changes = (si_regime != si_regime.shift()).sum()
        persistence = 1 - regime_changes / len(si_regime)
        
        regime_results[name] = {
            'vol_by_regime': {k: float(v) for k, v in vol_by_regime.items()},
            'persistence': float(persistence),
            'regime_counts': si_regime.value_counts().to_dict(),
        }
        
        print(f"    Volatility by regime:")
        for regime, vol_val in vol_by_regime.items():
            print(f"      {regime}: {vol_val:.4f}")
        print(f"    Regime persistence: {persistence:.1%}")
        
        # Check if regimes are distinct
        if 'Low_SI' in vol_by_regime and 'High_SI' in vol_by_regime:
            vol_spread = vol_by_regime['Low_SI'] - vol_by_regime['High_SI']
            if vol_spread > 0:
                result(f"SI regimes capture volatility in {name} (Low-High spread: {vol_spread:.4f})")
    
    all_results['regime_detection'] = regime_results
    
    # ============================================================
    section("APP 2: POSITION SIZING STRATEGY")
    # ============================================================
    
    print("\n  Using SI extremes (p10/p90) for dynamic position sizing")
    
    sizing_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        returns = data['close'].pct_change()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        common_final = si_aligned.index.intersection(returns_aligned.dropna().index)
        si_vals = si_aligned.loc[common_final]
        ret_vals = returns_aligned.loc[common_final].shift(-1).dropna()  # Next day return
        
        common_shift = si_vals.index.intersection(ret_vals.index)
        si_vals = si_vals.loc[common_shift]
        ret_vals = ret_vals.loc[common_shift]
        
        # Strategy 1: Constant sizing
        const_returns = ret_vals
        const_sharpe = sharpe_ratio(const_returns)
        const_dd = max_drawdown(const_returns)
        
        # Strategy 2: SI-based sizing (high SI = larger position)
        si_normalized = (si_vals - si_vals.min()) / (si_vals.max() - si_vals.min() + 1e-10)
        si_weight = 0.5 + si_normalized  # Range: 0.5 to 1.5
        si_returns = ret_vals * si_weight
        si_sharpe = sharpe_ratio(si_returns)
        si_dd = max_drawdown(si_returns)
        
        # Strategy 3: Inverse volatility sizing
        vol = returns_aligned.rolling(14).std()
        vol_aligned = vol.loc[common_shift]
        inv_vol_weight = 1 / (vol_aligned + 0.001)
        inv_vol_weight = inv_vol_weight / inv_vol_weight.mean()  # Normalize
        inv_vol_returns = ret_vals * inv_vol_weight
        inv_vol_sharpe = sharpe_ratio(inv_vol_returns)
        inv_vol_dd = max_drawdown(inv_vol_returns)
        
        sizing_results[name] = {
            'constant': {'sharpe': float(const_sharpe), 'max_dd': float(const_dd)},
            'si_sizing': {'sharpe': float(si_sharpe), 'max_dd': float(si_dd)},
            'inv_vol': {'sharpe': float(inv_vol_sharpe), 'max_dd': float(inv_vol_dd)},
        }
        
        print(f"    {'Strategy':<15} {'Sharpe':>10} {'Max DD':>10}")
        print("    " + "-"*37)
        print(f"    {'Constant':<15} {const_sharpe:>10.2f} {const_dd:>10.1%}")
        print(f"    {'SI-Based':<15} {si_sharpe:>10.2f} {si_dd:>10.1%}")
        print(f"    {'Inv-Vol':<15} {inv_vol_sharpe:>10.2f} {inv_vol_dd:>10.1%}")
        
        if si_sharpe > const_sharpe:
            result(f"SI sizing improves Sharpe in {name} ({const_sharpe:.2f} → {si_sharpe:.2f})")
    
    all_results['position_sizing'] = sizing_results
    
    # ============================================================
    section("APP 3: CRISIS EARLY WARNING")
    # ============================================================
    
    print("\n  Using SI crash behavior (-27% in 2022) for early warning")
    
    warning_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        returns = data['close'].pct_change()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        # Detect SI drops > 2σ
        si_change = si_aligned.diff()
        si_change_std = si_change.std()
        si_drops = si_change < -2 * si_change_std
        
        # Check if SI drops precede drawdowns
        future_20d_return = returns_aligned.rolling(20).sum().shift(-20)
        
        common_final = si_drops.index.intersection(future_20d_return.dropna().index)
        
        si_drop_mask = si_drops.loc[common_final]
        future_ret = future_20d_return.loc[common_final]
        
        # Average return after SI drop vs no drop
        ret_after_drop = future_ret[si_drop_mask].mean() if si_drop_mask.sum() > 0 else 0
        ret_after_normal = future_ret[~si_drop_mask].mean()
        
        # Early warning metrics
        n_drops = si_drop_mask.sum()
        true_warnings = (si_drop_mask & (future_ret < -0.05)).sum()  # Drop followed by 5%+ loss
        false_alarms = (si_drop_mask & (future_ret >= 0)).sum()
        
        precision = true_warnings / (n_drops + 1e-10)
        
        warning_results[name] = {
            'n_warnings': int(n_drops),
            'avg_return_after_warning': float(ret_after_drop),
            'avg_return_normal': float(ret_after_normal),
            'precision': float(precision),
            'true_warnings': int(true_warnings),
            'false_alarms': int(false_alarms),
        }
        
        print(f"    SI drop warnings: {n_drops}")
        print(f"    Avg 20d return after SI drop: {ret_after_drop:+.2%}")
        print(f"    Avg 20d return normal: {ret_after_normal:+.2%}")
        print(f"    Precision (true positives): {precision:.1%}")
        
        if ret_after_drop < ret_after_normal - 0.01:
            result(f"SI drops predict worse returns in {name} ({ret_after_drop:.1%} vs {ret_after_normal:.1%})")
    
    all_results['crisis_warning'] = warning_results
    
    # ============================================================
    section("APP 4: VOLATILITY FORECASTING")
    # ============================================================
    
    print("\n  Using SI-volatility correlation for forecasting")
    
    vol_forecast_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        returns = data['close'].pct_change()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        
        # Realized volatility
        realized_vol = returns.rolling(14).std()
        
        # Future volatility (target)
        future_vol = realized_vol.shift(-5)  # 5 days ahead
        
        common_final = si_aligned.index.intersection(future_vol.dropna().index).intersection(realized_vol.dropna().index)
        
        si_vals = si_aligned.loc[common_final].values
        current_vol = realized_vol.loc[common_final].values
        target_vol = future_vol.loc[common_final].values
        
        # Baseline: persistence model (future vol = current vol)
        baseline_rmse = np.sqrt(mean_squared_error(target_vol, current_vol))
        
        # SI-enhanced model: future vol = a + b*current_vol + c*SI
        X = np.column_stack([np.ones(len(si_vals)), current_vol, si_vals])
        
        # Train/test split
        train_size = int(len(X) * 0.7)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = target_vol[:train_size], target_vol[train_size:]
        
        # Fit model
        model = Ridge(alpha=0.1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        si_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        baseline_test_rmse = np.sqrt(mean_squared_error(y_test, current_vol[train_size:]))
        
        improvement = (baseline_test_rmse - si_rmse) / baseline_test_rmse * 100
        
        vol_forecast_results[name] = {
            'baseline_rmse': float(baseline_test_rmse),
            'si_model_rmse': float(si_rmse),
            'improvement_pct': float(improvement),
            'si_coefficient': float(model.coef_[2]),
        }
        
        print(f"    Baseline RMSE: {baseline_test_rmse:.6f}")
        print(f"    SI-Model RMSE: {si_rmse:.6f}")
        print(f"    Improvement: {improvement:+.1f}%")
        print(f"    SI coefficient: {model.coef_[2]:+.6f}")
        
        if improvement > 5:
            result(f"SI improves vol forecasting in {name} by {improvement:.0f}%")
    
    all_results['vol_forecasting'] = vol_forecast_results
    
    # ============================================================
    section("APP 5: FACTOR TIMING")
    # ============================================================
    
    print("\n  Using SI as market clarity to time momentum factor")
    
    factor_timing_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        returns = data['close'].pct_change()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        # Momentum signal: past 7-day return
        momentum = returns_aligned.rolling(7).sum()
        
        common_final = si_aligned.index.intersection(momentum.dropna().index)
        
        si_vals = si_aligned.loc[common_final]
        mom_vals = momentum.loc[common_final]
        ret_vals = returns_aligned.loc[common_final].shift(-1).dropna()
        
        common_shift = si_vals.index.intersection(ret_vals.index)
        si_vals = si_vals.loc[common_shift]
        mom_vals = mom_vals.loc[common_shift]
        ret_vals = ret_vals.loc[common_shift]
        
        # Strategy 1: Always follow momentum
        mom_signal = np.sign(mom_vals)
        mom_returns = mom_signal * ret_vals
        mom_sharpe = sharpe_ratio(mom_returns)
        
        # Strategy 2: Follow momentum only when SI is high
        si_median = si_vals.median()
        high_si_mask = si_vals > si_median
        
        timed_signal = mom_signal.copy()
        timed_signal[~high_si_mask] = 0  # No position when SI is low
        timed_returns = timed_signal * ret_vals
        timed_sharpe = sharpe_ratio(timed_returns)
        
        # Strategy 3: Follow momentum when SI high, mean-revert when SI low
        mr_signal = -mom_signal  # Opposite of momentum
        combined_signal = mom_signal.copy()
        combined_signal[~high_si_mask] = mr_signal[~high_si_mask]
        combined_returns = combined_signal * ret_vals
        combined_sharpe = sharpe_ratio(combined_returns)
        
        factor_timing_results[name] = {
            'momentum_always': {'sharpe': float(mom_sharpe)},
            'momentum_high_si': {'sharpe': float(timed_sharpe)},
            'regime_switching': {'sharpe': float(combined_sharpe)},
        }
        
        print(f"    {'Strategy':<20} {'Sharpe':>10}")
        print("    " + "-"*32)
        print(f"    {'Momentum Always':<20} {mom_sharpe:>10.2f}")
        print(f"    {'Momentum High-SI':<20} {timed_sharpe:>10.2f}")
        print(f"    {'Regime Switching':<20} {combined_sharpe:>10.2f}")
        
        best_sharpe = max(timed_sharpe, combined_sharpe)
        if best_sharpe > mom_sharpe:
            result(f"SI timing improves momentum in {name} ({mom_sharpe:.2f} → {best_sharpe:.2f})")
    
    all_results['factor_timing'] = factor_timing_results
    
    # ============================================================
    section("APP 6: SI-ADX PAIRS TRADING")
    # ============================================================
    
    print("\n  Using SI-ADX cointegration for spread trading")
    
    pairs_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        
        # Compute ADX
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
        adx = dx.rolling(14).mean()
        
        common = si.index.intersection(adx.dropna().index)
        si_aligned = si.loc[common].dropna()
        adx_aligned = adx.loc[common].dropna()
        
        common_final = si_aligned.index.intersection(adx_aligned.index)
        si_vals = si_aligned.loc[common_final]
        adx_vals = adx_aligned.loc[common_final]
        
        # Normalize both series
        si_norm = (si_vals - si_vals.mean()) / si_vals.std()
        adx_norm = (adx_vals - adx_vals.mean()) / adx_vals.std()
        
        # Spread
        spread = si_norm - 0.5 * adx_norm  # Adjust hedge ratio
        
        # Mean reversion strategy on spread
        spread_mean = spread.rolling(30).mean()
        spread_std = spread.rolling(30).std()
        z_score = (spread - spread_mean) / (spread_std + 1e-10)
        
        # Entry: z-score > 2 (short spread) or < -2 (long spread)
        position = pd.Series(0, index=z_score.index)
        position[z_score > 2] = -1
        position[z_score < -2] = 1
        
        # Returns: use SI change as proxy for spread return
        si_returns = si_vals.pct_change()
        strategy_returns = position.shift(1) * si_returns
        
        common_ret = strategy_returns.dropna().index
        strategy_returns = strategy_returns.loc[common_ret]
        
        pairs_sharpe = sharpe_ratio(strategy_returns)
        win_rate = (strategy_returns > 0).mean()
        n_trades = (position.diff() != 0).sum()
        
        pairs_results[name] = {
            'sharpe': float(pairs_sharpe),
            'win_rate': float(win_rate),
            'n_trades': int(n_trades),
        }
        
        print(f"    Spread trading Sharpe: {pairs_sharpe:.2f}")
        print(f"    Win rate: {win_rate:.1%}")
        print(f"    Number of trades: {n_trades}")
        
        if pairs_sharpe > 0.5:
            result(f"SI-ADX spread trading works in {name} (Sharpe={pairs_sharpe:.2f})")
    
    all_results['pairs_trading'] = pairs_results
    
    # ============================================================
    section("APP 7: DRAWDOWN PREDICTION")
    # ============================================================
    
    print("\n  Using SI level to predict future drawdowns")
    
    dd_pred_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        returns = data['close'].pct_change()
        
        common = si.index.intersection(returns.index)
        si_aligned = si.loc[common].dropna()
        returns_aligned = returns.loc[common]
        
        # Future 20-day drawdown
        future_dd = []
        for i in range(len(returns_aligned) - 20):
            future_ret = returns_aligned.iloc[i:i+20]
            cum_ret = (1 + future_ret).cumprod()
            peak = cum_ret.expanding().max()
            dd = (cum_ret / peak - 1).min()
            future_dd.append(dd)
        
        future_dd = pd.Series(future_dd, index=returns_aligned.index[:len(future_dd)])
        
        common_final = si_aligned.index[:len(future_dd)]
        common_final = common_final.intersection(si_aligned.dropna().index)
        
        si_vals = si_aligned.loc[common_final]
        dd_vals = future_dd.loc[common_final]
        
        # Correlation
        r, p = spearmanr(si_vals, dd_vals)
        
        # Binary classification: major drawdown (< -5%)
        major_dd = dd_vals < -0.05
        
        # Low SI predicts drawdown?
        si_median = si_vals.median()
        low_si = si_vals < si_median
        
        # Accuracy
        pred_dd = low_si
        if major_dd.sum() > 0:
            try:
                auc = roc_auc_score(major_dd, ~low_si)  # Flip: high SI = less likely drawdown
            except:
                auc = 0.5
        else:
            auc = 0.5
        
        dd_pred_results[name] = {
            'correlation': float(r),
            'p_value': float(p),
            'auc_roc': float(auc),
            'major_dd_rate': float(major_dd.mean()),
        }
        
        print(f"    SI-Drawdown correlation: r={r:+.3f} (p={p:.4f})")
        print(f"    AUC-ROC: {auc:.3f}")
        print(f"    Major drawdown rate: {major_dd.mean():.1%}")
        
        if auc > 0.55:
            result(f"SI predicts drawdowns in {name} (AUC={auc:.2f})")
    
    all_results['drawdown_prediction'] = dd_pred_results
    
    # ============================================================
    section("APP 8: ENHANCED TECHNICAL SIGNALS")
    # ============================================================
    
    print("\n  Combining SI with RSI for enhanced signals")
    
    enhanced_results = {}
    
    for name, (data, mtype) in assets.items():
        print(f"\n  {name}:")
        si = compute_si(data)
        returns = data['close'].pct_change()
        
        # Compute RSI
        delta = data['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        common = si.index.intersection(rsi.dropna().index).intersection(returns.dropna().index)
        si_aligned = si.loc[common].dropna()
        rsi_aligned = rsi.loc[common]
        returns_aligned = returns.loc[common]
        
        common_final = si_aligned.index.intersection(rsi_aligned.dropna().index)
        si_vals = si_aligned.loc[common_final]
        rsi_vals = rsi_aligned.loc[common_final]
        ret_vals = returns_aligned.loc[common_final].shift(-1).dropna()
        
        common_shift = si_vals.index.intersection(ret_vals.index)
        si_vals = si_vals.loc[common_shift]
        rsi_vals = rsi_vals.loc[common_shift]
        ret_vals = ret_vals.loc[common_shift]
        
        # Strategy 1: RSI alone
        rsi_signal = pd.Series(0, index=rsi_vals.index)
        rsi_signal[rsi_vals < 30] = 1   # Oversold = buy
        rsi_signal[rsi_vals > 70] = -1  # Overbought = sell
        rsi_returns = rsi_signal * ret_vals
        rsi_sharpe = sharpe_ratio(rsi_returns)
        rsi_win_rate = (rsi_returns[rsi_signal != 0] > 0).mean() if (rsi_signal != 0).sum() > 0 else 0
        
        # Strategy 2: RSI + SI confirmation
        si_median = si_vals.median()
        enhanced_signal = rsi_signal.copy()
        # Only take RSI signals when SI is high (clearer market)
        enhanced_signal[(rsi_signal != 0) & (si_vals < si_median)] = 0
        enhanced_returns = enhanced_signal * ret_vals
        enhanced_sharpe = sharpe_ratio(enhanced_returns)
        enhanced_win_rate = (enhanced_returns[enhanced_signal != 0] > 0).mean() if (enhanced_signal != 0).sum() > 0 else 0
        
        enhanced_results[name] = {
            'rsi_only': {'sharpe': float(rsi_sharpe), 'win_rate': float(rsi_win_rate)},
            'rsi_plus_si': {'sharpe': float(enhanced_sharpe), 'win_rate': float(enhanced_win_rate)},
        }
        
        print(f"    {'Strategy':<15} {'Sharpe':>10} {'Win Rate':>10}")
        print("    " + "-"*37)
        print(f"    {'RSI Only':<15} {rsi_sharpe:>10.2f} {rsi_win_rate:>10.1%}")
        print(f"    {'RSI + SI':<15} {enhanced_sharpe:>10.2f} {enhanced_win_rate:>10.1%}")
        
        if enhanced_win_rate > rsi_win_rate + 0.02:
            result(f"SI improves RSI win rate in {name} ({rsi_win_rate:.0%} → {enhanced_win_rate:.0%})")
    
    all_results['enhanced_signals'] = enhanced_results
    
    # ============================================================
    section("PHASE 2 SUMMARY")
    # ============================================================
    
    print(f"\n  Total successful applications: {len(applications)}")
    for i, app in enumerate(applications, 1):
        print(f"    {i}. {app}")
    
    out_path = Path("results/phase2_applications/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'applications': applications,
            'results': all_results,
        }, f, indent=2, default=str)
    
    print(f"\n  Results saved: {out_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
