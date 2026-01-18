#!/usr/bin/env python3
"""
SI DEEP EXPLORATION - CORRECTED VERSION

Fixes all audit issues:
1. Test on ALL assets, not just BTC
2. Train/test split for threshold optimization
3. FDR correction for multiple testing
4. Proper return calculation with costs
5. Cross-asset consistency reporting
6. Out-of-sample validation

Author: Yuhao Li, University of Pennsylvania
Date: January 18, 2026
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.agents.strategies_v2 import get_default_strategies
from src.competition.niche_population_v2 import NichePopulationV2
from src.data.loader_v2 import DataLoaderV2, MarketType

# ============================================================================
# CONFIGURATION
# ============================================================================

MARKETS = {
    MarketType.CRYPTO: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    MarketType.FOREX: ['EURUSD', 'GBPUSD', 'USDJPY'],
    MarketType.STOCKS: ['SPY', 'QQQ', 'AAPL'],
    MarketType.COMMODITIES: ['GC', 'CL'],
}

TRANSACTION_COSTS = {
    MarketType.CRYPTO: 0.001,      # 10 bps
    MarketType.FOREX: 0.0002,      # 2 bps
    MarketType.STOCKS: 0.0005,     # 5 bps
    MarketType.COMMODITIES: 0.001, # 10 bps
}

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
# TEST_RATIO = 0.2 (remaining)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for analysis."""
    features = pd.DataFrame(index=data.index)

    returns = data['close'].pct_change()
    features['returns'] = returns

    # ADX (14-day)
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    plus_dm = (data['high'] - data['high'].shift()).clip(lower=0)
    minus_dm = (data['low'].shift() - data['low']).clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    features['adx'] = dx.rolling(14).mean()

    # Volatility (14-day)
    features['volatility'] = returns.rolling(14).std()

    # RSI (14-day)
    delta = data['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))

    # Future returns (for prediction testing) - 1 day ahead
    features['future_return_1d'] = returns.shift(-1)
    features['future_return_5d'] = returns.rolling(5).sum().shift(-5)

    return features


def compute_si(data: pd.DataFrame, window: int = 7) -> pd.Series:
    """Compute SI for a given dataset."""
    if len(data) < window * 3:
        return pd.Series(dtype=float)

    strategies = get_default_strategies('daily')
    population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')

    try:
        population.run(data)
        si = population.compute_si_timeseries(data, window=window)
        return si
    except Exception as e:
        print(f"    Error computing SI: {e}")
        return pd.Series(dtype=float)


def split_data(data: pd.DataFrame) -> tuple:
    """Split data into train/val/test sets."""
    n = len(data)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = data.iloc[:train_end]
    val = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]

    return train, val, test


# ============================================================================
# CORRECTED ANALYSIS FUNCTIONS
# ============================================================================

def analyze_correlations_single_asset(data: pd.DataFrame, si: pd.Series,
                                       features: pd.DataFrame, asset_name: str) -> dict:
    """
    Analyze SI correlations for a single asset with proper methodology.
    """
    results = {
        'asset': asset_name,
        'n_observations': 0,
        'correlations': {},
    }

    # Align data
    common_idx = si.index.intersection(features.index)
    if len(common_idx) < 50:
        results['status'] = 'insufficient_data'
        return results

    si_aligned = si.loc[common_idx]
    features_aligned = features.loc[common_idx]
    results['n_observations'] = len(common_idx)

    # Compute correlations with ALL features
    feature_cols = ['adx', 'volatility', 'rsi', 'future_return_1d', 'future_return_5d']

    for feat in feature_cols:
        if feat not in features_aligned.columns:
            continue

        feat_series = features_aligned[feat].dropna()
        si_subset = si_aligned.loc[feat_series.index]

        if len(si_subset) < 30:
            continue

        # Remove any remaining NaN
        mask = ~(si_subset.isna() | feat_series.isna())
        if mask.sum() < 30:
            continue

        r, p = spearmanr(si_subset[mask], feat_series[mask])

        results['correlations'][feat] = {
            'r': float(r),
            'p': float(p),
            'n': int(mask.sum()),
        }

    results['status'] = 'success'
    return results


def analyze_lead_lag_corrected(si: pd.Series, feature: pd.Series,
                                max_lag: int = 10) -> dict:
    """
    Lead-lag analysis with proper methodology.
    Returns raw results for later FDR correction.
    """
    results = []

    for lag in range(-max_lag, max_lag + 1):
        try:
            if lag > 0:
                si_aligned = si.iloc[:-lag].values
                feat_aligned = feature.iloc[lag:].values
            elif lag < 0:
                si_aligned = si.iloc[-lag:].values
                feat_aligned = feature.iloc[:lag].values
            else:
                si_aligned = si.values
                feat_aligned = feature.values

            min_len = min(len(si_aligned), len(feat_aligned))
            si_aligned = si_aligned[:min_len]
            feat_aligned = feat_aligned[:min_len]

            mask = ~(np.isnan(si_aligned) | np.isnan(feat_aligned))
            if mask.sum() < 30:
                continue

            r, p = spearmanr(si_aligned[mask], feat_aligned[mask])

            results.append({
                'lag': lag,
                'r': float(r),
                'p': float(p),
                'n': int(mask.sum()),
            })
        except:
            continue

    return results


def analyze_thresholds_oos(train_si: pd.Series, train_returns: pd.Series,
                           test_si: pd.Series, test_returns: pd.Series,
                           cost: float) -> dict:
    """
    Find optimal threshold on TRAIN, validate on TEST.
    Includes transaction costs.
    """
    result = {
        'train_threshold': None,
        'train_return': None,
        'test_return': None,
        'validated': False,
    }

    # Check inputs
    if not isinstance(train_si, pd.Series) or not isinstance(train_returns, pd.Series):
        return result
    if not isinstance(test_si, pd.Series) or not isinstance(test_returns, pd.Series):
        return result

    if len(train_si) < 50 or len(test_si) < 20:
        return result

    # Align train data
    try:
        common_train = train_si.index.intersection(train_returns.index)
        if len(common_train) < 50:
            return result
        train_si_aligned = train_si.loc[common_train]
        train_ret_aligned = train_returns.loc[common_train]
    except Exception:
        return result

    df_train = pd.DataFrame({
        'si': train_si_aligned,
        'returns': train_ret_aligned
    }).dropna()

    if len(df_train) < 50:
        return result

    # Find optimal threshold on TRAIN
    thresholds = np.percentile(df_train['si'], [20, 40, 50, 60, 80])
    best_threshold = None
    best_train_return = -np.inf

    for thresh in thresholds:
        # Signal: 1 when SI >= threshold, 0 otherwise
        signal = (df_train['si'] >= thresh).astype(int)

        # Calculate returns with costs
        position_changes = signal.diff().abs().fillna(0)
        costs = position_changes * cost

        gross_returns = signal.shift(1).fillna(0) * df_train['returns']
        net_returns = gross_returns - costs

        total_return = (1 + net_returns).prod() - 1

        if total_return > best_train_return:
            best_train_return = total_return
            best_threshold = thresh

    if best_threshold is None:
        return result

    result['train_threshold'] = float(best_threshold)
    result['train_return'] = float(best_train_return)

    # Validate on TEST
    try:
        common_test = test_si.index.intersection(test_returns.index)
        if len(common_test) < 20:
            return result
        test_si_aligned = test_si.loc[common_test]
        test_ret_aligned = test_returns.loc[common_test]
    except Exception:
        return result

    df_test = pd.DataFrame({
        'si': test_si_aligned,
        'returns': test_ret_aligned
    }).dropna()

    if len(df_test) < 20:
        return result

    # Apply threshold from train to test
    signal = (df_test['si'] >= best_threshold).astype(int)
    position_changes = signal.diff().abs().fillna(0)
    costs = position_changes * cost

    gross_returns = signal.shift(1).fillna(0) * df_test['returns']
    net_returns = gross_returns - costs

    test_total_return = (1 + net_returns).prod() - 1

    result['test_return'] = float(test_total_return)

    # Annualize
    train_years = len(df_train) / 252
    test_years = len(df_test) / 252

    result['train_annual_return'] = float((1 + result['train_return']) ** (1/train_years) - 1) if train_years > 0 else 0
    result['test_annual_return'] = float((1 + result['test_return']) ** (1/test_years) - 1) if test_years > 0 else 0

    # Validated if test return > 0 and same sign as train
    result['validated'] = (result['test_return'] > 0) and (result['train_return'] > 0)

    return result


def analyze_quadrants_oos(train_data: pd.DataFrame, test_data: pd.DataFrame,
                          train_si: pd.Series, test_si: pd.Series,
                          cost: float) -> dict:
    """
    Quadrant analysis with out-of-sample validation.
    """
    result = {
        'train_best_quadrant': None,
        'train_spread': None,
        'test_spread': None,
        'validated': False,
    }

    # Check inputs
    if not isinstance(train_si, pd.Series) or not isinstance(test_si, pd.Series):
        return result

    try:
        # Compute features for train and test
        train_features = compute_features(train_data)
        test_features = compute_features(test_data)

        # Align train
        common_train = train_si.index.intersection(train_features.index)
        if len(common_train) < 50:
            return result

        train_df = pd.DataFrame({
            'si': train_si.reindex(common_train),
            'adx': train_features.loc[common_train, 'adx'],
            'future_return': train_features.loc[common_train, 'future_return_1d'],
        }).dropna()

        if len(train_df) < 50:
            return result

        # Find medians on TRAIN
        si_median = train_df['si'].median()
        adx_median = train_df['adx'].median()

        # Quadrant analysis on train
        si_high = train_df['si'] > si_median
        adx_high = train_df['adx'] > adx_median

        quadrants_train = {}
        for si_cond, si_name in [(si_high, 'High SI'), (~si_high, 'Low SI')]:
            for adx_cond, adx_name in [(adx_high, 'High ADX'), (~adx_high, 'Low ADX')]:
                mask = si_cond & adx_cond
                if mask.sum() > 10:
                    quadrants_train[f"{si_name} + {adx_name}"] = train_df.loc[mask, 'future_return'].mean()

        if len(quadrants_train) < 4:
            return result

        best_q = max(quadrants_train, key=quadrants_train.get)
        worst_q = min(quadrants_train, key=quadrants_train.get)
        train_spread = quadrants_train[best_q] - quadrants_train[worst_q]

        result['train_best_quadrant'] = best_q
        result['train_spread'] = float(train_spread * 252)  # Annualized

        # Apply same medians to TEST
        common_test = test_si.index.intersection(test_features.index)
        if len(common_test) < 20:
            return result

        test_df = pd.DataFrame({
            'si': test_si.reindex(common_test),
            'adx': test_features.loc[common_test, 'adx'],
            'future_return': test_features.loc[common_test, 'future_return_1d'],
        }).dropna()

        if len(test_df) < 20:
            return result

        # Use TRAIN medians on TEST data
        si_high_test = test_df['si'] > si_median
        adx_high_test = test_df['adx'] > adx_median

        quadrants_test = {}
        for si_cond, si_name in [(si_high_test, 'High SI'), (~si_high_test, 'Low SI')]:
            for adx_cond, adx_name in [(adx_high_test, 'High ADX'), (~adx_high_test, 'Low ADX')]:
                mask = si_cond & adx_cond
                if mask.sum() > 5:
                    quadrants_test[f"{si_name} + {adx_name}"] = test_df.loc[mask, 'future_return'].mean()

        if best_q in quadrants_test and worst_q in quadrants_test:
            test_spread = quadrants_test[best_q] - quadrants_test[worst_q]
            result['test_spread'] = float(test_spread * 252)
            result['validated'] = test_spread > 0  # Same direction

    except Exception as e:
        result['error'] = str(e)

    return result


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("\n" + "="*70)
    print("SI DEEP EXPLORATION - CORRECTED VERSION")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("\nFixes applied:")
    print("  1. Testing ALL assets (not just BTC)")
    print("  2. Train/Val/Test split for threshold optimization")
    print("  3. FDR correction for multiple testing")
    print("  4. Proper return calculation with costs")
    print("  5. Cross-asset consistency reporting")
    print("="*70)

    loader = DataLoaderV2()
    all_results = []
    all_lead_lag_tests = []

    # ========================================================================
    # PART 1: COLLECT DATA FROM ALL ASSETS
    # ========================================================================
    print("\n" + "="*60)
    print("PART 1: ANALYZING ALL ASSETS")
    print("="*60)

    for market_type, symbols in MARKETS.items():
        print(f"\n  [{market_type.value.upper()}]")
        cost = TRANSACTION_COSTS[market_type]

        for symbol in symbols:
            print(f"    Processing {symbol}...")

            try:
                data = loader.load(symbol, market_type)
                if len(data) < 100:
                    print(f"      ⚠️ Insufficient data ({len(data)} bars)")
                    continue

                # Split data
                train, val, test = split_data(data)
                print(f"      Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

                # Compute SI on each split SEPARATELY
                si_train = compute_si(train)
                si_val = compute_si(val)
                si_test = compute_si(test)

                if len(si_train) < 30:
                    print(f"      ⚠️ SI computation failed")
                    continue

                # Compute features
                features_train = compute_features(train)
                features_val = compute_features(val)
                features_test = compute_features(test)

                # Correlation analysis (on train only, for discovery)
                corr_results = analyze_correlations_single_asset(
                    train, si_train, features_train, symbol
                )

                # Lead-lag analysis (on train only)
                lead_lag_results = {}
                for feat in ['adx', 'volatility', 'rsi']:
                    if feat in features_train.columns:
                        common_idx = si_train.index.intersection(features_train.index)
                        si_aligned = si_train.loc[common_idx]
                        feat_aligned = features_train.loc[common_idx, feat]

                        ll_tests = analyze_lead_lag_corrected(si_aligned, feat_aligned, max_lag=10)
                        if ll_tests:
                            lead_lag_results[feat] = ll_tests
                            # Collect all p-values for FDR
                            for test in ll_tests:
                                all_lead_lag_tests.append({
                                    'asset': symbol,
                                    'feature': feat,
                                    'lag': test['lag'],
                                    'r': test['r'],
                                    'p': test['p'],
                                })

                # Threshold analysis (train -> test)
                try:
                    if 'future_return_1d' in features_train.columns and 'future_return_1d' in features_test.columns:
                        # Use returns (which is available) as the target
                        train_returns = features_train['returns'].shift(-1)  # Next-day return
                        test_returns = features_test['returns'].shift(-1)

                        threshold_results = analyze_thresholds_oos(
                            si_train, train_returns,
                            si_test, test_returns,
                            cost
                        )
                    else:
                        threshold_results = {'status': 'missing_features'}
                except Exception as e:
                    threshold_results = {'status': f'error: {e}'}

                # Quadrant analysis (train -> test)
                try:
                    quadrant_results = analyze_quadrants_oos(
                        train, test, si_train, si_test, cost
                    )
                except Exception as e:
                    quadrant_results = {'status': f'error: {e}'}

                all_results.append({
                    'asset': symbol,
                    'market_type': market_type.value,
                    'n_train': len(train),
                    'n_test': len(test),
                    'correlations': corr_results.get('correlations', {}),
                    'threshold': threshold_results,
                    'quadrant': quadrant_results,
                })

                print(f"      ✅ Complete")

            except Exception as e:
                print(f"      ❌ Error: {e}")
                continue

    # ========================================================================
    # PART 2: FDR CORRECTION FOR LEAD-LAG TESTS
    # ========================================================================
    print("\n" + "="*60)
    print("PART 2: FDR CORRECTION FOR LEAD-LAG TESTS")
    print("="*60)

    if all_lead_lag_tests:
        p_values = [t['p'] for t in all_lead_lag_tests]
        rejected, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')

        n_tests = len(p_values)
        n_significant_raw = sum(1 for p in p_values if p < 0.05)
        n_significant_fdr = sum(rejected)

        print(f"\n  Total tests: {n_tests}")
        print(f"  Significant at p<0.05 (uncorrected): {n_significant_raw} ({100*n_significant_raw/n_tests:.1f}%)")
        print(f"  Significant after FDR correction: {n_significant_fdr} ({100*n_significant_fdr/n_tests:.1f}%)")

        # Find which survive FDR
        significant_tests = []
        for i, test in enumerate(all_lead_lag_tests):
            if rejected[i]:
                significant_tests.append({
                    **test,
                    'corrected_p': float(corrected_p[i]),
                })

        print(f"\n  Significant lead-lag relationships (after FDR):")
        if significant_tests:
            for t in sorted(significant_tests, key=lambda x: abs(x['r']), reverse=True)[:10]:
                direction = "SI leads" if t['lag'] > 0 else "SI lags"
                print(f"    {t['asset']}: SI-{t['feature']} (lag={t['lag']}, r={t['r']:.3f}, FDR-p={t['corrected_p']:.4f}) → {direction}")
        else:
            print("    ⚠️ NONE survive FDR correction!")

    lead_lag_fdr = {
        'n_tests': len(all_lead_lag_tests),
        'n_significant_uncorrected': n_significant_raw if all_lead_lag_tests else 0,
        'n_significant_fdr': n_significant_fdr if all_lead_lag_tests else 0,
        'significant_tests': significant_tests if all_lead_lag_tests else [],
    }

    # ========================================================================
    # PART 3: CROSS-ASSET CONSISTENCY
    # ========================================================================
    print("\n" + "="*60)
    print("PART 3: CROSS-ASSET CONSISTENCY")
    print("="*60)

    # Check how many assets show same correlation direction
    feature_directions = {}
    for result in all_results:
        for feat, corr_info in result.get('correlations', {}).items():
            if feat not in feature_directions:
                feature_directions[feat] = {'positive': 0, 'negative': 0, 'total': 0}

            feature_directions[feat]['total'] += 1
            if corr_info['r'] > 0:
                feature_directions[feat]['positive'] += 1
            else:
                feature_directions[feat]['negative'] += 1

    print("\n  Correlation direction consistency across assets:")
    for feat, counts in feature_directions.items():
        total = counts['total']
        pos = counts['positive']
        neg = counts['negative']
        consistency = max(pos, neg) / total if total > 0 else 0
        dominant = "positive" if pos > neg else "negative"
        print(f"    {feat}: {pos}/{total} positive, {neg}/{total} negative → {consistency:.0%} consistent ({dominant})")

    # ========================================================================
    # PART 4: OUT-OF-SAMPLE VALIDATION SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("PART 4: OUT-OF-SAMPLE VALIDATION SUMMARY")
    print("="*60)

    # Threshold trading
    threshold_validated = [r for r in all_results if r.get('threshold', {}).get('validated', False)]
    threshold_failed = [r for r in all_results if r.get('threshold') and not r.get('threshold', {}).get('validated', False)]

    print(f"\n  Threshold Trading (Train → Test):")
    print(f"    Validated: {len(threshold_validated)}/{len(all_results)} assets")

    if threshold_validated:
        avg_train = np.mean([r['threshold']['train_annual_return'] for r in threshold_validated])
        avg_test = np.mean([r['threshold']['test_annual_return'] for r in threshold_validated])
        print(f"    Avg Train Annual Return: {avg_train:.1%}")
        print(f"    Avg Test Annual Return: {avg_test:.1%}")
        print(f"    Degradation: {(avg_train - avg_test) / abs(avg_train) * 100:.0f}%")
    else:
        print(f"    ⚠️ NO assets validated!")

    # Quadrant analysis
    quadrant_validated = [r for r in all_results if r.get('quadrant', {}).get('validated', False)]

    print(f"\n  Quadrant Analysis (Train → Test):")
    print(f"    Validated: {len(quadrant_validated)}/{len(all_results)} assets")

    if quadrant_validated:
        avg_train_spread = np.mean([r['quadrant']['train_spread'] for r in quadrant_validated])
        avg_test_spread = np.mean([r['quadrant']['test_spread'] for r in quadrant_validated])
        print(f"    Avg Train Spread: {avg_train_spread:.1%}")
        print(f"    Avg Test Spread: {avg_test_spread:.1%}")
    else:
        print(f"    ⚠️ NO assets validated!")

    # ========================================================================
    # PART 5: HONEST ASSESSMENT
    # ========================================================================
    print("\n" + "="*60)
    print("PART 5: HONEST ASSESSMENT")
    print("="*60)

    # Compute summary statistics
    n_assets = len(all_results)
    n_threshold_validated = len(threshold_validated)
    n_quadrant_validated = len(quadrant_validated)
    n_leadlag_fdr = lead_lag_fdr.get('n_significant_fdr', 0)

    print(f"\n  REALISTIC FINDINGS:")
    print(f"  {'='*50}")

    # Correlation consistency
    consistent_features = []
    for feat, counts in feature_directions.items():
        consistency = max(counts['positive'], counts['negative']) / counts['total']
        if consistency >= 0.7:  # At least 70% same direction
            dominant = "positive" if counts['positive'] > counts['negative'] else "negative"
            consistent_features.append((feat, dominant, consistency))

    if consistent_features:
        print(f"\n  ✅ CONSISTENT CORRELATIONS (≥70% same direction):")
        for feat, direction, cons in consistent_features:
            print(f"     SI-{feat}: {direction} ({cons:.0%} of assets)")
    else:
        print(f"\n  ⚠️ NO correlations consistent across ≥70% of assets!")

    # Lead-lag after FDR
    if n_leadlag_fdr > 0:
        print(f"\n  ✅ LEAD-LAG (after FDR): {n_leadlag_fdr} significant relationships")
    else:
        print(f"\n  ❌ LEAD-LAG: No relationships survive FDR correction")

    # Threshold trading
    if n_threshold_validated > n_assets * 0.5:
        print(f"\n  ✅ THRESHOLD TRADING: Validated on {n_threshold_validated}/{n_assets} assets")
    else:
        print(f"\n  ❌ THRESHOLD TRADING: Only {n_threshold_validated}/{n_assets} assets validated")

    # Quadrant
    if n_quadrant_validated > n_assets * 0.5:
        print(f"\n  ✅ QUADRANT STRATEGY: Validated on {n_quadrant_validated}/{n_assets} assets")
    else:
        print(f"\n  ❌ QUADRANT STRATEGY: Only {n_quadrant_validated}/{n_assets} assets validated")

    # Overall verdict
    print(f"\n  {'='*50}")
    print(f"  VERDICT:")

    strong_findings = len(consistent_features) + (1 if n_leadlag_fdr > 0 else 0)
    validated_strategies = (n_threshold_validated + n_quadrant_validated) / (2 * n_assets) if n_assets > 0 else 0

    if strong_findings >= 2 and validated_strategies >= 0.5:
        print(f"    SI shows MODERATE evidence of predictive value")
        print(f"    Some correlations are consistent, some strategies validate OOS")
    elif strong_findings >= 1 or validated_strategies >= 0.3:
        print(f"    SI shows WEAK evidence of predictive value")
        print(f"    Results are asset-specific, not universal")
    else:
        print(f"    SI shows NO ROBUST evidence of predictive value")
        print(f"    Earlier findings were likely due to BTC bull market + in-sample optimization")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    output_path = Path('results/deep_exploration_corrected/findings.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_results = {
        'timestamp': datetime.now().isoformat(),
        'n_assets_tested': n_assets,
        'methodology': {
            'train_ratio': TRAIN_RATIO,
            'val_ratio': VAL_RATIO,
            'fdr_correction': 'benjamini_hochberg',
            'includes_transaction_costs': True,
        },
        'correlation_consistency': {
            feat: {
                'direction': direction,
                'consistency': float(cons)
            } for feat, direction, cons in consistent_features
        },
        'lead_lag_fdr': lead_lag_fdr,
        'threshold_validation': {
            'n_validated': n_threshold_validated,
            'n_total': n_assets,
            'rate': n_threshold_validated / n_assets if n_assets > 0 else 0,
        },
        'quadrant_validation': {
            'n_validated': n_quadrant_validated,
            'n_total': n_assets,
            'rate': n_quadrant_validated / n_assets if n_assets > 0 else 0,
        },
        'per_asset_results': all_results,
        'verdict': 'moderate' if strong_findings >= 2 and validated_strategies >= 0.5 else
                   'weak' if strong_findings >= 1 or validated_strategies >= 0.3 else 'none',
    }

    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\n  Results saved to {output_path}")
    print("="*70 + "\n")

    return final_results


if __name__ == "__main__":
    results = main()
