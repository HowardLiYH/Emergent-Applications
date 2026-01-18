#!/usr/bin/env python3
"""
DEEP COMPREHENSIVE AUDIT

Thorough audit of ALL aspects of the project:
1. Data pipeline integrity
2. SI computation methodology
3. Feature calculation correctness
4. Correlation analysis validity
5. Strategy implementation
6. Statistical methodology
7. Previous discoveries validation
8. Code consistency
9. Results reproducibility
10. Documentation accuracy
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

# ============================================================
# AUDIT FRAMEWORK
# ============================================================

class DeepAuditResult:
    def __init__(self, category: str, name: str):
        self.category = category
        self.name = name
        self.status = "PENDING"
        self.issues = []
        self.warnings = []
        self.fixes = []
        self.details = {}

    def critical(self, issue: str):
        self.status = "CRITICAL"
        self.issues.append(f"🔴 CRITICAL: {issue}")

    def error(self, issue: str):
        if self.status != "CRITICAL":
            self.status = "ERROR"
        self.issues.append(f"❌ ERROR: {issue}")

    def warn(self, warning: str):
        if self.status not in ["CRITICAL", "ERROR"]:
            self.status = "WARNING"
        self.warnings.append(f"⚠️ {warning}")

    def passed(self):
        if self.status == "PENDING":
            self.status = "PASS"

    def fix(self, description: str):
        self.fixes.append(f"🔧 FIX: {description}")

    def to_dict(self):
        return {
            'category': self.category,
            'name': self.name,
            'status': self.status,
            'issues': self.issues,
            'warnings': self.warnings,
            'fixes': self.fixes,
            'details': self.details,
        }


def print_result(result: DeepAuditResult):
    status_icons = {
        "PASS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🔴",
        "PENDING": "⏳",
    }
    icon = status_icons.get(result.status, "❓")
    print(f"  {icon} {result.name}: {result.status}")
    for issue in result.issues[:3]:
        print(f"      {issue}")
    for warning in result.warnings[:2]:
        print(f"      {warning}")


# ============================================================
# CATEGORY 1: DATA PIPELINE
# ============================================================

def audit_data_loading():
    """Check data loading and preprocessing."""
    result = DeepAuditResult("Data Pipeline", "Data Loading")

    try:
        from src.data.loader_v2 import DataLoaderV2, MarketType
        loader = DataLoaderV2()

        # Test each asset type
        test_cases = [
            ('BTCUSDT', MarketType.CRYPTO),
            ('SPY', MarketType.STOCKS),
            ('EURUSD', MarketType.FOREX),
        ]

        for symbol, market_type in test_cases:
            data = loader.load(symbol, market_type)

            # Check required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in data.columns]
            if missing:
                result.error(f"{symbol}: Missing columns {missing}")

            # Check index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                result.error(f"{symbol}: Index is not DatetimeIndex")

            # Check no future dates
            if data.index.max() > pd.Timestamp.now() + pd.Timedelta(days=1):
                result.error(f"{symbol}: Contains future dates")

            # Check data freshness
            staleness = (pd.Timestamp.now() - data.index.max()).days
            if staleness > 30:
                result.warn(f"{symbol}: Data is {staleness} days old")

        result.passed()

    except Exception as e:
        result.critical(f"Data loading failed: {e}")

    return result


def audit_data_quality():
    """Check data quality across all assets."""
    result = DeepAuditResult("Data Pipeline", "Data Quality")

    try:
        from src.data.loader_v2 import DataLoaderV2, MarketType
        loader = DataLoaderV2()

        assets = {
            'BTCUSDT': MarketType.CRYPTO,
            'ETHUSDT': MarketType.CRYPTO,
            'SPY': MarketType.STOCKS,
            'QQQ': MarketType.STOCKS,
            'EURUSD': MarketType.FOREX,
            'GBPUSD': MarketType.FOREX,
        }

        for symbol, market_type in assets.items():
            data = loader.load(symbol, market_type)

            # Check for NaN
            nan_pct = data.isnull().sum().max() / len(data)
            if nan_pct > 0.05:
                result.error(f"{symbol}: {nan_pct:.1%} NaN values")
            elif nan_pct > 0.01:
                result.warn(f"{symbol}: {nan_pct:.2%} NaN values")

            # Check for zero prices
            zero_prices = (data['close'] <= 0).sum()
            if zero_prices > 0:
                result.error(f"{symbol}: {zero_prices} zero/negative prices")

            # Check for extreme returns
            returns = data['close'].pct_change()
            extreme = (returns.abs() > 0.5).sum()
            if extreme > 10:
                result.warn(f"{symbol}: {extreme} extreme returns (>50%)")

            # Check OHLC consistency
            ohlc_issues = ((data['high'] < data['low']) |
                          (data['high'] < data['open']) |
                          (data['high'] < data['close']) |
                          (data['low'] > data['open']) |
                          (data['low'] > data['close'])).sum()
            if ohlc_issues > 0:
                result.error(f"{symbol}: {ohlc_issues} OHLC inconsistencies")

        result.passed()

    except Exception as e:
        result.critical(f"Data quality check failed: {e}")

    return result


# ============================================================
# CATEGORY 2: SI COMPUTATION
# ============================================================

def audit_si_computation():
    """Check SI computation methodology."""
    result = DeepAuditResult("SI Computation", "Methodology")

    try:
        from src.agents.strategies_v2 import get_default_strategies
        from src.competition.niche_population_v2 import NichePopulationV2
        from src.data.loader_v2 import DataLoaderV2, MarketType

        loader = DataLoaderV2()
        data = loader.load('BTCUSDT', MarketType.CRYPTO)

        strategies = get_default_strategies('daily')
        population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')

        # Run competition
        population.run(data)
        si = population.compute_si_timeseries(data, window=7)

        # Check SI range
        if si.min() < 0:
            result.error(f"SI below 0: min={si.min():.4f}")
        if si.max() > 1:
            result.error(f"SI above 1: max={si.max():.4f}")

        # Check SI variance (should be meaningful)
        if si.std() < 0.01:
            result.warn(f"SI variance too low: std={si.std():.4f}")

        # Check SI autocorrelation (should be high - it's a state variable)
        autocorr = si.autocorr(lag=1)
        if autocorr < 0.7:
            result.warn(f"SI autocorrelation low: {autocorr:.2f}")

        # Check competition is sequential (no look-ahead)
        # Verify history length matches data
        expected_history = len(data) - population.lookback_7d
        actual_history = len(population.history)
        if abs(expected_history - actual_history) > 10:
            result.warn(f"History length mismatch: {actual_history} vs expected ~{expected_history}")

        # Check agent affinities sum to 1
        for agent in population.agents:
            aff_sum = agent.niche_affinity.sum()
            if abs(aff_sum - 1.0) > 0.01:
                result.error(f"Agent {agent.agent_id} affinities don't sum to 1: {aff_sum:.4f}")

        result.details['si_mean'] = float(si.mean())
        result.details['si_std'] = float(si.std())
        result.details['si_autocorr'] = float(autocorr)

        result.passed()

    except Exception as e:
        result.critical(f"SI computation audit failed: {e}")

    return result


def audit_si_formula():
    """Verify SI formula is implemented correctly."""
    result = DeepAuditResult("SI Computation", "Formula Verification")

    try:
        # SI = 1 - mean(normalized_entropy)
        # normalized_entropy = entropy / max_entropy
        # entropy = -sum(p * log(p))

        # Manual calculation
        test_affinities = np.array([
            [0.8, 0.1, 0.1],  # High specialization
            [0.33, 0.33, 0.34],  # Low specialization
            [0.5, 0.3, 0.2],  # Medium specialization
        ])

        def manual_si(affinities):
            entropies = []
            for aff in affinities:
                p = aff / aff.sum()
                entropy = -np.sum(p * np.log(p + 1e-10))
                max_entropy = np.log(len(p))
                normalized = entropy / max_entropy
                entropies.append(normalized)
            return 1 - np.mean(entropies)

        expected_si = manual_si(test_affinities)

        # Should be between 0 and 1
        if expected_si < 0 or expected_si > 1:
            result.error(f"Manual SI calculation out of range: {expected_si:.4f}")

        # High specialization agent should contribute to higher SI
        high_spec_entropy = -np.sum(test_affinities[0] * np.log(test_affinities[0] + 1e-10))
        low_spec_entropy = -np.sum(test_affinities[1] * np.log(test_affinities[1] + 1e-10))

        if high_spec_entropy >= low_spec_entropy:
            result.error("Entropy calculation wrong: high spec should have lower entropy")

        result.details['manual_si'] = float(expected_si)
        result.details['high_spec_entropy'] = float(high_spec_entropy)
        result.details['low_spec_entropy'] = float(low_spec_entropy)

        result.passed()

    except Exception as e:
        result.critical(f"SI formula verification failed: {e}")

    return result


# ============================================================
# CATEGORY 3: FEATURE CALCULATIONS
# ============================================================

def audit_feature_calculations():
    """Verify technical indicator calculations."""
    result = DeepAuditResult("Features", "Technical Indicators")

    try:
        from src.data.loader_v2 import DataLoaderV2, MarketType
        loader = DataLoaderV2()
        data = loader.load('SPY', MarketType.STOCKS)

        close = data['close']
        high = data['high']
        low = data['low']

        # Test RSI calculation
        def calc_rsi(close, period=14):
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            return 100 - (100 / (1 + rs))

        rsi = calc_rsi(close)

        # RSI should be 0-100
        if rsi.dropna().min() < 0 or rsi.dropna().max() > 100:
            result.error(f"RSI out of range: [{rsi.min():.1f}, {rsi.max():.1f}]")

        # Test ATR calculation
        def calc_atr(data, period=14):
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return tr.rolling(period).mean()

        atr = calc_atr(data)

        # ATR should be positive
        if (atr.dropna() < 0).any():
            result.error("ATR has negative values")

        # ATR should be reasonable (< 20% of price typically)
        atr_pct = (atr / close).dropna()
        if atr_pct.mean() > 0.2:
            result.warn(f"ATR unusually high: {atr_pct.mean():.1%} of price")

        # Test Bollinger Bands
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std

        # Upper should be above lower
        if (upper < lower).any():
            result.error("Bollinger bands inverted")

        # Price should mostly be within bands
        within_bands = ((close >= lower) & (close <= upper)).mean()
        if within_bands < 0.9:
            result.warn(f"Only {within_bands:.1%} of prices within Bollinger bands (expected ~95%)")

        result.passed()

    except Exception as e:
        result.critical(f"Feature calculation audit failed: {e}")

    return result


def audit_adx_calculation():
    """Specifically audit ADX as it's key to SI correlation."""
    result = DeepAuditResult("Features", "ADX Calculation")

    try:
        from src.data.loader_v2 import DataLoaderV2, MarketType
        loader = DataLoaderV2()
        data = loader.load('SPY', MarketType.STOCKS)

        def calc_adx(data, period=14):
            high, low, close = data['high'], data['low'], data['close']

            # +DM and -DM
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.clip(lower=0)
            minus_dm = minus_dm.clip(lower=0)

            # When +DM > -DM, -DM = 0 and vice versa
            plus_dm[plus_dm <= minus_dm] = 0
            minus_dm[minus_dm <= plus_dm] = 0

            # True Range
            tr = pd.concat([
                high - low,
                abs(high - close.shift()),
                abs(low - close.shift())
            ], axis=1).max(axis=1)

            # Smoothed
            atr = tr.rolling(period).sum()
            plus_di = 100 * plus_dm.rolling(period).sum() / atr
            minus_di = 100 * minus_dm.rolling(period).sum() / atr

            # DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = dx.rolling(period).mean()

            return adx

        adx = calc_adx(data)

        # ADX should be 0-100
        adx_clean = adx.dropna()
        if adx_clean.min() < 0:
            result.error(f"ADX below 0: min={adx_clean.min():.1f}")
        if adx_clean.max() > 100:
            result.error(f"ADX above 100: max={adx_clean.max():.1f}")

        # ADX distribution check
        adx_mean = adx_clean.mean()
        if adx_mean < 10 or adx_mean > 50:
            result.warn(f"ADX mean unusual: {adx_mean:.1f} (expected 15-30)")

        result.details['adx_mean'] = float(adx_mean)
        result.details['adx_std'] = float(adx_clean.std())

        result.passed()

    except Exception as e:
        result.critical(f"ADX audit failed: {e}")

    return result


# ============================================================
# CATEGORY 4: CORRELATION ANALYSIS
# ============================================================

def audit_correlation_methodology():
    """Check correlation analysis methodology."""
    result = DeepAuditResult("Correlations", "Methodology")

    try:
        # Check for common correlation issues

        # 1. Spurious correlations from non-stationarity
        from src.data.loader_v2 import DataLoaderV2, MarketType
        loader = DataLoaderV2()
        data = loader.load('BTCUSDT', MarketType.CRYPTO)

        from src.agents.strategies_v2 import get_default_strategies
        from src.competition.niche_population_v2 import NichePopulationV2

        strategies = get_default_strategies('daily')
        population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
        population.run(data)
        si = population.compute_si_timeseries(data, window=7)

        # Test stationarity of SI
        from scipy.stats import pearsonr

        # Check if SI is stationary (ADF test approximation)
        si_diff = si.diff().dropna()
        si_autocorr_diff = si_diff.autocorr(lag=1)

        if abs(si_autocorr_diff) > 0.9:
            result.warn("SI differences highly autocorrelated - may need further differencing")

        # 2. Check for look-ahead in correlations
        # (This would require checking the analysis code)

        # 3. Sample size adequacy
        n_obs = len(si.dropna())
        min_required = 100
        if n_obs < min_required:
            result.error(f"Insufficient observations: {n_obs} < {min_required}")

        result.details['n_observations'] = n_obs
        result.details['si_diff_autocorr'] = float(si_autocorr_diff)

        result.passed()

    except Exception as e:
        result.critical(f"Correlation methodology audit failed: {e}")

    return result


def audit_si_adx_correlation():
    """Verify the key SI-ADX correlation finding."""
    result = DeepAuditResult("Correlations", "SI-ADX Relationship")

    try:
        from src.data.loader_v2 import DataLoaderV2, MarketType
        from src.agents.strategies_v2 import get_default_strategies
        from src.competition.niche_population_v2 import NichePopulationV2
        from scipy.stats import spearmanr, pearsonr

        loader = DataLoaderV2()

        assets = ['BTCUSDT', 'SPY', 'EURUSD']
        correlations = []

        for asset in assets:
            market_type = {
                'BTCUSDT': MarketType.CRYPTO,
                'SPY': MarketType.STOCKS,
                'EURUSD': MarketType.FOREX,
            }[asset]

            data = loader.load(asset, market_type)

            # Compute SI
            strategies = get_default_strategies('daily')
            population = NichePopulationV2(strategies, n_agents_per_strategy=3, frequency='daily')
            population.run(data)
            si = population.compute_si_timeseries(data, window=7)

            # Compute ADX
            def calc_adx(data, period=14):
                high, low, close = data['high'], data['low'], data['close']
                plus_dm = high.diff().clip(lower=0)
                minus_dm = (-low.diff()).clip(lower=0)
                tr = pd.concat([
                    high - low,
                    abs(high - close.shift()),
                    abs(low - close.shift())
                ], axis=1).max(axis=1)
                atr = tr.rolling(period).sum()
                plus_di = 100 * plus_dm.rolling(period).sum() / (atr + 1e-10)
                minus_di = 100 * minus_dm.rolling(period).sum() / (atr + 1e-10)
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
                return dx.rolling(period).mean()

            adx = calc_adx(data) / 100  # Normalize to 0-1

            # Align
            common = si.index.intersection(adx.dropna().index)
            si_aligned = si.loc[common]
            adx_aligned = adx.loc[common]

            # Compute correlation
            corr, pval = spearmanr(si_aligned, adx_aligned)
            correlations.append({
                'asset': asset,
                'correlation': corr,
                'p_value': pval,
                'n': len(common),
            })

        # Check consistency
        signs = [c['correlation'] > 0 for c in correlations]
        if not all(signs) and not all(not s for s in signs):
            result.warn("SI-ADX correlation sign inconsistent across assets")

        # Check significance
        significant = [c['p_value'] < 0.05 for c in correlations]
        if not all(significant):
            result.warn("SI-ADX correlation not significant in all assets")

        result.details['correlations'] = correlations

        result.passed()

    except Exception as e:
        result.critical(f"SI-ADX correlation audit failed: {e}")

    return result


# ============================================================
# CATEGORY 5: STRATEGY IMPLEMENTATION
# ============================================================

def audit_position_shifting():
    """Verify positions are shifted to avoid look-ahead."""
    result = DeepAuditResult("Strategies", "Position Shifting")

    try:
        # Read the v2 test file
        v2_path = Path("experiments/test_all_applications_v2.py")
        if not v2_path.exists():
            result.error("test_all_applications_v2.py not found")
            return result

        with open(v2_path, 'r') as f:
            code = f.read()

        # Check for position shifting patterns
        strategies = [
            'risk_budgeting',
            'spread',
            'factor_timing',
            'regime_rebalance',
            'entry_timing',
        ]

        # Count shift(1) usage
        shift_count = code.count('.shift(1)')
        position_count = code.count('position')

        if shift_count < 10:
            result.warn(f"Low shift(1) count ({shift_count}) - verify all positions shifted")

        # Check for gross_returns = position * returns (wrong) vs position.shift(1) * returns (right)
        if 'position * returns' in code and 'position.shift' not in code:
            result.error("Found 'position * returns' without shifting")

        # Check for signal lag
        if 'si_rank.shift(1)' in code:
            result.details['si_lagged'] = True
        else:
            result.warn("SI rank may not be lagged")

        result.passed()

    except Exception as e:
        result.critical(f"Position shifting audit failed: {e}")

    return result


def audit_cost_application():
    """Verify transaction costs applied correctly."""
    result = DeepAuditResult("Strategies", "Transaction Costs")

    try:
        v2_path = Path("experiments/test_all_applications_v2.py")
        with open(v2_path, 'r') as f:
            code = f.read()

        # Check costs are applied to position changes
        if 'position_changes' in code or 'diff().abs()' in code:
            result.details['cost_on_changes'] = True
        else:
            result.warn("Costs may not be applied to position changes only")

        # Check cost values are reasonable
        if '0.0004' in code:  # 4 bps for crypto
            result.details['crypto_cost'] = 0.0004
        if '0.0002' in code:  # 2 bps for stocks
            result.details['stock_cost'] = 0.0002
        if '0.0001' in code:  # 1 bp for forex
            result.details['forex_cost'] = 0.0001

        # Check for round-trip consideration
        # (Costs are applied on each position change, so entry+exit counted)

        result.passed()

    except Exception as e:
        result.critical(f"Cost application audit failed: {e}")

    return result


# ============================================================
# CATEGORY 6: STATISTICAL METHODOLOGY
# ============================================================

def audit_train_test_split():
    """Verify proper temporal train/test splitting."""
    result = DeepAuditResult("Statistics", "Train/Test Split")

    try:
        v2_path = Path("experiments/test_all_applications_v2.py")
        with open(v2_path, 'r') as f:
            code = f.read()

        # Check for temporal split
        if 'train_end' in code and 'val_end' in code:
            result.details['has_val_set'] = True
        else:
            result.warn("No validation set detected")

        # Check split ratios
        if 'TRAIN_RATIO = 0.60' in code:
            result.details['train_ratio'] = 0.60
        if 'VAL_RATIO = 0.20' in code:
            result.details['val_ratio'] = 0.20
        if 'TEST_RATIO = 0.20' in code:
            result.details['test_ratio'] = 0.20

        # Verify no shuffle (would break temporal order)
        if 'shuffle' in code.lower() and 'shuffle=False' not in code:
            result.error("Data may be shuffled, breaking temporal order")

        result.passed()

    except Exception as e:
        result.critical(f"Train/test split audit failed: {e}")

    return result


def audit_multiple_testing():
    """Check multiple testing correction."""
    result = DeepAuditResult("Statistics", "Multiple Testing")

    try:
        v2_path = Path("experiments/test_all_applications_v2.py")
        with open(v2_path, 'r') as f:
            code = f.read()

        # Check for FDR correction
        if 'benjamini_hochberg' in code.lower() or 'fdr' in code.lower():
            result.details['fdr_correction'] = True
        else:
            result.warn("No FDR correction detected")

        # Check for Bonferroni (more conservative)
        if 'bonferroni' in code.lower():
            result.details['bonferroni'] = True

        # Check number of tests
        # 5 strategies × 6 assets = 30 tests
        result.details['estimated_tests'] = 30

        result.passed()

    except Exception as e:
        result.critical(f"Multiple testing audit failed: {e}")

    return result


# ============================================================
# CATEGORY 7: PREVIOUS DISCOVERIES
# ============================================================

def audit_master_findings():
    """Verify claims in MASTER_FINDINGS.md."""
    result = DeepAuditResult("Discoveries", "Master Findings")

    try:
        findings_path = Path("docs/MASTER_FINDINGS.md")
        if not findings_path.exists():
            result.warn("MASTER_FINDINGS.md not found")
            return result

        with open(findings_path, 'r') as f:
            content = f.read()

        # Check for key claims
        claims_to_verify = [
            'SI-ADX correlation',
            'cointegration',
            'mean reversion',
            'regime',
        ]

        found_claims = []
        for claim in claims_to_verify:
            if claim.lower() in content.lower():
                found_claims.append(claim)

        result.details['documented_claims'] = found_claims

        # Check if claims have supporting evidence
        if 'p-value' in content.lower() or 'significant' in content.lower():
            result.details['has_statistical_evidence'] = True
        else:
            result.warn("No statistical evidence in findings document")

        result.passed()

    except Exception as e:
        result.critical(f"Master findings audit failed: {e}")

    return result


# ============================================================
# CATEGORY 8: RESULTS REPRODUCIBILITY
# ============================================================

def audit_reproducibility():
    """Check if results are reproducible."""
    result = DeepAuditResult("Reproducibility", "Random Seeds")

    try:
        # Check for random seed setting
        files_to_check = [
            "experiments/test_all_applications_v2.py",
            "src/competition/niche_population_v2.py",
        ]

        seed_found = False
        for filepath in files_to_check:
            path = Path(filepath)
            if path.exists():
                with open(path, 'r') as f:
                    code = f.read()
                if 'seed' in code.lower() or 'random_state' in code.lower():
                    seed_found = True
                    break

        if not seed_found:
            result.warn("No random seed setting found - results may not be reproducible")

        # Check for saved results
        results_path = Path("results/application_testing_v2/full_results.json")
        if results_path.exists():
            result.details['results_saved'] = True
        else:
            result.warn("No saved results found")

        result.passed()

    except Exception as e:
        result.critical(f"Reproducibility audit failed: {e}")

    return result


# ============================================================
# CATEGORY 9: WALK-FORWARD VALIDATION
# ============================================================

def audit_walk_forward():
    """Verify walk-forward validation implementation."""
    result = DeepAuditResult("Validation", "Walk-Forward")

    try:
        v2_path = Path("experiments/test_all_applications_v2.py")
        with open(v2_path, 'r') as f:
            code = f.read()

        # Check for walk-forward function
        if 'walk_forward_validation' in code:
            result.details['has_walk_forward'] = True
        else:
            result.warn("No walk-forward validation function found")

        # Check window sizes
        if 'train_size' in code:
            result.details['train_window_defined'] = True
        if 'test_size' in code:
            result.details['test_window_defined'] = True

        # Check for expanding vs rolling
        if 'expanding' in code.lower():
            result.details['window_type'] = 'expanding'
        elif 'rolling' in code.lower():
            result.details['window_type'] = 'rolling'

        result.passed()

    except Exception as e:
        result.critical(f"Walk-forward audit failed: {e}")

    return result


# ============================================================
# CATEGORY 10: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def audit_bootstrap_ci():
    """Verify bootstrap CI implementation."""
    result = DeepAuditResult("Statistics", "Bootstrap CIs")

    try:
        v2_path = Path("experiments/test_all_applications_v2.py")
        with open(v2_path, 'r') as f:
            code = f.read()

        # Check for bootstrap function
        if 'bootstrap_ci' in code:
            result.details['has_bootstrap'] = True
        else:
            result.warn("No bootstrap CI function found")

        # Check number of bootstrap samples
        if 'N_BOOTSTRAP = 1000' in code or 'n_boot: int = 1000' in code:
            result.details['n_bootstrap'] = 1000
        elif 'N_BOOTSTRAP' in code:
            result.details['n_bootstrap'] = 'defined'
        else:
            result.warn("Bootstrap sample count not clearly defined")

        # Check for percentile method
        if 'percentile' in code:
            result.details['percentile_method'] = True

        result.passed()

    except Exception as e:
        result.critical(f"Bootstrap CI audit failed: {e}")

    return result


# ============================================================
# MAIN AUDIT RUNNER
# ============================================================

def main():
    print("\n" + "="*80)
    print("  DEEP COMPREHENSIVE AUDIT")
    print("="*80)
    print(f"  Time: {datetime.now().isoformat()}")

    all_results = []

    # Category 1: Data Pipeline
    print("\n" + "-"*80)
    print("  CATEGORY 1: DATA PIPELINE")
    print("-"*80)

    for audit_func in [audit_data_loading, audit_data_quality]:
        result = audit_func()
        all_results.append(result)
        print_result(result)

    # Category 2: SI Computation
    print("\n" + "-"*80)
    print("  CATEGORY 2: SI COMPUTATION")
    print("-"*80)

    for audit_func in [audit_si_computation, audit_si_formula]:
        result = audit_func()
        all_results.append(result)
        print_result(result)

    # Category 3: Features
    print("\n" + "-"*80)
    print("  CATEGORY 3: FEATURE CALCULATIONS")
    print("-"*80)

    for audit_func in [audit_feature_calculations, audit_adx_calculation]:
        result = audit_func()
        all_results.append(result)
        print_result(result)

    # Category 4: Correlations
    print("\n" + "-"*80)
    print("  CATEGORY 4: CORRELATION ANALYSIS")
    print("-"*80)

    for audit_func in [audit_correlation_methodology, audit_si_adx_correlation]:
        result = audit_func()
        all_results.append(result)
        print_result(result)

    # Category 5: Strategies
    print("\n" + "-"*80)
    print("  CATEGORY 5: STRATEGY IMPLEMENTATION")
    print("-"*80)

    for audit_func in [audit_position_shifting, audit_cost_application]:
        result = audit_func()
        all_results.append(result)
        print_result(result)

    # Category 6: Statistics
    print("\n" + "-"*80)
    print("  CATEGORY 6: STATISTICAL METHODOLOGY")
    print("-"*80)

    for audit_func in [audit_train_test_split, audit_multiple_testing]:
        result = audit_func()
        all_results.append(result)
        print_result(result)

    # Category 7: Discoveries
    print("\n" + "-"*80)
    print("  CATEGORY 7: PREVIOUS DISCOVERIES")
    print("-"*80)

    result = audit_master_findings()
    all_results.append(result)
    print_result(result)

    # Category 8: Reproducibility
    print("\n" + "-"*80)
    print("  CATEGORY 8: REPRODUCIBILITY")
    print("-"*80)

    result = audit_reproducibility()
    all_results.append(result)
    print_result(result)

    # Category 9: Walk-Forward
    print("\n" + "-"*80)
    print("  CATEGORY 9: WALK-FORWARD VALIDATION")
    print("-"*80)

    result = audit_walk_forward()
    all_results.append(result)
    print_result(result)

    # Category 10: Bootstrap
    print("\n" + "-"*80)
    print("  CATEGORY 10: BOOTSTRAP CIs")
    print("-"*80)

    result = audit_bootstrap_ci()
    all_results.append(result)
    print_result(result)

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n" + "="*80)
    print("  AUDIT SUMMARY")
    print("="*80)

    status_counts = {}
    for r in all_results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    total = len(all_results)
    print(f"\n  Total Audits: {total}")
    for status, count in sorted(status_counts.items()):
        icon = {"PASS": "✅", "WARNING": "⚠️", "ERROR": "❌", "CRITICAL": "🔴"}.get(status, "❓")
        print(f"  {icon} {status}: {count}")

    # List all issues
    issues = [i for r in all_results for i in r.issues]
    warnings = [w for r in all_results for w in r.warnings]

    if issues:
        print("\n" + "-"*80)
        print("  ISSUES TO FIX")
        print("-"*80)
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    if warnings:
        print("\n" + "-"*80)
        print("  WARNINGS")
        print("-"*80)
        for i, warning in enumerate(warnings[:10], 1):
            print(f"  {i}. {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")

    # Overall verdict
    print("\n" + "="*80)
    if any(r.status == "CRITICAL" for r in all_results):
        print("  🔴 VERDICT: CRITICAL ISSUES FOUND - MUST FIX BEFORE PROCEEDING")
    elif any(r.status == "ERROR" for r in all_results):
        print("  ❌ VERDICT: ERRORS FOUND - SHOULD FIX")
    elif any(r.status == "WARNING" for r in all_results):
        print("  ⚠️ VERDICT: WARNINGS FOUND - REVIEW RECOMMENDED")
    else:
        print("  ✅ VERDICT: ALL AUDITS PASSED")
    print("="*80 + "\n")

    # Save results
    out_path = Path("results/deep_audit/audit_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': total,
                'status_counts': status_counts,
                'n_issues': len(issues),
                'n_warnings': len(warnings),
            },
            'audits': [r.to_dict() for r in all_results],
        }, f, indent=2)

    print(f"  Report saved: {out_path}")

    return all_results

if __name__ == "__main__":
    main()
