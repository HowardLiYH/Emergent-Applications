# Comprehensive Audit Implementation: All 8 Expert Recommendations

**Date**: January 17, 2026
**Status**: Implementing all audits

---

## 📋 Audit Checklist

| # | Audit | Priority | Status |
|---|-------|----------|--------|
| 1 | Ethical/Integrity (Pre-Registration) | 🔴 HIGH | ✅ Implemented |
| 2 | Implementation Correctness | 🔴 HIGH | ✅ Implemented |
| 3 | Causal Inference | 🔴 HIGH | ✅ Implemented |
| 4 | Strategy Validity | 🔴 HIGH | ✅ Implemented |
| 5 | Reproducibility | 🟡 MEDIUM | ✅ Implemented |
| 6 | Crypto-Specific | 🟡 MEDIUM | ✅ Implemented |
| 7 | Multi-Asset Validity | 🟡 MEDIUM | ✅ Implemented |
| 8 | Adversarial Testing | 🟡 MEDIUM | ✅ Implemented |

---

## 1️⃣ ETHICAL/INTEGRITY AUDIT: Pre-Registration

### Pre-Registration Document

```json
{
  "pre_registration": {
    "title": "Specialization Index (SI) Signal Discovery Study",
    "version": "1.0",
    "date": "2026-01-17",
    "authors": ["Research Team"],

    "primary_hypothesis": {
      "statement": "Specialization Index (SI) from agent populations correlates with market features that can inform trading decisions",
      "direction": "We expect SI to correlate positively with volatility (higher SI during volatile markets) and negatively with trend strength (higher SI during ranging markets)",
      "minimum_effect_size": 0.15,
      "significance_threshold": 0.05,
      "multiple_testing_correction": "Benjamini-Hochberg FDR"
    },

    "secondary_hypotheses": [
      {
        "id": "H2",
        "statement": "SI predicts next-day return magnitude (not direction)",
        "minimum_effect_size": 0.10
      },
      {
        "id": "H3",
        "statement": "SI velocity (dSI/dt) provides additional predictive information beyond SI level",
        "minimum_effect_size": 0.10
      }
    ],

    "methodology": {
      "data_split": {
        "train": "70% (first portion chronologically)",
        "validation": "15% (middle portion)",
        "test": "15% (final portion)",
        "no_overlap": true
      },
      "primary_si_variant": "si_rolling_7d",
      "correlation_method": "Spearman with HAC standard errors",
      "confidence_intervals": "Block bootstrap (1000 iterations)",
      "discovery_pipeline": "Train → Validate → Test",
      "prediction_pipeline": "Lagged correlations + Granger causality"
    },

    "success_criteria": {
      "discovery_success": "At least 3 features with |r| > 0.15, FDR < 0.05, confirmed in validation AND test",
      "prediction_success": "SI Granger-causes next_day_return at p < 0.05",
      "generalization_success": "Findings replicate across at least 3 of 5 major assets"
    },

    "analysis_plan": {
      "step_1": "Compute SI and all features on TRAIN set only",
      "step_2": "Run correlation discovery (46 features)",
      "step_3": "Apply FDR correction, identify candidates",
      "step_4": "Validate candidates on VAL set",
      "step_5": "Final test on TEST set",
      "step_6": "Run prediction pipeline (Granger, signal decay)",
      "step_7": "Run robustness checks",
      "step_8": "Report ALL results including nulls"
    },

    "commitment": {
      "report_all_results": true,
      "report_null_results": true,
      "no_post_hoc_hypothesis_changes": true,
      "document_all_deviations": true
    },

    "data_status": {
      "data_collected": false,
      "data_analyzed": false,
      "results_known": false
    }
  }
}
```

### Pre-Registration Commit Script

```bash
#!/bin/bash
# pre_register.sh - Run BEFORE any analysis

# Create pre-registration file
cat > experiments/pre_registration.json << 'EOF'
{
  "hypothesis": "SI correlates with volatility (r > 0.15) and predicts next-day returns",
  "primary_target": "volatility_7d",
  "secondary_targets": ["next_day_return", "trend_strength"],
  "min_effect_size": 0.15,
  "alpha": 0.05,
  "si_variant": "si_rolling_7d",
  "registered_at": "2026-01-17T00:00:00Z",
  "data_seen": false,
  "signed": "COMMIT_HASH_PLACEHOLDER"
}
EOF

# Commit with timestamp
git add experiments/pre_registration.json
git commit -m "PRE-REGISTRATION: SI hypothesis ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
COMMIT_HASH=$(git rev-parse HEAD)

# Update with actual commit hash
sed -i '' "s/COMMIT_HASH_PLACEHOLDER/$COMMIT_HASH/" experiments/pre_registration.json
git add experiments/pre_registration.json
git commit --amend -m "PRE-REGISTRATION: SI hypothesis - $COMMIT_HASH"

echo "Pre-registration complete. Commit: $COMMIT_HASH"
echo "DO NOT ANALYZE DATA UNTIL THIS IS PUSHED TO GITHUB"

git push origin main
```

---

## 2️⃣ IMPLEMENTATION CORRECTNESS AUDIT

### Unit Tests for SI Calculation

```python
# tests/test_si_calculation.py
import pytest
import numpy as np
from src.si_calculator import calculate_si, calculate_entropy

class TestSICalculation:
    """Unit tests for SI calculation correctness."""

    def test_perfect_specialists_high_si(self):
        """
        3 agents, each perfectly specialized to different niche.
        SI should be high (close to 1).
        """
        niche_affinities = np.array([
            [1.0, 0.0, 0.0],  # Agent 1: 100% niche A
            [0.0, 1.0, 0.0],  # Agent 2: 100% niche B
            [0.0, 0.0, 1.0],  # Agent 3: 100% niche C
        ])

        si = calculate_si(niche_affinities)

        # Perfect specialists → SI close to 1
        assert si > 0.9, f"Expected SI > 0.9 for perfect specialists, got {si}"

    def test_perfect_generalists_low_si(self):
        """
        3 agents, each equally distributed across all niches.
        SI should be low (close to 0).
        """
        niche_affinities = np.array([
            [0.33, 0.33, 0.34],  # Agent 1: generalist
            [0.33, 0.33, 0.34],  # Agent 2: generalist
            [0.33, 0.33, 0.34],  # Agent 3: generalist
        ])

        si = calculate_si(niche_affinities)

        # All generalists → SI close to 0
        assert si < 0.1, f"Expected SI < 0.1 for generalists, got {si}"

    def test_mixed_population(self):
        """
        Mix of specialists and generalists.
        SI should be moderate.
        """
        niche_affinities = np.array([
            [1.0, 0.0, 0.0],      # Specialist
            [0.0, 1.0, 0.0],      # Specialist
            [0.33, 0.33, 0.34],   # Generalist
        ])

        si = calculate_si(niche_affinities)

        # Mixed → SI between 0.3 and 0.8
        assert 0.3 < si < 0.8, f"Expected 0.3 < SI < 0.8 for mixed, got {si}"

    def test_single_agent_handled(self):
        """
        Single agent should return NaN or 0, not crash.
        """
        niche_affinities = np.array([[1.0, 0.0, 0.0]])

        si = calculate_si(niche_affinities)

        # Single agent → undefined, should handle gracefully
        assert np.isnan(si) or si == 0, f"Single agent should be NaN or 0, got {si}"

    def test_identical_agents_handled(self):
        """
        All identical agents should return low SI, not crash.
        """
        niche_affinities = np.array([
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ])

        si = calculate_si(niche_affinities)

        # All identical → low SI (no diversity)
        assert not np.isnan(si), "Identical agents should not produce NaN"
        assert not np.isinf(si), "Identical agents should not produce Inf"

    def test_zero_affinities_handled(self):
        """
        All-zero affinities should be handled gracefully.
        """
        niche_affinities = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

        # Should not crash
        try:
            si = calculate_si(niche_affinities)
            assert not np.isinf(si), "Zero affinities should not produce Inf"
        except ValueError:
            pass  # Acceptable to raise error for invalid input

    def test_entropy_formula_correct(self):
        """
        Test the entropy formula directly.
        """
        # Uniform distribution: entropy = log(n)
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = calculate_entropy(uniform)
        expected = np.log(4)  # ~1.386

        assert abs(entropy - expected) < 0.01, f"Entropy wrong: {entropy} != {expected}"

        # Point mass: entropy = 0
        point_mass = np.array([1.0, 0.0, 0.0, 0.0])
        entropy = calculate_entropy(point_mass)

        assert abs(entropy - 0) < 0.01, f"Point mass entropy should be 0, got {entropy}"

    def test_si_range(self):
        """
        SI should always be in [0, 1] range.
        """
        # Random test cases
        for _ in range(100):
            n_agents = np.random.randint(2, 20)
            n_niches = np.random.randint(2, 10)

            # Random affinities (normalized)
            affinities = np.random.random((n_agents, n_niches))
            affinities = affinities / affinities.sum(axis=1, keepdims=True)

            si = calculate_si(affinities)

            if not np.isnan(si):
                assert 0 <= si <= 1, f"SI out of range: {si}"
```

### Time-Travel Test (Lookahead Bug Detection)

```python
# tests/test_no_lookahead.py
import pytest
import pandas as pd
import numpy as np
from src.feature_calculator import FeatureCalculator

class TestNoLookahead:
    """Tests to ensure no features use future data."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
        np.random.seed(42)

        data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(1000) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(1000) * 0.5) + 1,
            'low': 100 + np.cumsum(np.random.randn(1000) * 0.5) - 1,
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
            'volume': np.random.randint(100, 10000, 1000),
        }, index=dates)

        return data

    def test_features_no_lookahead(self, sample_data):
        """
        Core test: Features computed with truncated data must match
        features computed with full data at the truncation point.
        """
        calc = FeatureCalculator()

        # Compute features on FULL data
        features_full = calc.compute_all(sample_data)

        # Test at multiple truncation points
        for t_idx in [500, 600, 700, 800]:
            t = sample_data.index[t_idx]

            # Truncate data at t
            data_truncated = sample_data.loc[:t].copy()

            # Compute features on truncated data
            features_truncated = calc.compute_all(data_truncated)

            # For each NON-LOOKAHEAD feature, values at t should match
            lookahead_features = ['next_day_return', 'next_day_volatility']

            for col in features_truncated.columns:
                if col in lookahead_features:
                    continue  # Skip known lookahead features

                full_val = features_full.loc[t, col]
                trunc_val = features_truncated.loc[t, col]

                # Handle NaN consistently
                if pd.isna(full_val) and pd.isna(trunc_val):
                    continue

                # Values must match
                if not np.isclose(full_val, trunc_val, rtol=1e-10, equal_nan=True):
                    pytest.fail(
                        f"LOOKAHEAD BUG in '{col}' at {t}!\n"
                        f"Full data value: {full_val}\n"
                        f"Truncated value: {trunc_val}"
                    )

    def test_lookahead_features_correctly_marked(self, sample_data):
        """
        Ensure lookahead features are properly identified.
        """
        calc = FeatureCalculator()

        # These should use future data
        expected_lookahead = {'next_day_return', 'next_day_volatility'}

        actual_lookahead = set(calc.get_lookahead_features())

        assert expected_lookahead == actual_lookahead, (
            f"Lookahead features mismatch!\n"
            f"Expected: {expected_lookahead}\n"
            f"Actual: {actual_lookahead}"
        )

    def test_rolling_windows_correct(self, sample_data):
        """
        Test that rolling windows only use past data.
        """
        calc = FeatureCalculator()

        # volatility_7d at time t should only use data from [t-7d, t]
        features = calc.compute_all(sample_data)

        for t_idx in range(168, len(sample_data), 100):  # Start after first 7 days
            t = sample_data.index[t_idx]

            # Manual calculation
            window_start = t - pd.Timedelta(days=7)
            window_data = sample_data.loc[window_start:t, 'close']
            expected_vol = window_data.pct_change().std()

            actual_vol = features.loc[t, 'volatility_7d']

            if not np.isclose(expected_vol, actual_vol, rtol=0.01, equal_nan=True):
                pytest.fail(
                    f"Rolling window may use future data at {t}!\n"
                    f"Expected: {expected_vol}\n"
                    f"Actual: {actual_vol}"
                )
```

### Numerical Stability Tests

```python
# tests/test_numerical_stability.py
import pytest
import numpy as np
import pandas as pd
from src.feature_calculator import FeatureCalculator
from src.si_calculator import calculate_si

class TestNumericalStability:
    """Tests for edge cases and numerical stability."""

    def test_zero_returns_handled(self):
        """
        Zero returns (flat market) should not produce NaN/Inf.
        """
        data = pd.DataFrame({
            'close': [100.0] * 1000,  # Flat price
            'volume': [1000] * 1000,
        })

        calc = FeatureCalculator()
        features = calc.compute_all(data)

        # volatility should be 0 or very small, not NaN
        assert not np.isnan(features['volatility_24h'].iloc[-1]), "Volatility NaN on flat data"
        assert not np.isinf(features['volatility_24h'].iloc[-1]), "Volatility Inf on flat data"

    def test_extreme_returns_handled(self):
        """
        Extreme returns (flash crash) should not break calculations.
        """
        prices = [100.0] * 500 + [10.0] + [100.0] * 499  # 90% drop then recovery
        data = pd.DataFrame({
            'close': prices,
            'volume': [1000] * 1000,
        })

        calc = FeatureCalculator()
        features = calc.compute_all(data)

        # Should produce valid (possibly large) values, not NaN/Inf
        for col in features.columns:
            has_nan = features[col].isna().sum()
            has_inf = np.isinf(features[col]).sum()

            # Some NaN is OK at the start (warm-up period)
            # But should not have Inf
            assert has_inf == 0, f"Feature '{col}' has Inf values"

    def test_very_small_values(self):
        """
        Very small values should not cause underflow.
        """
        prices = [1e-10 + i * 1e-15 for i in range(1000)]
        data = pd.DataFrame({
            'close': prices,
            'volume': [1] * 1000,
        })

        calc = FeatureCalculator()

        # Should not crash
        try:
            features = calc.compute_all(data)
        except Exception as e:
            pytest.fail(f"Small values caused crash: {e}")

    def test_very_large_values(self):
        """
        Very large values should not cause overflow.
        """
        prices = [1e15 + i * 1e10 for i in range(1000)]
        data = pd.DataFrame({
            'close': prices,
            'volume': [int(1e12)] * 1000,
        })

        calc = FeatureCalculator()

        # Should not crash or produce Inf
        try:
            features = calc.compute_all(data)
            has_inf = np.isinf(features).sum().sum()
            assert has_inf == 0, "Large values caused overflow"
        except Exception as e:
            pytest.fail(f"Large values caused crash: {e}")

    def test_missing_data_handled(self):
        """
        Missing data (NaN) should be handled gracefully.
        """
        data = pd.DataFrame({
            'close': [100.0] * 100 + [np.nan] * 10 + [100.0] * 890,
            'volume': [1000] * 1000,
        })

        calc = FeatureCalculator()

        # Should not crash
        try:
            features = calc.compute_all(data)
        except Exception as e:
            pytest.fail(f"Missing data caused crash: {e}")
```

---

## 3️⃣ CAUSAL INFERENCE AUDIT

### Bidirectional Granger Causality

```python
# src/causal_tests.py
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Dict, Tuple

class CausalInferenceAudit:
    """Tests for causal relationships between SI and features."""

    def __init__(self, max_lag: int = 24):
        self.max_lag = max_lag

    def bidirectional_granger(self, si: pd.Series, feature: pd.Series,
                               max_lag: int = None) -> Dict:
        """
        Test both directions of Granger causality.

        Returns:
            dict with 'si_causes_feature' and 'feature_causes_si'
        """
        max_lag = max_lag or self.max_lag

        # Prepare data
        df = pd.DataFrame({'si': si, 'feature': feature}).dropna()

        if len(df) < max_lag * 3:
            return {'error': 'Insufficient data'}

        results = {}

        # Test: SI → Feature
        try:
            test_1 = grangercausalitytests(
                df[['feature', 'si']],
                maxlag=max_lag,
                verbose=False
            )
            # Get minimum p-value across all lags
            p_values_1 = [test_1[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
            results['si_causes_feature'] = {
                'min_p': min(p_values_1),
                'best_lag': p_values_1.index(min(p_values_1)) + 1,
                'significant': min(p_values_1) < 0.05
            }
        except Exception as e:
            results['si_causes_feature'] = {'error': str(e)}

        # Test: Feature → SI
        try:
            test_2 = grangercausalitytests(
                df[['si', 'feature']],
                maxlag=max_lag,
                verbose=False
            )
            p_values_2 = [test_2[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
            results['feature_causes_si'] = {
                'min_p': min(p_values_2),
                'best_lag': p_values_2.index(min(p_values_2)) + 1,
                'significant': min(p_values_2) < 0.05
            }
        except Exception as e:
            results['feature_causes_si'] = {'error': str(e)}

        # Interpretation
        si_causes = results.get('si_causes_feature', {}).get('significant', False)
        feat_causes = results.get('feature_causes_si', {}).get('significant', False)

        if si_causes and not feat_causes:
            results['interpretation'] = 'SI → Feature (SI is predictive)'
        elif feat_causes and not si_causes:
            results['interpretation'] = 'Feature → SI (Feature drives SI)'
        elif si_causes and feat_causes:
            results['interpretation'] = 'Bidirectional (possibly common cause)'
        else:
            results['interpretation'] = 'No Granger causality detected'

        return results

    def placebo_test(self, real_si: pd.Series, feature: pd.Series,
                     n_placebos: int = 100) -> Dict:
        """
        Test if random SI also correlates with feature.
        If yes, our finding might be spurious.
        """
        from scipy.stats import spearmanr

        # Real correlation
        real_r, real_p = spearmanr(real_si, feature)

        # Generate placebos (random walks with similar properties)
        placebo_rs = []
        for _ in range(n_placebos):
            # Create random walk with same length and approximate variance
            steps = np.random.randn(len(real_si)) * real_si.diff().std()
            fake_si = pd.Series(np.cumsum(steps), index=real_si.index)

            r, _ = spearmanr(fake_si, feature)
            placebo_rs.append(r)

        # Calculate p-value from permutation distribution
        placebo_rs = np.array(placebo_rs)
        p_permutation = np.mean(np.abs(placebo_rs) >= np.abs(real_r))

        # Confidence interval from placebos
        ci_lower, ci_upper = np.percentile(placebo_rs, [2.5, 97.5])

        return {
            'real_r': real_r,
            'real_p': real_p,
            'placebo_mean_r': np.mean(placebo_rs),
            'placebo_std_r': np.std(placebo_rs),
            'placebo_95_ci': (ci_lower, ci_upper),
            'permutation_p': p_permutation,
            'significant': abs(real_r) > ci_upper,
            'interpretation': (
                'Real correlation exceeds random baseline'
                if abs(real_r) > ci_upper
                else 'Correlation may be spurious (within random range)'
            )
        }

    def permutation_test(self, si: pd.Series, feature: pd.Series,
                         n_permutations: int = 1000) -> Dict:
        """
        Shuffle SI labels and check if correlation persists.
        """
        from scipy.stats import spearmanr

        real_r, _ = spearmanr(si, feature)

        shuffled_rs = []
        for _ in range(n_permutations):
            shuffled_si = np.random.permutation(si.values)
            r, _ = spearmanr(shuffled_si, feature)
            shuffled_rs.append(r)

        shuffled_rs = np.array(shuffled_rs)
        p_permutation = np.mean(np.abs(shuffled_rs) >= np.abs(real_r))

        return {
            'real_r': real_r,
            'permutation_p': p_permutation,
            'significant': p_permutation < 0.05,
            'shuffled_95_ci': np.percentile(shuffled_rs, [2.5, 97.5])
        }

    def run_full_causal_audit(self, si: pd.Series, features: pd.DataFrame) -> Dict:
        """
        Run complete causal inference audit for all features.
        """
        results = {}

        for col in features.columns:
            feature = features[col]

            # Align data
            aligned = pd.concat([si, feature], axis=1).dropna()
            if len(aligned) < 100:
                results[col] = {'error': 'Insufficient data'}
                continue

            si_aligned = aligned.iloc[:, 0]
            feat_aligned = aligned.iloc[:, 1]

            results[col] = {
                'granger': self.bidirectional_granger(si_aligned, feat_aligned),
                'placebo': self.placebo_test(si_aligned, feat_aligned, n_placebos=50),
                'permutation': self.permutation_test(si_aligned, feat_aligned, n_permutations=500)
            }

            # Summary
            granger_ok = results[col]['granger'].get('si_causes_feature', {}).get('significant', False)
            placebo_ok = results[col]['placebo'].get('significant', False)
            perm_ok = results[col]['permutation'].get('significant', False)

            results[col]['summary'] = {
                'granger_supports': granger_ok,
                'placebo_supports': placebo_ok,
                'permutation_supports': perm_ok,
                'all_tests_pass': granger_ok and placebo_ok and perm_ok
            }

        return results
```

---

## 4️⃣ STRATEGY VALIDITY AUDIT

```python
# src/strategy_validation.py
import numpy as np
import pandas as pd
from typing import Dict, List

class StrategyValidityAudit:
    """Validate that strategies are realistic and not toy implementations."""

    def __init__(self, transaction_cost: float = 0.001):
        self.transaction_cost = transaction_cost

    def parameter_sensitivity(self, data: pd.DataFrame,
                               strategy_class,
                               param_name: str,
                               param_values: List) -> Dict:
        """
        Test if results are sensitive to strategy parameters.
        """
        results = {}

        for value in param_values:
            params = {param_name: value}
            strategy = strategy_class(**params)

            returns = strategy.run(data)
            sharpe = self._calculate_sharpe(returns)

            results[f'{param_name}={value}'] = {
                'total_return': returns.sum(),
                'sharpe': sharpe,
                'max_drawdown': self._calculate_max_drawdown(returns)
            }

        # Check stability
        sharpes = [r['sharpe'] for r in results.values()]
        returns_list = [r['total_return'] for r in results.values()]

        return {
            'results': results,
            'sharpe_range': max(sharpes) - min(sharpes),
            'return_range': max(returns_list) - min(returns_list),
            'stable': (max(sharpes) - min(sharpes)) < 0.5,  # Sharpe shouldn't vary by >0.5
            'interpretation': (
                'Results stable across parameters'
                if (max(sharpes) - min(sharpes)) < 0.5
                else 'WARNING: Results highly sensitive to parameters'
            )
        }

    def benchmark_comparison(self, data: pd.DataFrame,
                              strategy_returns: pd.Series) -> Dict:
        """
        Compare strategy to simple benchmarks.
        """
        # Buy and hold
        buy_hold = data['close'].pct_change()
        buy_hold_return = (1 + buy_hold).prod() - 1
        buy_hold_sharpe = self._calculate_sharpe(buy_hold)

        # Random trading
        np.random.seed(42)
        random_positions = np.random.choice([-1, 0, 1], size=len(data))
        random_returns = data['close'].pct_change() * random_positions[:-1]
        random_return = (1 + random_returns).prod() - 1
        random_sharpe = self._calculate_sharpe(random_returns)

        # Our strategy
        strategy_total = (1 + strategy_returns).prod() - 1
        strategy_sharpe = self._calculate_sharpe(strategy_returns)

        return {
            'benchmarks': {
                'buy_hold': {'return': buy_hold_return, 'sharpe': buy_hold_sharpe},
                'random': {'return': random_return, 'sharpe': random_sharpe},
            },
            'strategy': {'return': strategy_total, 'sharpe': strategy_sharpe},
            'beats_buy_hold': strategy_sharpe > buy_hold_sharpe,
            'beats_random': strategy_sharpe > random_sharpe,
            'interpretation': (
                'Strategy outperforms benchmarks'
                if strategy_sharpe > buy_hold_sharpe and strategy_sharpe > random_sharpe
                else 'WARNING: Strategy does not beat benchmarks'
            )
        }

    def transaction_cost_sensitivity(self, data: pd.DataFrame,
                                      strategy_class,
                                      cost_scenarios: Dict[str, float] = None) -> Dict:
        """
        Test profitability under different transaction costs.
        """
        if cost_scenarios is None:
            cost_scenarios = {
                'zero': 0.0,
                'maker_0.1%': 0.001,
                'taker_0.3%': 0.003,
                'adverse_0.5%': 0.005,
                'high_1.0%': 0.01,
            }

        results = {}

        for name, cost in cost_scenarios.items():
            strategy = strategy_class(transaction_cost=cost)
            returns = strategy.run(data)

            results[name] = {
                'cost': cost,
                'total_return': returns.sum(),
                'sharpe': self._calculate_sharpe(returns),
                'profitable': returns.sum() > 0
            }

        # Find breakeven cost
        for name, result in results.items():
            if not result['profitable']:
                breakeven = result['cost']
                break
        else:
            breakeven = max(cost_scenarios.values())

        return {
            'results': results,
            'breakeven_cost': breakeven,
            'realistic_profitable': results.get('taker_0.3%', {}).get('profitable', False),
            'interpretation': (
                f'Strategy profitable up to {breakeven*100:.2f}% cost'
                if breakeven > 0.001
                else 'WARNING: Strategy not profitable with realistic costs'
            )
        }

    def _calculate_sharpe(self, returns: pd.Series, rf: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        excess_returns = returns - rf / 8760  # Hourly
        return excess_returns.mean() / excess_returns.std() * np.sqrt(8760)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
```

---

## 5️⃣ REPRODUCIBILITY AUDIT

```python
# src/reproducibility.py
import json
import hashlib
import subprocess
import sys
import os
import random
import numpy as np
from datetime import datetime
from typing import Dict

class ReproducibilityManifest:
    """Create and verify reproducibility manifests."""

    @staticmethod
    def create_manifest(data_files: Dict[str, str], config: Dict) -> Dict:
        """
        Create a reproducibility manifest.

        Args:
            data_files: Dict mapping name to file path
            config: Experiment configuration
        """
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'python_version': sys.version,
                'platform': sys.platform,
                'cwd': os.getcwd(),
            },
            'git': {},
            'packages': {},
            'random_seeds': {},
            'data_hashes': {},
            'config': config,
        }

        # Git info
        try:
            manifest['git']['commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']
            ).decode().strip()
            manifest['git']['dirty'] = len(subprocess.check_output(
                ['git', 'status', '--porcelain']
            )) > 0
            manifest['git']['branch'] = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
            ).decode().strip()
        except Exception as e:
            manifest['git']['error'] = str(e)

        # Package versions
        try:
            import pkg_resources
            for pkg in ['numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn']:
                try:
                    manifest['packages'][pkg] = pkg_resources.get_distribution(pkg).version
                except:
                    pass
        except:
            pass

        # Random seeds
        try:
            manifest['random_seeds']['numpy'] = int(np.random.get_state()[1][0])
            manifest['random_seeds']['python'] = random.getstate()[1][0]
        except:
            pass

        # Data hashes
        for name, filepath in data_files.items():
            try:
                with open(filepath, 'rb') as f:
                    manifest['data_hashes'][name] = hashlib.md5(f.read()).hexdigest()
            except Exception as e:
                manifest['data_hashes'][name] = f'error: {e}'

        return manifest

    @staticmethod
    def save_manifest(manifest: Dict, filepath: str):
        """Save manifest to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

    @staticmethod
    def verify_reproducibility(manifest1: Dict, manifest2: Dict) -> Dict:
        """Compare two manifests to check reproducibility."""
        issues = []

        # Check git commit
        if manifest1.get('git', {}).get('commit') != manifest2.get('git', {}).get('commit'):
            issues.append('Git commit differs')

        # Check data hashes
        for name, hash1 in manifest1.get('data_hashes', {}).items():
            hash2 = manifest2.get('data_hashes', {}).get(name)
            if hash1 != hash2:
                issues.append(f'Data file {name} hash differs')

        # Check key packages
        for pkg in ['numpy', 'pandas', 'scipy']:
            v1 = manifest1.get('packages', {}).get(pkg)
            v2 = manifest2.get('packages', {}).get(pkg)
            if v1 != v2:
                issues.append(f'Package {pkg} version differs: {v1} vs {v2}')

        return {
            'reproducible': len(issues) == 0,
            'issues': issues
        }

def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # If using PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # If using TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
```

---

## 6️⃣ CRYPTO-SPECIFIC AUDIT

```python
# src/crypto_specific_audit.py
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict

class CryptoSpecificAudit:
    """Crypto-specific validation checks."""

    def time_of_day_effects(self, data: pd.DataFrame, si: pd.Series,
                            feature: str) -> Dict:
        """
        Check if SI-feature correlations vary by time of day.
        Crypto trades 24/7, different sessions may behave differently.
        """
        results = {}

        # Define trading sessions (UTC)
        sessions = {
            'asian': (0, 8),      # 00:00-08:00 UTC
            'european': (8, 16),  # 08:00-16:00 UTC
            'us': (16, 24),       # 16:00-24:00 UTC
        }

        data = data.copy()
        data['si'] = si
        data['hour'] = data.index.hour

        for session_name, (start, end) in sessions.items():
            subset = data[(data['hour'] >= start) & (data['hour'] < end)]

            if len(subset) < 100:
                results[session_name] = {'error': 'Insufficient data'}
                continue

            r, p = spearmanr(subset['si'], subset[feature])
            results[session_name] = {'r': r, 'p': p, 'n': len(subset)}

        # Check consistency
        rs = [v['r'] for v in results.values() if 'r' in v]
        if len(rs) >= 2:
            consistent = all(np.sign(r) == np.sign(rs[0]) for r in rs)
            range_r = max(rs) - min(rs)
        else:
            consistent = True
            range_r = 0

        return {
            'sessions': results,
            'consistent_direction': consistent,
            'range': range_r,
            'interpretation': (
                'Correlations consistent across trading sessions'
                if consistent and range_r < 0.2
                else 'WARNING: Correlations vary by trading session'
            )
        }

    def weekend_effects(self, data: pd.DataFrame, si: pd.Series,
                        feature: str) -> Dict:
        """
        Check if patterns differ on weekends.
        Traditional markets closed on weekends, crypto is not.
        """
        data = data.copy()
        data['si'] = si
        data['is_weekend'] = data.index.dayofweek >= 5

        weekday = data[~data['is_weekend']]
        weekend = data[data['is_weekend']]

        if len(weekday) < 100 or len(weekend) < 50:
            return {'error': 'Insufficient data'}

        r_weekday, p_weekday = spearmanr(weekday['si'], weekday[feature])
        r_weekend, p_weekend = spearmanr(weekend['si'], weekend[feature])

        return {
            'weekday': {'r': r_weekday, 'p': p_weekday, 'n': len(weekday)},
            'weekend': {'r': r_weekend, 'p': p_weekend, 'n': len(weekend)},
            'same_direction': np.sign(r_weekday) == np.sign(r_weekend),
            'difference': abs(r_weekday - r_weekend),
            'interpretation': (
                'Patterns consistent across weekday/weekend'
                if np.sign(r_weekday) == np.sign(r_weekend) and abs(r_weekday - r_weekend) < 0.15
                else 'WARNING: Weekend patterns differ from weekday'
            )
        }

    def liquidity_regime_check(self, data: pd.DataFrame, si: pd.Series,
                                feature: str) -> Dict:
        """
        Check if correlations change with liquidity.
        Crypto liquidity varies dramatically.
        """
        data = data.copy()
        data['si'] = si

        # Split by volume median
        median_volume = data['volume'].median()
        high_liq = data[data['volume'] > median_volume]
        low_liq = data[data['volume'] <= median_volume]

        r_high, p_high = spearmanr(high_liq['si'], high_liq[feature])
        r_low, p_low = spearmanr(low_liq['si'], low_liq[feature])

        return {
            'high_liquidity': {'r': r_high, 'p': p_high, 'n': len(high_liq)},
            'low_liquidity': {'r': r_low, 'p': p_low, 'n': len(low_liq)},
            'same_direction': np.sign(r_high) == np.sign(r_low),
            'difference': abs(r_high - r_low),
            'interpretation': (
                'Correlations hold across liquidity regimes'
                if np.sign(r_high) == np.sign(r_low) and abs(r_high - r_low) < 0.15
                else 'WARNING: Correlations depend on liquidity'
            )
        }

    def outlier_analysis(self, data: pd.DataFrame, si: pd.Series,
                         feature: str, sigma_threshold: float = 5.0) -> Dict:
        """
        Identify impact of extreme events (flash crashes, hacks, etc.)
        """
        data = data.copy()
        data['si'] = si
        data['returns'] = data['close'].pct_change()

        # Identify outliers (>5 sigma)
        returns_std = data['returns'].std()
        outliers = abs(data['returns']) > sigma_threshold * returns_std

        # Correlations with and without outliers
        r_with, p_with = spearmanr(data['si'], data[feature])

        clean_data = data[~outliers]
        r_without, p_without = spearmanr(clean_data['si'], clean_data[feature])

        return {
            'outlier_count': outliers.sum(),
            'outlier_pct': outliers.mean() * 100,
            'with_outliers': {'r': r_with, 'p': p_with},
            'without_outliers': {'r': r_without, 'p': p_without},
            'outlier_driven': abs(r_with - r_without) > 0.1,
            'interpretation': (
                'Results robust to outliers'
                if abs(r_with - r_without) < 0.1
                else 'WARNING: Results may be driven by outliers'
            )
        }

    def run_full_crypto_audit(self, data: pd.DataFrame, si: pd.Series,
                               feature: str) -> Dict:
        """Run all crypto-specific checks."""
        return {
            'time_of_day': self.time_of_day_effects(data, si, feature),
            'weekend': self.weekend_effects(data, si, feature),
            'liquidity': self.liquidity_regime_check(data, si, feature),
            'outliers': self.outlier_analysis(data, si, feature),
        }
```

---

## 7️⃣ MULTI-ASSET VALIDITY AUDIT

```python
# src/multi_asset_audit.py
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List

class MultiAssetAudit:
    """Test if findings generalize across multiple assets."""

    def __init__(self, assets: List[str] = None):
        self.assets = assets or ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']

    def cross_asset_generalization(self, data_dict: Dict[str, pd.DataFrame],
                                    si_dict: Dict[str, pd.Series],
                                    feature: str) -> Dict:
        """
        Test if SI-feature correlation holds across multiple assets.
        """
        results = {}

        for asset in self.assets:
            if asset not in data_dict or asset not in si_dict:
                results[asset] = {'error': 'Data not available'}
                continue

            data = data_dict[asset]
            si = si_dict[asset]

            if feature not in data.columns:
                results[asset] = {'error': f'Feature {feature} not in data'}
                continue

            r, p = spearmanr(si, data[feature])
            results[asset] = {
                'r': r,
                'p': p,
                'n': len(si),
                'significant': p < 0.05
            }

        # Summary statistics
        valid_results = [v for v in results.values() if 'r' in v]
        if len(valid_results) == 0:
            return {'error': 'No valid results'}

        rs = [v['r'] for v in valid_results]
        positive_count = sum(1 for r in rs if r > 0)
        significant_count = sum(1 for v in valid_results if v['significant'])

        return {
            'results': results,
            'summary': {
                'n_assets': len(valid_results),
                'positive_direction': positive_count,
                'negative_direction': len(valid_results) - positive_count,
                'significant_count': significant_count,
                'mean_r': np.mean(rs),
                'std_r': np.std(rs),
            },
            'consistent': positive_count >= len(valid_results) * 0.8 or positive_count <= len(valid_results) * 0.2,
            'mostly_significant': significant_count >= len(valid_results) * 0.5,
            'interpretation': (
                'Findings generalize across assets'
                if (positive_count >= len(valid_results) * 0.8) and (significant_count >= len(valid_results) * 0.5)
                else 'WARNING: Findings may not generalize'
            )
        }

    def time_period_robustness(self, data: pd.DataFrame, si: pd.Series,
                                feature: str, periods: Dict[str, tuple] = None) -> Dict:
        """
        Test if findings hold across different time periods.
        """
        if periods is None:
            # Auto-detect periods based on data range
            start = data.index.min()
            end = data.index.max()
            total_days = (end - start).days

            if total_days < 365:
                # Less than a year: split into quarters
                periods = {
                    'q1': (start, start + pd.Timedelta(days=total_days//4)),
                    'q2': (start + pd.Timedelta(days=total_days//4), start + pd.Timedelta(days=total_days//2)),
                    'q3': (start + pd.Timedelta(days=total_days//2), start + pd.Timedelta(days=3*total_days//4)),
                    'q4': (start + pd.Timedelta(days=3*total_days//4), end),
                }
            else:
                # Multiple years: split by year
                years = data.index.year.unique()
                periods = {f'year_{y}': (f'{y}-01-01', f'{y}-12-31') for y in years}

        results = {}

        for period_name, (start, end) in periods.items():
            mask = (data.index >= start) & (data.index <= end)
            subset = data.loc[mask]
            si_subset = si.loc[mask]

            if len(subset) < 100:
                results[period_name] = {'error': 'Insufficient data'}
                continue

            r, p = spearmanr(si_subset, subset[feature])
            results[period_name] = {'r': r, 'p': p, 'n': len(subset)}

        # Check consistency
        valid_rs = [v['r'] for v in results.values() if 'r' in v]
        if len(valid_rs) < 2:
            return {'results': results, 'error': 'Need at least 2 periods'}

        consistent = all(np.sign(r) == np.sign(valid_rs[0]) for r in valid_rs)
        range_r = max(valid_rs) - min(valid_rs)

        return {
            'results': results,
            'consistent_direction': consistent,
            'range': range_r,
            'interpretation': (
                'Findings consistent across time periods'
                if consistent and range_r < 0.25
                else 'WARNING: Findings vary across time periods'
            )
        }

    def market_regime_analysis(self, data: pd.DataFrame, si: pd.Series,
                                feature: str) -> Dict:
        """
        Test if correlations differ by market regime (bull/bear/sideways).
        """
        data = data.copy()
        data['si'] = si

        # Simple regime classification based on rolling returns
        data['returns_30d'] = data['close'].pct_change(720)  # 30 days at hourly

        # Define regimes
        data['regime'] = 'sideways'
        data.loc[data['returns_30d'] > 0.1, 'regime'] = 'bull'
        data.loc[data['returns_30d'] < -0.1, 'regime'] = 'bear'

        results = {}

        for regime in ['bull', 'bear', 'sideways']:
            subset = data[data['regime'] == regime]

            if len(subset) < 100:
                results[regime] = {'error': 'Insufficient data', 'n': len(subset)}
                continue

            r, p = spearmanr(subset['si'], subset[feature])
            results[regime] = {'r': r, 'p': p, 'n': len(subset)}

        # Check consistency
        valid_rs = [v['r'] for v in results.values() if 'r' in v]

        return {
            'results': results,
            'regime_distribution': data['regime'].value_counts().to_dict(),
            'consistent': len(set(np.sign(r) for r in valid_rs)) == 1 if valid_rs else False,
            'interpretation': (
                'Correlations consistent across market regimes'
                if len(set(np.sign(r) for r in valid_rs)) == 1
                else 'WARNING: Correlations differ by market regime'
            )
        }
```

---

## 8️⃣ ADVERSARIAL AUDIT

```python
# src/adversarial_audit.py
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List
from itertools import product

class AdversarialAudit:
    """Adversarial tests to stress-test findings."""

    def devils_advocate(self, data: pd.DataFrame, si: pd.Series,
                        feature: str) -> Dict:
        """
        Actively try to find conditions where SI-feature correlation BREAKS.
        """
        data = data.copy()
        data['si'] = si
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek

        counterexamples = []

        # Check every hour x day combination
        for hour, day in product(range(24), range(7)):
            mask = (data['hour'] == hour) & (data['dayofweek'] == day)
            subset = data.loc[mask]

            if len(subset) < 50:
                continue

            r, p = spearmanr(subset['si'], subset[feature])

            # Record if correlation is opposite direction AND significant
            overall_r = spearmanr(data['si'], data[feature])[0]

            if np.sign(r) != np.sign(overall_r) and p < 0.05:
                counterexamples.append({
                    'hour': hour,
                    'day': day,
                    'r': r,
                    'p': p,
                    'n': len(subset)
                })

        return {
            'counterexamples_found': len(counterexamples),
            'details': counterexamples,
            'robust': len(counterexamples) == 0,
            'interpretation': (
                'No counterexamples found - finding is robust'
                if len(counterexamples) == 0
                else f'WARNING: Found {len(counterexamples)} counterexamples'
            )
        }

    def random_feature_control(self, si: pd.Series, n_random: int = 100) -> Dict:
        """
        SI should NOT correlate with random features.
        If it does, something is wrong.
        """
        false_positives = []

        for i in range(n_random):
            random_feature = pd.Series(
                np.random.randn(len(si)),
                index=si.index
            )

            r, p = spearmanr(si, random_feature)

            if p < 0.05:
                false_positives.append({
                    'random_id': i,
                    'r': r,
                    'p': p
                })

        # Expected false positive rate: 5%
        expected_fp = n_random * 0.05
        actual_fp = len(false_positives)

        return {
            'n_random_features': n_random,
            'false_positives': actual_fp,
            'expected_false_positives': expected_fp,
            'ratio': actual_fp / expected_fp if expected_fp > 0 else float('inf'),
            'acceptable': actual_fp <= expected_fp * 2,  # Allow up to 2x expected
            'interpretation': (
                'False positive rate within expected range'
                if actual_fp <= expected_fp * 2
                else 'WARNING: Too many false positives - check for data issues'
            )
        }

    def permutation_test(self, si: pd.Series, feature: pd.Series,
                         n_permutations: int = 1000) -> Dict:
        """
        Shuffle SI labels. Correlation should vanish.
        """
        real_r, real_p = spearmanr(si, feature)

        shuffled_rs = []
        for _ in range(n_permutations):
            shuffled_si = np.random.permutation(si.values)
            r, _ = spearmanr(shuffled_si, feature)
            shuffled_rs.append(r)

        shuffled_rs = np.array(shuffled_rs)

        # P-value from permutation distribution
        p_permutation = np.mean(np.abs(shuffled_rs) >= np.abs(real_r))

        return {
            'real_r': real_r,
            'real_p': real_p,
            'shuffled_mean': np.mean(shuffled_rs),
            'shuffled_std': np.std(shuffled_rs),
            'shuffled_95_ci': list(np.percentile(shuffled_rs, [2.5, 97.5])),
            'permutation_p': p_permutation,
            'significant': p_permutation < 0.05,
            'interpretation': (
                'Correlation survives permutation test'
                if p_permutation < 0.05
                else 'WARNING: Correlation may be spurious'
            )
        }

    def subset_stability(self, si: pd.Series, feature: pd.Series,
                         n_subsets: int = 100,
                         subset_fraction: float = 0.5) -> Dict:
        """
        Test if correlation holds in random subsets of data.
        """
        full_r, _ = spearmanr(si, feature)

        subset_rs = []
        consistent_count = 0

        for _ in range(n_subsets):
            # Random subset
            indices = np.random.choice(
                len(si),
                size=int(len(si) * subset_fraction),
                replace=False
            )

            subset_si = si.iloc[indices]
            subset_feature = feature.iloc[indices]

            r, p = spearmanr(subset_si, subset_feature)
            subset_rs.append(r)

            # Count consistent results (same direction and significant)
            if np.sign(r) == np.sign(full_r) and p < 0.05:
                consistent_count += 1

        return {
            'full_r': full_r,
            'subset_mean_r': np.mean(subset_rs),
            'subset_std_r': np.std(subset_rs),
            'consistent_pct': consistent_count / n_subsets * 100,
            'stable': consistent_count >= n_subsets * 0.8,
            'interpretation': (
                f'Correlation stable in {consistent_count}/{n_subsets} subsets'
                if consistent_count >= n_subsets * 0.8
                else f'WARNING: Correlation unstable ({consistent_count}/{n_subsets} consistent)'
            )
        }

    def run_full_adversarial_audit(self, data: pd.DataFrame, si: pd.Series,
                                    feature: str) -> Dict:
        """Run all adversarial tests."""
        return {
            'devils_advocate': self.devils_advocate(data, si, feature),
            'random_control': self.random_feature_control(si),
            'permutation': self.permutation_test(si, data[feature]),
            'subset_stability': self.subset_stability(si, data[feature]),
        }
```

---

## 📋 Master Audit Runner

```python
# src/run_all_audits.py
import json
from datetime import datetime
from typing import Dict

from src.causal_tests import CausalInferenceAudit
from src.strategy_validation import StrategyValidityAudit
from src.reproducibility import ReproducibilityManifest, set_all_seeds
from src.crypto_specific_audit import CryptoSpecificAudit
from src.multi_asset_audit import MultiAssetAudit
from src.adversarial_audit import AdversarialAudit

def run_all_audits(data: pd.DataFrame, si: pd.Series, feature: str,
                   config: Dict) -> Dict:
    """
    Run all 8 audit categories and compile results.
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'feature_tested': feature,
        'audits': {}
    }

    print("=" * 60)
    print("RUNNING COMPREHENSIVE AUDIT SUITE")
    print("=" * 60)

    # 1. Reproducibility (run first to set seeds)
    print("\n1/8: Reproducibility Audit...")
    set_all_seeds(42)
    manifest = ReproducibilityManifest.create_manifest(
        data_files={'main_data': 'data/prices.csv'},
        config=config
    )
    results['audits']['reproducibility'] = {
        'manifest_created': True,
        'git_commit': manifest.get('git', {}).get('commit', 'unknown')
    }

    # 2. Causal Inference
    print("2/8: Causal Inference Audit...")
    causal = CausalInferenceAudit()
    results['audits']['causal'] = {
        'granger': causal.bidirectional_granger(si, data[feature]),
        'placebo': causal.placebo_test(si, data[feature]),
        'permutation': causal.permutation_test(si, data[feature])
    }

    # 3. Strategy Validity
    print("3/8: Strategy Validity Audit...")
    strategy_audit = StrategyValidityAudit()
    # Note: Would need actual strategy class
    results['audits']['strategy'] = {
        'status': 'Requires strategy implementation',
        'note': 'Run after strategies are built'
    }

    # 4. Crypto-Specific
    print("4/8: Crypto-Specific Audit...")
    crypto = CryptoSpecificAudit()
    results['audits']['crypto'] = crypto.run_full_crypto_audit(data, si, feature)

    # 5. Multi-Asset
    print("5/8: Multi-Asset Audit...")
    multi = MultiAssetAudit()
    results['audits']['multi_asset'] = multi.time_period_robustness(data, si, feature)
    results['audits']['market_regime'] = multi.market_regime_analysis(data, si, feature)

    # 6. Adversarial
    print("6/8: Adversarial Audit...")
    adversarial = AdversarialAudit()
    results['audits']['adversarial'] = adversarial.run_full_adversarial_audit(data, si, feature)

    # 7. Implementation (unit tests)
    print("7/8: Implementation Audit...")
    results['audits']['implementation'] = {
        'status': 'Run pytest tests/test_*.py',
        'note': 'Unit tests defined in test files'
    }

    # 8. Pre-Registration
    print("8/8: Pre-Registration Check...")
    try:
        with open('experiments/pre_registration.json', 'r') as f:
            prereg = json.load(f)
        results['audits']['pre_registration'] = {
            'status': 'Found',
            'date': prereg.get('registered_at', 'unknown'),
            'hypothesis': prereg.get('hypothesis', 'unknown')
        }
    except FileNotFoundError:
        results['audits']['pre_registration'] = {
            'status': 'NOT FOUND - CREATE BEFORE ANALYSIS',
            'critical': True
        }

    # Summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    issues = []

    # Check causal
    if not results['audits']['causal'].get('granger', {}).get('si_causes_feature', {}).get('significant', False):
        issues.append('Causal: SI does not Granger-cause feature')

    # Check crypto
    for check_name, check_result in results['audits']['crypto'].items():
        if 'WARNING' in check_result.get('interpretation', ''):
            issues.append(f'Crypto: {check_name} flagged')

    # Check adversarial
    if not results['audits']['adversarial']['permutation'].get('significant', False):
        issues.append('Adversarial: Permutation test failed')

    results['summary'] = {
        'total_issues': len(issues),
        'issues': issues,
        'verdict': 'PASS' if len(issues) == 0 else 'ISSUES FOUND'
    }

    print(f"\nTotal issues found: {len(issues)}")
    for issue in issues:
        print(f"  - {issue}")

    print(f"\nVerdict: {results['summary']['verdict']}")

    return results
```

---

## ✅ Summary of Implementations

| Audit | Implementation | Tests |
|-------|----------------|-------|
| 1. Pre-Registration | `pre_registration.json` template | Commit check |
| 2. Implementation | Unit tests in `tests/` | pytest |
| 3. Causal Inference | `CausalInferenceAudit` class | Granger, placebo, permutation |
| 4. Strategy Validity | `StrategyValidityAudit` class | Parameters, benchmarks, costs |
| 5. Reproducibility | `ReproducibilityManifest` class | Manifest creation/verification |
| 6. Crypto-Specific | `CryptoSpecificAudit` class | Time, weekend, liquidity, outliers |
| 7. Multi-Asset | `MultiAssetAudit` class | Cross-asset, time periods, regimes |
| 8. Adversarial | `AdversarialAudit` class | Devil's advocate, random, permutation |

---

*All 8 audits now have concrete implementations. Run `run_all_audits()` before proceeding with analysis.*

*Last Updated: January 17, 2026*
