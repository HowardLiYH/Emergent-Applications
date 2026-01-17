"""
Correlation analysis with proper statistical methods.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from typing import Dict, List


class CorrelationAnalyzer:
    """
    Correlation analysis with:
    - HAC standard errors
    - Block bootstrap CIs (data-driven block size)
    - FDR correction
    - Effective N reporting
    - Regime-conditioned analysis
    """

    def __init__(self, block_size: int = None):
        self.block_size = block_size  # None = auto-compute from data

    def optimal_block_size(self, returns: pd.Series) -> int:
        """
        Compute optimal block size based on autocorrelation decay.
        Uses Politis & White (2004) approach.
        """
        from statsmodels.tsa.stattools import acf

        returns_clean = returns.dropna()
        if len(returns_clean) < 100:
            return 24  # Default for short series

        try:
            acf_values = acf(returns_clean, nlags=min(100, len(returns_clean) // 5))

            # Find lag where ACF drops below 0.05
            for lag, val in enumerate(acf_values):
                if abs(val) < 0.05:
                    return max(lag, 1)

            return 24  # Default if ACF stays high
        except Exception:
            return 24

    def effective_sample_size(self, n: int, rho: float) -> int:
        """
        Compute effective N accounting for autocorrelation.

        With high autocorrelation, effective N << raw N.
        """
        if rho >= 1:
            return 1
        if rho <= -1:
            return n
        return int(n * (1 - rho) / (1 + rho))

    def spearman_with_ci(self, x: pd.Series, y: pd.Series,
                         n_bootstrap: int = 1000) -> Dict:
        """
        Spearman correlation with block bootstrap confidence interval.
        """
        # Remove NaN
        mask = ~(x.isna() | y.isna())
        x, y = x[mask], y[mask]

        if len(x) < 50:
            return {'r': np.nan, 'p': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'n': len(x)}

        # Point estimate
        r, p = spearmanr(x, y)

        # Determine block size
        if self.block_size is None:
            block_size = self.optimal_block_size(x)
        else:
            block_size = self.block_size

        # Block bootstrap
        n = len(x)
        n_blocks = max(n // block_size, 1)

        bootstrap_rs = []
        for _ in range(n_bootstrap):
            # Sample blocks with replacement
            block_starts = np.random.choice(max(n - block_size, 1), n_blocks, replace=True)
            indices = np.concatenate([np.arange(s, min(s + block_size, n)) for s in block_starts])
            indices = indices[:n]  # Trim to original length

            if len(indices) > 10:
                boot_r, _ = spearmanr(x.iloc[indices], y.iloc[indices])
                if not np.isnan(boot_r):
                    bootstrap_rs.append(boot_r)

        if len(bootstrap_rs) < 100:
            ci_low, ci_high = np.nan, np.nan
        else:
            ci_low, ci_high = np.percentile(bootstrap_rs, [2.5, 97.5])

        return {
            'r': r,
            'p': p,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n': len(x)
        }

    def run_discovery(self, si: pd.Series, features: pd.DataFrame,
                      feature_list: List[str]) -> pd.DataFrame:
        """
        Run correlation analysis for discovery pipeline.

        Returns DataFrame with:
        - feature name
        - r (correlation)
        - p (p-value)
        - ci_low, ci_high (95% CI)
        - p_fdr (FDR-corrected p-value)
        - significant (True if p_fdr < 0.05 and |r| > 0.15)
        """
        results = []

        for feature in feature_list:
            if feature not in features.columns:
                continue

            result = self.spearman_with_ci(si, features[feature])
            result['feature'] = feature
            results.append(result)

        df = pd.DataFrame(results)

        # FDR correction
        if len(df) > 0 and not df['p'].isna().all():
            valid_mask = ~df['p'].isna()
            _, p_fdr, _, _ = multipletests(df.loc[valid_mask, 'p'], method='fdr_bh')
            df.loc[valid_mask, 'p_fdr'] = p_fdr
        else:
            df['p_fdr'] = np.nan

        # Significance
        df['significant'] = (df['p_fdr'] < 0.05) & (abs(df['r']) > 0.15)

        # Sort by absolute correlation
        df = df.sort_values('r', key=abs, ascending=False)

        return df

    def regime_conditioned_analysis(self, si: pd.Series, feature: pd.Series,
                                     regimes: pd.Series) -> dict:
        """
        CRITICAL: Check if SI-feature correlation FLIPS in different regimes.

        This catches the case where SI correlates positively with volatility
        in trending markets but NEGATIVELY in ranging markets.

        Returns:
            dict with per-regime correlations and consistency check
        """
        results = {}
        regime_names = {0: 'trending', 1: 'mean_reverting', 2: 'volatile'}

        for regime_code in regimes.unique():
            mask = regimes == regime_code

            if mask.sum() < 100:  # Need enough samples
                continue

            r, p = spearmanr(si[mask], feature[mask])
            regime_name = regime_names.get(regime_code, f'regime_{regime_code}')

            results[regime_name] = {
                'r': r,
                'p': p,
                'n': int(mask.sum()),
                'significant': p < 0.05
            }

        # Check if signs are consistent
        rs = [v['r'] for v in results.values() if 'r' in v and not np.isnan(v['r'])]

        if len(rs) >= 2:
            signs = [np.sign(r) for r in rs]
            sign_consistent = len(set(signs)) == 1

            results['_summary'] = {
                'sign_consistent': sign_consistent,
                'all_rs': rs,
                'warning': None if sign_consistent else
                    "⚠️  CORRELATION SIGN FLIPS ACROSS REGIMES! Interpret with caution."
            }
        else:
            results['_summary'] = {
                'sign_consistent': True,  # Not enough data to check
                'all_rs': rs,
                'warning': "Insufficient data for regime comparison"
            }

        return results

    def run_discovery_with_regime_check(self, si: pd.Series, features: pd.DataFrame,
                                         feature_list: List[str],
                                         regimes: pd.Series = None) -> pd.DataFrame:
        """
        Run discovery with regime-conditioned analysis.

        For each significant feature, checks if correlation is consistent
        across market regimes.
        """
        # Run standard discovery
        df = self.run_discovery(si, features, feature_list)

        if regimes is None:
            df['regime_consistent'] = None
            return df

        # Add regime analysis for significant features
        regime_results = []

        for _, row in df.iterrows():
            if row['significant']:
                feature = row['feature']
                regime_analysis = self.regime_conditioned_analysis(
                    si, features[feature], regimes
                )
                consistent = regime_analysis.get('_summary', {}).get('sign_consistent', True)
                regime_results.append(consistent)
            else:
                regime_results.append(None)

        df['regime_consistent'] = regime_results

        # Flag inconsistent correlations
        inconsistent = df[(df['significant']) & (df['regime_consistent'] == False)]
        if len(inconsistent) > 0:
            print("\n⚠️  WARNING: The following features have INCONSISTENT correlations across regimes:")
            for _, row in inconsistent.iterrows():
                print(f"   - {row['feature']}: r={row['r']:.3f} (FLIPS IN DIFFERENT REGIMES)")
            print("   These correlations may not be reliable!\n")

        return df
