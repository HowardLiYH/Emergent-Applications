"""
Regime Detection Module - Multiple Methods

Implements:
1. Rule-based (simple thresholds) - baseline
2. Hidden Markov Model (HMM) - industry standard
3. Gaussian Mixture Model (GMM) - data-driven clustering

References:
- QuantStart: HMM for market regime detection
- Two Sigma: GMM clustering for regime identification
"""
import numpy as np
import pandas as pd
from typing import Literal, Tuple, List, Dict
from abc import ABC, abstractmethod


class RegimeDetector(ABC):
    """Base class for regime detection methods."""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'RegimeDetector':
        """Fit the model to historical data."""
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict regime for each data point."""
        pass

    @abstractmethod
    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict regime probabilities (if available)."""
        pass


class RuleBasedRegimeDetector(RegimeDetector):
    """
    Simple rule-based regime detection using volatility and trend.

    Regimes:
    - 0: Mean-reverting (low trend strength)
    - 1: Trending (high trend strength)
    - 2: Volatile (extreme volatility) - optional
    """

    def __init__(self, lookback: int = 7, trend_threshold: float = 1.5,
                 vol_threshold: float = 2.0, n_regimes: int = 2):
        self.lookback = lookback
        self.trend_threshold = trend_threshold
        self.vol_threshold = vol_threshold
        self.n_regimes = n_regimes
        self.regime_names = ['mean_reverting', 'trending'] if n_regimes == 2 else ['mean_reverting', 'trending', 'volatile']

    def fit(self, data: pd.DataFrame) -> 'RuleBasedRegimeDetector':
        """No fitting needed for rule-based."""
        return self

    def _compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute trend strength and volatility features."""
        returns = data['close'].pct_change()

        # Rolling volatility
        volatility = returns.rolling(self.lookback).std()

        # Rolling cumulative return
        cum_return = data['close'].pct_change(self.lookback)

        # Trend strength = |cumulative return| / volatility
        trend_strength = cum_return.abs() / (volatility + 1e-10)

        return pd.DataFrame({
            'volatility': volatility,
            'trend_strength': trend_strength,
            'cum_return': cum_return
        }, index=data.index)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict regime using rules."""
        features = self._compute_features(data)

        regimes = pd.Series(index=data.index, dtype=int)
        regimes[:] = 0  # Default: mean-reverting

        # Trending: high trend strength
        trending_mask = features['trend_strength'] > self.trend_threshold
        regimes[trending_mask] = 1

        # Volatile (if 3 regimes): extreme volatility
        if self.n_regimes == 3:
            vol_median = features['volatility'].median()
            volatile_mask = features['volatility'] > vol_median * self.vol_threshold
            regimes[volatile_mask] = 2

        return regimes

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Rule-based gives hard labels, return one-hot probabilities."""
        regimes = self.predict(data)
        proba = pd.DataFrame(0.0, index=data.index, columns=range(self.n_regimes))
        for i, r in enumerate(regimes):
            if not pd.isna(r):
                proba.iloc[i, int(r)] = 1.0
        return proba


class HMMRegimeDetector(RegimeDetector):
    """
    Hidden Markov Model based regime detection.

    Uses returns and volatility as observations.
    Industry standard approach.
    """

    def __init__(self, n_regimes: int = 2, lookback: int = 7,
                 n_iter: int = 100, random_state: int = 42):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.regime_names = self._get_regime_names()

    def _get_regime_names(self) -> List[str]:
        if self.n_regimes == 2:
            return ['low_vol', 'high_vol']
        else:
            return ['low_vol', 'medium', 'high_vol']

    def _compute_features(self, data: pd.DataFrame) -> np.ndarray:
        """Compute features for HMM: returns and volatility."""
        returns = data['close'].pct_change().fillna(0)
        volatility = returns.rolling(self.lookback, min_periods=1).std().fillna(0)

        # Stack features: [returns, volatility]
        features = np.column_stack([returns.values, volatility.values])

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def fit(self, data: pd.DataFrame) -> 'HMMRegimeDetector':
        """Fit HMM to data."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("hmmlearn not installed. Run: pip install hmmlearn")

        features = self._compute_features(data)

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=self.random_state
        )

        self.model.fit(features)

        # Reorder states by volatility (low to high)
        # This ensures consistent labeling across assets
        self._reorder_states(features)

        return self

    def _reorder_states(self, features: np.ndarray):
        """Reorder HMM states by volatility (low vol = state 0)."""
        # Get mean volatility for each state
        states = self.model.predict(features)
        state_vols = []

        for s in range(self.n_regimes):
            mask = states == s
            if mask.sum() > 0:
                mean_vol = features[mask, 1].mean()  # volatility is column 1
            else:
                mean_vol = 0
            state_vols.append((s, mean_vol))

        # Sort by volatility
        sorted_states = sorted(state_vols, key=lambda x: x[1])

        # Create mapping: old_state -> new_state
        self.state_mapping = {old: new for new, (old, _) in enumerate(sorted_states)}

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict most likely regime sequence."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        features = self._compute_features(data)
        states = self.model.predict(features)

        # Apply reordering
        states = np.array([self.state_mapping.get(s, s) for s in states])

        return pd.Series(states, index=data.index, name='regime')

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict regime probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        features = self._compute_features(data)
        proba = self.model.predict_proba(features)

        # Reorder columns to match state ordering
        reordered = np.zeros_like(proba)
        for old, new in self.state_mapping.items():
            reordered[:, new] = proba[:, old]

        return pd.DataFrame(reordered, index=data.index,
                          columns=[f'regime_{i}' for i in range(self.n_regimes)])


class GMMRegimeDetector(RegimeDetector):
    """
    Gaussian Mixture Model based regime detection.

    Clusters data points based on multiple features.
    More flexible, data-driven approach.
    """

    def __init__(self, n_regimes: int = 2, lookback: int = 7, random_state: int = 42):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.random_state = random_state
        self.model = None
        self.regime_names = [f'cluster_{i}' for i in range(n_regimes)]

    def _compute_features(self, data: pd.DataFrame) -> np.ndarray:
        """Compute multiple features for clustering."""
        returns = data['close'].pct_change().fillna(0)
        volatility = returns.rolling(self.lookback, min_periods=1).std().fillna(0)
        momentum = data['close'].pct_change(self.lookback).fillna(0)

        # Volume if available
        if 'volume' in data.columns:
            volume_z = (data['volume'] - data['volume'].rolling(self.lookback).mean()) / \
                       (data['volume'].rolling(self.lookback).std() + 1e-10)
            volume_z = volume_z.fillna(0)
        else:
            volume_z = pd.Series(0, index=data.index)

        features = np.column_stack([
            returns.values,
            volatility.values,
            momentum.values,
            volume_z.values
        ])

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def fit(self, data: pd.DataFrame) -> 'GMMRegimeDetector':
        """Fit GMM to data."""
        from sklearn.mixture import GaussianMixture

        features = self._compute_features(data)

        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            n_init=5,
            random_state=self.random_state
        )

        self.model.fit(features)

        # Reorder by volatility
        self._reorder_states(features)

        return self

    def _reorder_states(self, features: np.ndarray):
        """Reorder GMM states by volatility."""
        states = self.model.predict(features)
        state_vols = []

        for s in range(self.n_regimes):
            mask = states == s
            if mask.sum() > 0:
                mean_vol = features[mask, 1].mean()
            else:
                mean_vol = 0
            state_vols.append((s, mean_vol))

        sorted_states = sorted(state_vols, key=lambda x: x[1])
        self.state_mapping = {old: new for new, (old, _) in enumerate(sorted_states)}

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict cluster assignment."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        features = self._compute_features(data)
        states = self.model.predict(features)
        states = np.array([self.state_mapping.get(s, s) for s in states])

        return pd.Series(states, index=data.index, name='regime')

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict cluster probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        features = self._compute_features(data)
        proba = self.model.predict_proba(features)

        reordered = np.zeros_like(proba)
        for old, new in self.state_mapping.items():
            reordered[:, new] = proba[:, old]

        return pd.DataFrame(reordered, index=data.index,
                          columns=[f'regime_{i}' for i in range(self.n_regimes)])


def get_detector(method: Literal['rule', 'hmm', 'gmm'] = 'hmm',
                 n_regimes: int = 2, **kwargs) -> RegimeDetector:
    """
    Factory function to get regime detector.

    Args:
        method: 'rule' (baseline), 'hmm' (industry standard), 'gmm' (data-driven)
        n_regimes: Number of regimes (2 or 3)
        **kwargs: Additional arguments for the detector

    Returns:
        RegimeDetector instance
    """
    if method == 'rule':
        return RuleBasedRegimeDetector(n_regimes=n_regimes, **kwargs)
    elif method == 'hmm':
        return HMMRegimeDetector(n_regimes=n_regimes, **kwargs)
    elif method == 'gmm':
        return GMMRegimeDetector(n_regimes=n_regimes, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rule', 'hmm', or 'gmm'")
