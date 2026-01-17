"""
Feature calculator for SI correlation analysis.
46 discovery features + 2 prediction features + 9 SI dynamics features
"""
import pandas as pd
import numpy as np
from typing import List


class FeatureCalculator:
    """Calculate all features for SI correlation analysis."""

    # Features that use future data (for prediction pipeline only)
    LOOKAHEAD_FEATURES = {'next_day_return', 'next_day_volatility'}

    # Features removed from discovery (circular with SI)
    CIRCULAR_FEATURES = {
        'dsi_dt', 'si_acceleration', 'si_rolling_std',
        'si_1h', 'si_4h', 'si_1d', 'si_1w', 'si_percentile',
        'strategy_concentration', 'niche_affinity_entropy'
    }

    def __init__(self):
        self.features_computed = []

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features."""
        features = pd.DataFrame(index=df.index)

        # Market features (15)
        features = self._add_market_features(df, features)

        # Risk features (10)
        features = self._add_risk_features(df, features)

        # Behavioral features (6)
        features = self._add_behavioral_features(df, features)

        # Liquidity features (4)
        features = self._add_liquidity_features(df, features)

        # Prediction features (2) - lookahead
        features = self._add_prediction_features(df, features)

        self.features_computed = list(features.columns)

        return features

    def _add_market_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add market state features."""
        returns = df['close'].pct_change()

        # Volatility at different windows
        features['volatility_24h'] = returns.rolling(24).std()
        features['volatility_7d'] = returns.rolling(168).std()
        features['volatility_30d'] = returns.rolling(720).std()

        # Trend strength (absolute return / volatility)
        features['trend_strength_7d'] = abs(returns.rolling(168).mean()) / features['volatility_7d']

        # Autocorrelation
        features['return_autocorr_7d'] = returns.rolling(168).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
        )

        # Hurst exponent (simplified)
        features['hurst_exponent'] = self._rolling_hurst(returns, window=168)

        # Return entropy
        features['return_entropy_7d'] = returns.rolling(168).apply(
            self._entropy, raw=True
        )

        # Volume features
        features['volume_24h'] = df['volume'].rolling(24).mean()
        features['volume_volatility_7d'] = df['volume'].rolling(168).std() / df['volume'].rolling(168).mean()

        # Jump frequency (returns > 3 std)
        std_7d = returns.rolling(168).std()
        features['jump_frequency_7d'] = (abs(returns) > 3 * std_7d).rolling(168).mean()

        # Variance ratio
        features['variance_ratio'] = features['volatility_7d'] / features['volatility_24h']

        # Technical indicators
        features['rsi'] = self._rsi(df['close'], 14)
        features['adx'] = self._adx(df, 14)
        features['atr'] = self._atr(df, 14)
        features['bb_width'] = self._bollinger_width(df['close'], 20)

        return features

    def _add_risk_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add risk metrics (using PAST data only)."""
        returns = df['close'].pct_change()

        # Max drawdown (rolling 30d)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.rolling(720).max()
        drawdown = (cumulative - running_max) / running_max
        features['max_drawdown_30d'] = drawdown.rolling(720).min()

        # VaR and CVaR
        features['var_95_30d'] = returns.rolling(720).quantile(0.05)
        features['cvar_95_30d'] = returns.rolling(720).apply(
            lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else np.nan
        )

        # Volatility of volatility
        vol_24h = returns.rolling(24).std()
        features['vol_of_vol_30d'] = vol_24h.rolling(720).std()

        # Tail ratio
        upper_quantile = returns.rolling(720).quantile(0.95)
        features['tail_ratio_30d'] = abs(features['var_95_30d']) / upper_quantile.replace(0, np.nan)

        # Win rate (positive returns)
        features['win_rate_30d'] = (returns > 0).rolling(720).mean()

        # Profit factor
        gains = returns.clip(lower=0).rolling(720).sum()
        losses = abs(returns.clip(upper=0).rolling(720).sum())
        features['profit_factor_30d'] = gains / losses.replace(0, np.nan)

        # Sharpe and Sortino (rolling)
        features['sharpe_ratio_30d'] = returns.rolling(720).mean() / returns.rolling(720).std() * np.sqrt(8760)
        downside = returns.clip(upper=0).rolling(720).std()
        features['sortino_ratio_30d'] = returns.rolling(720).mean() / downside * np.sqrt(8760)

        return features

    def _add_behavioral_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral/timing features."""
        returns = df['close'].pct_change()

        # Momentum vs mean-reversion returns
        features['momentum_return_7d'] = returns.rolling(168).mean()

        # Mean reversion signal (distance from MA)
        ma_30d = df['close'].rolling(720).mean()
        features['meanrev_signal'] = (df['close'] - ma_30d) / ma_30d

        # Fear/greed proxy (volatility rank)
        vol = returns.rolling(24).std()
        features['fear_greed_proxy'] = vol.rolling(720).rank(pct=True)

        # Regime duration (how long since vol regime change)
        vol_regime = (vol > vol.rolling(168).median()).astype(int)
        features['regime_duration'] = vol_regime.groupby((vol_regime != vol_regime.shift()).cumsum()).cumcount()

        return features

    def _add_liquidity_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity features."""
        returns = df['close'].pct_change()

        # Volume z-score
        features['volume_z'] = (df['volume'] - df['volume'].rolling(720).mean()) / df['volume'].rolling(720).std()

        # Amihud illiquidity (|return| / volume)
        features['amihud_log'] = np.log1p(abs(returns) / df['volume'].replace(0, np.nan) * 1e9)

        # Volume volatility
        features['volume_volatility_24h'] = df['volume'].rolling(24).std() / df['volume'].rolling(24).mean()

        # Spread proxy (high-low range / close)
        features['spread_proxy'] = (df['high'] - df['low']) / df['close']

        return features

    def _add_prediction_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add prediction features (LOOKAHEAD - separate pipeline)."""
        returns = df['close'].pct_change()

        # Next day return (24h forward)
        features['next_day_return'] = returns.shift(-24)

        # Next day volatility
        features['next_day_volatility'] = returns.rolling(24).std().shift(-24)

        return features

    def get_discovery_features(self) -> List[str]:
        """Get features for discovery pipeline (no lookahead, no circular)."""
        return [f for f in self.features_computed
                if f not in self.LOOKAHEAD_FEATURES
                and f not in self.CIRCULAR_FEATURES]

    def get_prediction_features(self) -> List[str]:
        """Get features for prediction pipeline."""
        return list(self.LOOKAHEAD_FEATURES)

    # Helper methods
    def _entropy(self, x):
        """Calculate entropy of return distribution."""
        if len(x) < 10:
            return np.nan
        x_clean = x[~np.isnan(x)]
        if len(x_clean) < 10:
            return np.nan
        hist, _ = np.histogram(x_clean, bins=10, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist + 1e-10))

    def _rolling_hurst(self, returns, window):
        """Simplified Hurst exponent estimation."""
        def hurst(x):
            x = x[~np.isnan(x)]
            if len(x) < 20:
                return np.nan
            lags = range(2, min(20, len(x) // 2))
            tau = []
            for lag in lags:
                try:
                    tau.append(np.std(np.subtract(x[lag:], x[:-lag])))
                except Exception:
                    pass
            if len(tau) < 2 or min(tau) <= 0:
                return np.nan
            try:
                reg = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
                return reg[0]
            except Exception:
                return np.nan
        return returns.rolling(window).apply(hurst, raw=True)

    def _rsi(self, prices, period=14):
        """Relative Strength Index."""
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _adx(self, df, period=14):
        """Average Directional Index (simplified)."""
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(period).mean()

    def _atr(self, df, period=14):
        """Average True Range."""
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _bollinger_width(self, prices, period=20):
        """Bollinger Band width."""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        return (upper - lower) / ma
