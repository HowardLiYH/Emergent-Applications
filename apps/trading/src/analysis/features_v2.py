"""
Feature calculator for SI correlation analysis - FREQUENCY AWARE VERSION.

Fixes the issue where hourly windows were applied to daily data.
Now supports both hourly and daily frequencies with correct window sizes.
"""
import pandas as pd
import numpy as np
from typing import List, Literal


# Window mapping: logical period -> bars
WINDOW_MAP = {
    'hourly': {
        '1d': 24,      # 24 hours
        '7d': 168,     # 168 hours
        '30d': 720,    # 720 hours
        '14': 14,      # Fixed periods (RSI, ADX, ATR)
        '20': 20,      # Bollinger
    },
    'daily': {
        '1d': 1,       # 1 day
        '7d': 7,       # 7 days
        '30d': 30,     # 30 days
        '14': 14,      # Fixed periods
        '20': 20,      # Bollinger
    }
}


class FeatureCalculatorV2:
    """
    Frequency-aware feature calculator.

    Usage:
        calc = FeatureCalculatorV2(frequency='daily')
        features = calc.compute_all(df)
    """

    LOOKAHEAD_FEATURES = {'next_day_return', 'next_day_volatility'}
    CIRCULAR_FEATURES = {
        'dsi_dt', 'si_acceleration', 'si_rolling_std',
        'si_1h', 'si_4h', 'si_1d', 'si_1w', 'si_percentile',
        'strategy_concentration', 'niche_affinity_entropy'
    }

    def __init__(self, frequency: Literal['hourly', 'daily'] = 'daily'):
        self.frequency = frequency
        self.windows = WINDOW_MAP[frequency]
        self.features_computed = []

    def _w(self, period: str) -> int:
        """Get window size for a period."""
        if period in self.windows:
            return self.windows[period]
        # Try to parse as integer
        try:
            return int(period)
        except ValueError:
            # Extract number from period string like '14'
            raise ValueError(f"Unknown period: {period}. Available: {list(self.windows.keys())}")

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features with correct window sizes."""
        features = pd.DataFrame(index=df.index)

        features = self._add_market_features(df, features)
        features = self._add_risk_features(df, features)
        features = self._add_behavioral_features(df, features)
        features = self._add_liquidity_features(df, features)
        features = self._add_prediction_features(df, features)

        self.features_computed = list(features.columns)
        return features

    def _add_market_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add market state features with correct windows."""
        returns = df['close'].pct_change()

        # Volatility at different windows (FIXED)
        features['volatility_24h'] = returns.rolling(self._w('1d')).std()
        features['volatility_7d'] = returns.rolling(self._w('7d')).std()
        features['volatility_30d'] = returns.rolling(self._w('30d')).std()

        # Trend strength
        features['trend_strength_7d'] = abs(returns.rolling(self._w('7d')).mean()) / features['volatility_7d']

        # Autocorrelation
        features['return_autocorr_7d'] = returns.rolling(self._w('7d')).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
        )

        # Hurst exponent
        features['hurst_exponent'] = self._rolling_hurst(returns, window=self._w('7d'))

        # Return entropy
        features['return_entropy_7d'] = returns.rolling(self._w('7d')).apply(
            self._entropy, raw=True
        )

        # Volume features
        features['volume_24h'] = df['volume'].rolling(self._w('1d')).mean()
        vol_mean = df['volume'].rolling(self._w('7d')).mean()
        vol_std = df['volume'].rolling(self._w('7d')).std()
        features['volume_volatility_7d'] = vol_std / vol_mean.replace(0, np.nan)

        # Jump frequency
        std_7d = returns.rolling(self._w('7d')).std()
        features['jump_frequency_7d'] = (abs(returns) > 3 * std_7d).rolling(self._w('7d')).mean()

        # Variance ratio
        features['variance_ratio'] = features['volatility_7d'] / features['volatility_24h'].replace(0, np.nan)

        # Technical indicators (these use fixed periods, OK as-is)
        features['rsi'] = self._rsi(df['close'], self._w('14'))
        features['adx'] = self._adx(df, self._w('14'))
        features['atr'] = self._atr(df, self._w('14'))
        features['bb_width'] = self._bollinger_width(df['close'], self._w('20'))

        return features

    def _add_risk_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add risk metrics with correct windows."""
        returns = df['close'].pct_change()
        w_30d = self._w('30d')
        w_1d = self._w('1d')

        # Max drawdown (rolling 30d)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.rolling(w_30d).max()
        drawdown = (cumulative - running_max) / running_max.replace(0, np.nan)
        features['max_drawdown_30d'] = drawdown.rolling(w_30d).min()

        # VaR and CVaR
        features['var_95_30d'] = returns.rolling(w_30d).quantile(0.05)
        features['cvar_95_30d'] = returns.rolling(w_30d).apply(
            lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else np.nan
        )

        # Volatility of volatility
        vol_1d = returns.rolling(w_1d).std()
        features['vol_of_vol_30d'] = vol_1d.rolling(w_30d).std()

        # Tail ratio
        upper_q = returns.rolling(w_30d).quantile(0.95)
        features['tail_ratio_30d'] = abs(features['var_95_30d']) / upper_q.replace(0, np.nan)

        # Win rate
        features['win_rate_30d'] = (returns > 0).rolling(w_30d).mean()

        # Profit factor
        gains = returns.clip(lower=0).rolling(w_30d).sum()
        losses = abs(returns.clip(upper=0).rolling(w_30d).sum())
        features['profit_factor_30d'] = gains / losses.replace(0, np.nan)

        # Sharpe and Sortino (rolling)
        ann_factor = 252 if self.frequency == 'daily' else 8760
        ret_mean = returns.rolling(w_30d).mean()
        ret_std = returns.rolling(w_30d).std()
        features['sharpe_ratio_30d'] = ret_mean / ret_std * np.sqrt(ann_factor)

        downside = returns.clip(upper=0).rolling(w_30d).std()
        features['sortino_ratio_30d'] = ret_mean / downside * np.sqrt(ann_factor)

        return features

    def _add_behavioral_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral/timing features."""
        returns = df['close'].pct_change()
        w_7d = self._w('7d')
        w_30d = self._w('30d')
        w_1d = self._w('1d')

        # Momentum
        features['momentum_return_7d'] = returns.rolling(w_7d).mean()

        # Mean reversion signal
        ma_30d = df['close'].rolling(w_30d).mean()
        features['meanrev_signal'] = (df['close'] - ma_30d) / ma_30d

        # Fear/greed proxy
        vol = returns.rolling(w_1d).std()
        features['fear_greed_proxy'] = vol.rolling(w_30d).rank(pct=True)

        # Regime duration
        vol_regime = (vol > vol.rolling(w_7d).median()).astype(int)
        features['regime_duration'] = vol_regime.groupby((vol_regime != vol_regime.shift()).cumsum()).cumcount()

        return features

    def _add_liquidity_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity features."""
        returns = df['close'].pct_change()
        w_30d = self._w('30d')
        w_1d = self._w('1d')

        # Volume z-score
        vol_mean = df['volume'].rolling(w_30d).mean()
        vol_std = df['volume'].rolling(w_30d).std()
        features['volume_z'] = (df['volume'] - vol_mean) / vol_std.replace(0, np.nan)

        # Amihud illiquidity
        features['amihud_log'] = np.log1p(abs(returns) / df['volume'].replace(0, np.nan) * 1e9)

        # Volume volatility
        vol_mean_1d = df['volume'].rolling(w_1d).mean()
        vol_std_1d = df['volume'].rolling(w_1d).std()
        features['volume_volatility_24h'] = vol_std_1d / vol_mean_1d.replace(0, np.nan)

        # Spread proxy
        features['spread_proxy'] = (df['high'] - df['low']) / df['close']

        return features

    def _add_prediction_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add prediction features (LOOKAHEAD)."""
        returns = df['close'].pct_change()
        w_1d = self._w('1d')

        # Next day return (1 period forward)
        features['next_day_return'] = returns.shift(-1)

        # Next day volatility
        features['next_day_volatility'] = returns.rolling(w_1d).std().shift(-1)

        return features

    def get_discovery_features(self) -> List[str]:
        """Get features for discovery pipeline."""
        return [f for f in self.features_computed
                if f not in self.LOOKAHEAD_FEATURES
                and f not in self.CIRCULAR_FEATURES]

    def get_prediction_features(self) -> List[str]:
        """Get features for prediction pipeline."""
        return list(self.LOOKAHEAD_FEATURES)

    # Helper methods
    def _entropy(self, x):
        if len(x) < 10:
            return np.nan
        hist, _ = np.histogram(x, bins=10, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist + 1e-10))

    def _rolling_hurst(self, returns, window):
        def hurst(x):
            if len(x) < 20:
                return np.nan
            lags = range(2, min(20, len(x) // 2))
            tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
            if len(tau) < 2 or min(tau) <= 0:
                return np.nan
            reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
            return reg[0]
        return returns.rolling(window).apply(hurst, raw=True)

    def _rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _adx(self, df, period=14):
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(period).mean()

    def _atr(self, df, period=14):
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _bollinger_width(self, prices, period=20):
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        return (upper - lower) / ma
