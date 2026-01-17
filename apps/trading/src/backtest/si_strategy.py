"""
SI-Based Trading Strategy

Uses the 3 confirmed SI correlates to generate trading signals:
1. trend_strength_7d - When SI high + trend strong = follow trend
2. jump_frequency_7d - When SI high + low jumps = stable regime
3. volume_volatility_7d - When SI high + volume stable = conviction trade

Strategy Logic:
- High SI = specialized agents = clear market signal
- Low SI = confused agents = uncertain market, reduce exposure
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class SISignal:
    """SI-based trading signal."""
    timestamp: pd.Timestamp
    si_level: float
    si_percentile: float
    trend_strength: float
    jump_frequency: float
    volume_volatility: float
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    regime: str


class SITradingStrategy:
    """
    Trading strategy based on Specialization Index correlations.

    Uses confirmed SI correlates to generate trading signals.
    """

    def __init__(self,
                 si_high_threshold: float = 0.7,
                 si_low_threshold: float = 0.3,
                 trend_threshold: float = 0.1,
                 lookback: int = 168):  # 7 days
        """
        Initialize strategy.

        Args:
            si_high_threshold: SI percentile above which we trust the signal
            si_low_threshold: SI percentile below which we reduce exposure
            trend_threshold: Minimum trend strength to follow
            lookback: Lookback period for calculations
        """
        self.si_high_threshold = si_high_threshold
        self.si_low_threshold = si_low_threshold
        self.trend_threshold = trend_threshold
        self.lookback = lookback

    def compute_features(self, data: pd.DataFrame, si: pd.Series) -> pd.DataFrame:
        """
        Compute the 3 confirmed SI correlates.
        """
        features = pd.DataFrame(index=data.index)
        returns = data['close'].pct_change()

        # 1. Trend strength (|mean return| / volatility)
        volatility = returns.rolling(self.lookback).std()
        mean_return = returns.rolling(self.lookback).mean()
        features['trend_strength'] = abs(mean_return) / volatility
        features['trend_direction'] = np.sign(mean_return)

        # 2. Jump frequency (returns > 3 std)
        std_rolling = returns.rolling(self.lookback).std()
        features['jump_frequency'] = (abs(returns) > 3 * std_rolling).rolling(self.lookback).mean()

        # 3. Volume volatility
        if 'volume' in data.columns and data['volume'].sum() > 0:
            vol_mean = data['volume'].rolling(self.lookback).mean()
            vol_std = data['volume'].rolling(self.lookback).std()
            features['volume_volatility'] = vol_std / vol_mean
        else:
            features['volume_volatility'] = 0.5  # Neutral if no volume data

        # SI percentile (rolling)
        si_aligned = si.reindex(data.index)
        features['si'] = si_aligned
        features['si_percentile'] = si_aligned.rolling(720).rank(pct=True)  # 30 day window

        return features

    def generate_signal(self, features: pd.DataFrame, idx: int) -> SISignal:
        """
        Generate trading signal at a specific index.

        Signal logic:
        1. If SI is high (specialized agents):
           - Market has clear structure
           - Follow trend if trend_strength is high
           - Size position by volume stability

        2. If SI is low (generalist agents):
           - Market is confused
           - Reduce exposure or go neutral
        """
        if idx < self.lookback:
            return SISignal(
                timestamp=features.index[idx],
                si_level=np.nan,
                si_percentile=np.nan,
                trend_strength=np.nan,
                jump_frequency=np.nan,
                volume_volatility=np.nan,
                signal=0.0,
                confidence=0.0,
                regime='insufficient_data'
            )

        row = features.iloc[idx]

        si_pct = row['si_percentile']
        trend = row['trend_strength']
        trend_dir = row['trend_direction']
        jumps = row['jump_frequency']
        vol_vol = row['volume_volatility']

        # Handle NaN
        if pd.isna(si_pct) or pd.isna(trend):
            return SISignal(
                timestamp=features.index[idx],
                si_level=row.get('si', np.nan),
                si_percentile=si_pct,
                trend_strength=trend,
                jump_frequency=jumps,
                volume_volatility=vol_vol,
                signal=0.0,
                confidence=0.0,
                regime='nan'
            )

        # Determine regime
        if si_pct > self.si_high_threshold:
            regime = 'specialized'
        elif si_pct < self.si_low_threshold:
            regime = 'generalized'
        else:
            regime = 'neutral'

        # Generate signal based on regime
        if regime == 'specialized':
            # High SI = trust market structure
            if trend > self.trend_threshold:
                # Strong trend - follow it
                base_signal = trend_dir * min(trend * 2, 1.0)  # Cap at 1
            else:
                # Weak trend but high SI - mean reversion might work
                base_signal = 0.0

            # Adjust confidence by stability metrics
            # Low jumps = stable = higher confidence
            jump_adj = 1 - min(jumps * 10, 0.5)  # Max 50% reduction

            # Low volume volatility = stable = higher confidence
            vol_adj = 1 - min(vol_vol, 0.5) if not pd.isna(vol_vol) else 1.0

            confidence = si_pct * jump_adj * vol_adj
            signal = base_signal * confidence

        elif regime == 'generalized':
            # Low SI = confused market = reduce exposure
            signal = 0.0
            confidence = 0.0

        else:
            # Neutral SI - partial exposure
            if trend > self.trend_threshold:
                signal = trend_dir * 0.3  # Reduced position
                confidence = (si_pct - self.si_low_threshold) / (self.si_high_threshold - self.si_low_threshold)
            else:
                signal = 0.0
                confidence = 0.0

        return SISignal(
            timestamp=features.index[idx],
            si_level=row.get('si', np.nan),
            si_percentile=si_pct,
            trend_strength=trend,
            jump_frequency=jumps,
            volume_volatility=vol_vol,
            signal=float(np.clip(signal, -1, 1)),
            confidence=float(np.clip(confidence, 0, 1)),
            regime=regime
        )

    def run(self, data: pd.DataFrame, si: pd.Series) -> pd.DataFrame:
        """
        Run strategy over entire dataset.

        Returns DataFrame with signals for each timestamp.
        """
        features = self.compute_features(data, si)

        signals = []
        for idx in range(len(features)):
            sig = self.generate_signal(features, idx)
            signals.append({
                'timestamp': sig.timestamp,
                'si_level': sig.si_level,
                'si_percentile': sig.si_percentile,
                'trend_strength': sig.trend_strength,
                'jump_frequency': sig.jump_frequency,
                'volume_volatility': sig.volume_volatility,
                'signal': sig.signal,
                'confidence': sig.confidence,
                'regime': sig.regime
            })

        return pd.DataFrame(signals).set_index('timestamp')

    def backtest(self, data: pd.DataFrame, si: pd.Series) -> Dict:
        """
        Backtest the strategy.

        Returns performance metrics.
        """
        signals_df = self.run(data, si)

        # Calculate returns
        returns = data['close'].pct_change()

        # Align signals with returns (signal at t generates return at t+1)
        strategy_returns = signals_df['signal'].shift(1) * returns

        # Drop NaN
        strategy_returns = strategy_returns.dropna()

        if len(strategy_returns) == 0:
            return {'error': 'No valid returns'}

        # Performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (8760 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(8760)
        sharpe = annual_return / volatility if volatility > 0 else 0

        # Drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        wins = (strategy_returns > 0).sum()
        total = (strategy_returns != 0).sum()
        win_rate = wins / total if total > 0 else 0

        # Regime analysis
        regime_returns = {}
        for regime in ['specialized', 'generalized', 'neutral']:
            regime_mask = signals_df['regime'].shift(1) == regime
            regime_mask = regime_mask.reindex(strategy_returns.index, fill_value=False)
            if regime_mask.sum() > 0:
                regime_ret = strategy_returns[regime_mask]
                regime_returns[regime] = {
                    'count': int(regime_mask.sum()),
                    'mean_return': float(regime_ret.mean()),
                    'total_return': float((1 + regime_ret).prod() - 1)
                }

        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'n_trades': int(total),
            'regime_performance': regime_returns
        }
