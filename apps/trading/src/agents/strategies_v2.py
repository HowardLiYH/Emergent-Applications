"""
Trading strategies - FREQUENCY AWARE VERSION.

Fixes the issue where hourly lookbacks were applied to daily data.
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Literal


# Lookback mapping: logical period -> bars
LOOKBACK_MAP = {
    'hourly': {
        'short': 24,    # 1 day
        'medium': 72,   # 3 days
        'long': 168,    # 7 days
        'breakout_s': 48,   # 2 days
        'breakout_l': 96,   # 4 days
    },
    'daily': {
        'short': 1,     # 1 day
        'medium': 3,    # 3 days
        'long': 7,      # 7 days
        'breakout_s': 2,    # 2 days
        'breakout_l': 4,    # 4 days
    }
}


class BaseStrategy(ABC):
    """Base class for trading strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def signal(self, data: pd.DataFrame, idx: int) -> float:
        """Generate trading signal in [-1, 1]."""
        pass


class MomentumStrategyV2(BaseStrategy):
    """Buy if price above MA, sell if below. Frequency-aware."""

    def __init__(self, lookback: int, name: str = None):
        super().__init__(name or f"Momentum_{lookback}")
        self.lookback = lookback

    def signal(self, data: pd.DataFrame, idx: int) -> float:
        if idx < self.lookback:
            return 0.0

        window = data['close'].iloc[idx-self.lookback:idx]
        ma = window.mean()
        current = data['close'].iloc[idx]

        if current > ma * 1.01:
            return 1.0
        elif current < ma * 0.99:
            return -1.0
        return 0.0


class MeanReversionStrategyV2(BaseStrategy):
    """Buy when oversold, sell when overbought. Frequency-aware."""

    def __init__(self, lookback: int, threshold: float = 2.0, name: str = None):
        super().__init__(name or f"MeanRev_{lookback}")
        self.lookback = lookback
        self.threshold = threshold

    def signal(self, data: pd.DataFrame, idx: int) -> float:
        if idx < self.lookback:
            return 0.0

        window = data['close'].iloc[idx-self.lookback:idx]
        ma = window.mean()
        std = window.std()
        current = data['close'].iloc[idx]

        z_score = (current - ma) / std if std > 0 else 0

        if z_score < -self.threshold:
            return 1.0
        elif z_score > self.threshold:
            return -1.0
        return 0.0


class BreakoutStrategyV2(BaseStrategy):
    """Buy on upward breakout, sell on downward. Frequency-aware."""

    def __init__(self, lookback: int, name: str = None):
        super().__init__(name or f"Breakout_{lookback}")
        self.lookback = lookback

    def signal(self, data: pd.DataFrame, idx: int) -> float:
        if idx < self.lookback:
            return 0.0

        window = data.iloc[idx-self.lookback:idx]
        current = data['close'].iloc[idx]

        high = window['high'].max()
        low = window['low'].min()

        if current > high:
            return 1.0
        elif current < low:
            return -1.0
        return 0.0


def get_default_strategies(frequency: Literal['hourly', 'daily'] = 'daily') -> List[BaseStrategy]:
    """
    Get default strategy set with correct lookbacks for the frequency.

    Args:
        frequency: 'hourly' or 'daily'

    Returns:
        List of 6 strategies with frequency-appropriate lookbacks
    """
    lb = LOOKBACK_MAP[frequency]

    return [
        MomentumStrategyV2(lookback=lb['long'], name=f"Momentum_long_{frequency}"),
        MomentumStrategyV2(lookback=lb['medium'], name=f"Momentum_medium_{frequency}"),
        MeanReversionStrategyV2(lookback=lb['short'], threshold=2.0, name=f"MeanRev_short_{frequency}"),
        MeanReversionStrategyV2(lookback=lb['medium'], threshold=1.5, name=f"MeanRev_medium_{frequency}"),
        BreakoutStrategyV2(lookback=lb['breakout_s'], name=f"Breakout_short_{frequency}"),
        BreakoutStrategyV2(lookback=lb['breakout_l'], name=f"Breakout_long_{frequency}"),
    ]


# Convenience aliases
def get_hourly_strategies() -> List[BaseStrategy]:
    return get_default_strategies('hourly')


def get_daily_strategies() -> List[BaseStrategy]:
    return get_default_strategies('daily')
