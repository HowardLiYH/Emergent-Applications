"""
Simple trading strategies for specialization testing.
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Base class for trading strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def signal(self, data: pd.DataFrame, idx: int) -> float:
        """
        Generate trading signal.

        Returns: float in [-1, 1] where:
            -1 = full short
             0 = no position
            +1 = full long
        """
        pass


class MomentumStrategy(BaseStrategy):
    """Buy if price above MA, sell if below."""

    def __init__(self, lookback: int = 168):  # 7 days
        super().__init__(f"Momentum_{lookback}")
        self.lookback = lookback

    def signal(self, data: pd.DataFrame, idx: int) -> float:
        if idx < self.lookback:
            return 0.0

        window = data['close'].iloc[idx-self.lookback:idx]
        ma = window.mean()
        current = data['close'].iloc[idx]

        if current > ma * 1.01:  # 1% above MA
            return 1.0
        elif current < ma * 0.99:  # 1% below MA
            return -1.0
        return 0.0


class MeanReversionStrategy(BaseStrategy):
    """Buy when oversold, sell when overbought."""

    def __init__(self, lookback: int = 24, threshold: float = 2.0):
        super().__init__(f"MeanRev_{lookback}")
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

        if z_score < -self.threshold:  # Oversold
            return 1.0
        elif z_score > self.threshold:  # Overbought
            return -1.0
        return 0.0


class BreakoutStrategy(BaseStrategy):
    """Buy on upward breakout, sell on downward."""

    def __init__(self, lookback: int = 48):
        super().__init__(f"Breakout_{lookback}")
        self.lookback = lookback

    def signal(self, data: pd.DataFrame, idx: int) -> float:
        if idx < self.lookback:
            return 0.0

        window = data.iloc[idx-self.lookback:idx]
        current = data['close'].iloc[idx]

        high = window['high'].max()
        low = window['low'].min()

        if current > high:  # Upward breakout
            return 1.0
        elif current < low:  # Downward breakout
            return -1.0
        return 0.0


# Create default strategy set
DEFAULT_STRATEGIES = [
    MomentumStrategy(lookback=168),
    MomentumStrategy(lookback=72),
    MeanReversionStrategy(lookback=24, threshold=2.0),
    MeanReversionStrategy(lookback=48, threshold=1.5),
    BreakoutStrategy(lookback=48),
    BreakoutStrategy(lookback=96),
]
