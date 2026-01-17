"""
Multi-market data loader for OHLCV data.
Supports: Crypto, Forex, Stocks, Commodities
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from enum import Enum


class MarketType(Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCKS = "stocks"
    COMMODITIES = "commodities"


# Market-specific configurations
MARKET_CONFIG = {
    MarketType.CRYPTO: {
        'data_dir': 'data/crypto',
        'trading_hours': 24,  # 24/7
        'has_volume': True,
        'assets': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT'],
    },
    MarketType.FOREX: {
        'data_dir': 'data/forex',
        'trading_hours': 24,  # 24/5 (closed weekends)
        'has_volume': False,  # Forex volume is unreliable
        'assets': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
    },
    MarketType.STOCKS: {
        'data_dir': 'data/stocks',
        'trading_hours': 6.5,  # 9:30-4:00
        'has_volume': True,
        'assets': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'],
    },
    MarketType.COMMODITIES: {
        'data_dir': 'data/commodities',
        'trading_hours': 23,  # Near 24h with breaks
        'has_volume': True,
        'assets': ['GOLD', 'OIL', 'SILVER', 'CORN', 'NATGAS'],
    },
}


class MultiMarketLoader:
    """Load and prepare data from multiple market types."""

    def __init__(self, market_type: MarketType = MarketType.CRYPTO, data_dir: str = None):
        self.market_type = market_type
        self.config = MARKET_CONFIG[market_type]
        self.data_dir = Path(data_dir) if data_dir else Path(self.config['data_dir'])

    def load(self, symbol: str,
             start: Optional[str] = None,
             end: Optional[str] = None) -> pd.DataFrame:
        """
        Load OHLCV data for a symbol.

        Returns DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex (hourly)
        """
        filepath = self.data_dir / f"{symbol}_1h.csv"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Data file not found: {filepath}\n"
                f"Available assets for {self.market_type.value}: {self.config['assets']}"
            )

        df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')

        # Filter date range
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]

        # Validate columns
        required = ['open', 'high', 'low', 'close']
        if self.config['has_volume']:
            required.append('volume')

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Add dummy volume for forex if needed
        if not self.config['has_volume'] and 'volume' not in df.columns:
            df['volume'] = 1.0  # Placeholder

        return df[['open', 'high', 'low', 'close', 'volume']]

    def temporal_split(self, df: pd.DataFrame,
                       train_pct: float = 0.70,
                       val_pct: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically (no shuffle).

        Returns: (train, val, test) DataFrames
        """
        n = len(df)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

        return train, val, test

    def get_available_assets(self) -> List[str]:
        """Get list of available assets for this market."""
        return self.config['assets']


# Convenience function for backward compatibility
class DataLoader(MultiMarketLoader):
    """Crypto-specific loader (default)."""
    def __init__(self, data_dir: str = "data/crypto"):
        super().__init__(MarketType.CRYPTO)
        self.data_dir = Path(data_dir)
