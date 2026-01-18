"""
Data Loader V2 - With Purging and Embargo

Fixes from audit:
1. Proper temporal split with purging gap
2. Market-specific handling (daily vs hourly)
3. Consistent timezone normalization
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from enum import Enum


def ensure_ohlc_consistency(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure OHLC data is consistent:
    - High >= max(open, close)
    - Low <= min(open, close)
    - High >= Low
    """
    df = data.copy()

    # Fix High: should be max of all
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)

    # Fix Low: should be min of all
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

    return df



class MarketType(Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCKS = "stocks"
    COMMODITIES = "commodities"


# Transaction costs by market (basis points, one-way)
TRANSACTION_COSTS = {
    MarketType.CRYPTO: 0.0004,      # 4 bps (0.04%) - Binance VIP tier
    MarketType.FOREX: 0.0001,       # 1 bp (0.01%) - tight spread majors
    MarketType.STOCKS: 0.0002,      # 2 bps (0.02%) - ETF spread
    MarketType.COMMODITIES: 0.0003, # 3 bps (0.03%) - futures spread
}


class DataLoaderV2:
    """
    Enhanced data loader with:
    - Purging/embargo between splits
    - Market-specific handling
    - Transaction cost configuration
    """

    def __init__(self, purge_days: int = 7, embargo_days: int = 1):
        """
        Args:
            purge_days: Gap between train and val (default 7 days)
            embargo_days: Gap between val and test (default 1 day)
        """
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def load(self, symbol: str, market_type: Optional[MarketType] = None) -> pd.DataFrame:
        """
        Load data for a symbol.

        Args:
            symbol: Asset symbol (e.g., 'BTCUSDT', 'EURUSD', 'SPY')
            market_type: Optional market type override
        """
        # Auto-detect market type and file
        if market_type is None:
            market_type = self._detect_market(symbol)

        # Find file
        filepath = self._find_file(symbol, market_type)
        if filepath is None:
            raise FileNotFoundError(f"Data not found for {symbol}")

        # Load
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]

        # Normalize timezone
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)

        # Ensure OHLC consistency (fix any high < low issues)
        df = ensure_ohlc_consistency(df)

        return df

    def _detect_market(self, symbol: str) -> MarketType:
        """Auto-detect market type from symbol."""
        symbol_upper = symbol.upper()

        if 'USDT' in symbol_upper or 'BTC' in symbol_upper or 'ETH' in symbol_upper or 'SOL' in symbol_upper:
            return MarketType.CRYPTO
        elif any(fx in symbol_upper for fx in ['EUR', 'GBP', 'JPY', 'USD']):
            if 'SPY' not in symbol_upper and 'QQQ' not in symbol_upper and 'AAPL' not in symbol_upper:
                return MarketType.FOREX
        elif symbol_upper in ['SPY', 'QQQ', 'AAPL']:
            return MarketType.STOCKS
        elif symbol_upper in ['GOLD', 'OIL']:
            return MarketType.COMMODITIES

        return MarketType.CRYPTO  # Default

    def _find_file(self, symbol: str, market_type: MarketType) -> Optional[Path]:
        """Find data file for symbol. Prefer daily for cross-market consistency."""
        base_dir = Path(f"data/{market_type.value}")

        # Prefer daily data for fair cross-market comparison
        patterns = [
            f"{symbol}_1d.csv",  # Daily first
            f"{symbol.replace('/', '')}_1d.csv",
            f"{symbol}_1h.csv",  # Hourly as fallback
            f"{symbol.replace('/', '')}_1h.csv",
        ]

        for pattern in patterns:
            filepath = base_dir / pattern
            if filepath.exists():
                return filepath

        return None

    def get_frequency(self, symbol: str, market_type: Optional[MarketType] = None) -> str:
        """Detect data frequency (hourly vs daily)."""
        if market_type is None:
            market_type = self._detect_market(symbol)

        filepath = self._find_file(symbol, market_type)
        if filepath and '_1h' in str(filepath):
            return 'hourly'
        return 'daily'

    def temporal_split_with_purging(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        frequency: str = 'daily'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data with purging gaps between sets.

        Purging prevents look-ahead bias from overlapping features.

        Args:
            data: Full dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            frequency: 'hourly' or 'daily' to compute purge periods

        Returns:
            (train, val, test) DataFrames with gaps between them
        """
        n = len(data)

        # Compute purge/embargo in rows
        if frequency == 'hourly':
            purge_rows = self.purge_days * 24
            embargo_rows = self.embargo_days * 24
        else:
            purge_rows = self.purge_days
            embargo_rows = self.embargo_days

        # Calculate split points accounting for gaps
        total_gap = purge_rows + embargo_rows
        usable_n = n - total_gap

        train_n = int(usable_n * train_ratio)
        val_n = int(usable_n * val_ratio)
        # test_n = remaining

        # Split indices
        train_end = train_n
        val_start = train_end + purge_rows
        val_end = val_start + val_n
        test_start = val_end + embargo_rows

        train = data.iloc[:train_end]
        val = data.iloc[val_start:val_end]
        test = data.iloc[test_start:]

        # Log split info
        print(f"Split with purging: train={len(train)}, val={len(val)}, test={len(test)}")
        print(f"  Purge gap: {purge_rows} rows ({self.purge_days} days)")
        print(f"  Embargo gap: {embargo_rows} rows ({self.embargo_days} days)")
        print(f"  Train: {train.index[0]} to {train.index[-1]}")
        print(f"  Val: {val.index[0]} to {val.index[-1]}")
        print(f"  Test: {test.index[0]} to {test.index[-1]}")

        return train, val, test

    def get_transaction_cost(self, market_type: MarketType) -> float:
        """Get transaction cost for a market (one-way, as decimal)."""
        return TRANSACTION_COSTS.get(market_type, 0.0005)
