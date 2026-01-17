#!/usr/bin/env python3
"""
Download historical OHLCV data for multiple assets.
ALWAYS downloads multiple assets - NEVER single-coin!

Usage:
    python experiments/download_data.py

Data Sources (in priority order):
1. Bybit (no US geo-restriction for public data)
2. Kraken (no geo-restriction)
3. yfinance (fallback)
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time


def download_crypto_bybit(symbol: str, days: int = 365, save_dir: str = "data/crypto"):
    """
    Download crypto data using ccxt with Bybit exchange.
    Bybit has no US geo-restriction for public market data.
    """
    try:
        import ccxt
    except ImportError:
        print("❌ ccxt not installed. Run: pip install ccxt")
        return None

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {symbol} from Bybit ({days} days)...")

    try:
        exchange = ccxt.bybit({
            'enableRateLimit': True,
        })

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Fetch OHLCV data
        all_candles = []
        since = int(start_time.timestamp() * 1000)

        while True:
            try:
                candles = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=1000)
            except Exception as e:
                print(f"   ⚠️  Bybit fetch error: {e}")
                break

            if len(candles) == 0:
                break

            all_candles.extend(candles)
            since = candles[-1][0] + 1

            if candles[-1][0] >= int(end_time.timestamp() * 1000):
                break

            time.sleep(0.2)  # Rate limiting

        if len(all_candles) == 0:
            print(f"   ❌ No data retrieved for {symbol} from Bybit")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Save
        # Convert symbol format: BTC/USDT -> BTCUSDT
        filename = symbol.replace('/', '') + '_1h.csv'
        filepath = Path(save_dir) / filename
        df.to_csv(filepath)

        print(f"   ✅ Saved {len(df)} rows to {filepath} (Bybit)")
        return df

    except Exception as e:
        print(f"   ❌ Failed to download {symbol} from Bybit: {e}")
        return None


def download_crypto_kraken(symbol: str, days: int = 365, save_dir: str = "data/crypto"):
    """
    Download crypto data using ccxt with Kraken exchange.
    Kraken has no geo-restriction.
    """
    try:
        import ccxt
    except ImportError:
        print("❌ ccxt not installed. Run: pip install ccxt")
        return None

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Kraken uses different symbol formats
    # Map common symbols to Kraken format
    kraken_map = {
        'BTC/USDT': 'BTC/USD',
        'ETH/USDT': 'ETH/USD',
        'SOL/USDT': 'SOL/USD',
        'BNB/USDT': None,  # Not on Kraken
        'XRP/USDT': 'XRP/USD',
    }

    kraken_symbol = kraken_map.get(symbol, symbol.replace('USDT', 'USD'))

    if kraken_symbol is None:
        print(f"   ⚠️  {symbol} not available on Kraken")
        return None

    print(f"📥 Downloading {symbol} from Kraken ({kraken_symbol})...")

    try:
        exchange = ccxt.kraken({
            'enableRateLimit': True,
        })

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Fetch OHLCV data - Kraken limits historical data
        all_candles = []
        since = int(start_time.timestamp() * 1000)

        max_iterations = 50  # Safety limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            try:
                candles = exchange.fetch_ohlcv(kraken_symbol, '1h', since=since, limit=720)
            except Exception as e:
                print(f"   ⚠️  Kraken fetch error: {e}")
                break

            if len(candles) == 0:
                break

            all_candles.extend(candles)
            since = candles[-1][0] + 1

            if candles[-1][0] >= int(end_time.timestamp() * 1000):
                break

            time.sleep(0.5)  # Kraken has stricter rate limits

        if len(all_candles) == 0:
            print(f"   ❌ No data retrieved for {symbol} from Kraken")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Check if we got enough data (at least 180 days = ~4320 hours)
        if len(df) < 4000:
            print(f"   ⚠️  Only {len(df)} rows from Kraken (need 4000+ for 180 days)")
            # Don't save, let fallback handle it
            return None

        # Save with original symbol name
        filename = symbol.replace('/', '') + '_1h.csv'
        filepath = Path(save_dir) / filename
        df.to_csv(filepath)

        print(f"   ✅ Saved {len(df)} rows to {filepath} (Kraken)")
        return df

    except Exception as e:
        print(f"   ❌ Failed to download {symbol} from Kraken: {e}")
        return None


def download_crypto_yfinance(symbol: str, days: int = 365, save_dir: str = "data/crypto"):
    """
    Download crypto data using yfinance (fallback).
    """
    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed. Run: pip install yfinance")
        return None

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Convert symbol format: BTCUSDT -> BTC-USD
    yf_symbol = symbol.replace('USDT', '-USD')

    print(f"📥 Downloading {symbol} via yfinance ({yf_symbol})...")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = yf.download(
            yf_symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1h',
            progress=False
        )

        if len(df) == 0:
            print(f"   ❌ No data retrieved for {symbol}")
            return None

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Rename columns to lowercase
        df.columns = [str(c).lower() for c in df.columns]
        df.index.name = 'timestamp'

        # Keep only needed columns
        needed_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in needed_cols if c in df.columns]
        if len(available_cols) < 4:
            print(f"   ❌ Missing required columns: {set(needed_cols) - set(available_cols)}")
            return None

        df = df[available_cols]

        # Save
        filename = symbol + '_1h.csv'
        filepath = Path(save_dir) / filename
        df.to_csv(filepath)

        print(f"   ✅ Saved {len(df)} rows to {filepath}")
        return df

    except Exception as e:
        print(f"   ❌ Failed to download {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("DATA DOWNLOAD")
    print("⚠️  MULTI-ASSET MODE - Downloading multiple coins!")
    print("=" * 60)
    print("\nData sources (in priority order):")
    print("  1. Bybit (no US geo-restriction)")
    print("  2. Kraken (no geo-restriction)")
    print("  3. yfinance (fallback)")
    print("=" * 60)

    # Crypto symbols - ALWAYS multiple
    crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

    print(f"\nDownloading {len(crypto_symbols)} crypto assets...")

    success_count = 0
    for symbol in crypto_symbols:
        # Try Bybit first (no geo-restriction)
        result = download_crypto_bybit(symbol, days=400)

        if result is None:
            # Try Kraken second
            result = download_crypto_kraken(symbol, days=400)

        if result is None:
            # Fallback to yfinance
            yf_symbol = symbol.replace('/', '')
            result = download_crypto_yfinance(yf_symbol, days=400)

        if result is not None:
            success_count += 1

        time.sleep(1)  # Rate limiting between symbols

    print("\n" + "=" * 60)
    print(f"DOWNLOAD COMPLETE: {success_count}/{len(crypto_symbols)} assets")
    print("=" * 60)

    if success_count == 0:
        print("\n❌ NO DATA DOWNLOADED!")
        print("   Try installing ccxt: pip install ccxt")
        print("   Or yfinance: pip install yfinance")
        return False

    print("\n✅ Next step: python experiments/validate_data.py")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
