#!/usr/bin/env python3
"""
Download 5 years of historical data from exchanges.

Crypto: Binance/Bybit (now accessible via VPN)
Forex: yfinance (5 years)
Stocks: yfinance (5 years)
Commodities: yfinance (5 years)

ALWAYS downloads multiple assets per market - NEVER single asset!
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time

# ============================================================
# CRYPTO: Binance/Bybit
# ============================================================

def download_crypto_binance(symbol: str, days: int = 1825, save_dir: str = "data/crypto"):
    """
    Download crypto data from Binance.
    5 years = ~1825 days
    """
    try:
        import ccxt
    except ImportError:
        print("❌ ccxt not installed. Run: pip install ccxt")
        return None

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {symbol} from Binance ({days} days)...")

    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        all_ohlcv = []
        current_since = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        while current_since < end_ms:
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1
            print(f"   Fetched {len(all_ohlcv)} candles...", end='\r')
            time.sleep(0.2)  # Rate limiting

        if not all_ohlcv:
            print(f"   ❌ No data returned from Binance")
            return None

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        df.index = df.index.tz_localize(None)  # Remove timezone for consistency

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        filepath = Path(save_dir) / f"{symbol.replace('/', '')}_1h.csv"
        df.to_csv(filepath)

        print(f"   ✅ Saved {len(df)} rows ({df.index[0]} to {df.index[-1]})")
        return df

    except Exception as e:
        print(f"   ❌ Binance failed: {e}")
        return None


def download_crypto_bybit(symbol: str, days: int = 1825, save_dir: str = "data/crypto"):
    """
    Download crypto data from Bybit as fallback.
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

        all_ohlcv = []
        current_since = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        while current_since < end_ms:
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1
            print(f"   Fetched {len(all_ohlcv)} candles...", end='\r')
            time.sleep(0.2)

        if not all_ohlcv:
            print(f"   ❌ No data returned from Bybit")
            return None

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        df.index = df.index.tz_localize(None)

        df = df[~df.index.duplicated(keep='first')]

        filepath = Path(save_dir) / f"{symbol.replace('/', '')}_1h.csv"
        df.to_csv(filepath)

        print(f"   ✅ Saved {len(df)} rows ({df.index[0]} to {df.index[-1]})")
        return df

    except Exception as e:
        print(f"   ❌ Bybit failed: {e}")
        return None


# ============================================================
# FOREX/STOCKS/COMMODITIES: yfinance
# ============================================================

def download_yfinance(symbol: str, yf_symbol: str, save_dir: str, years: int = 5):
    """
    Download data using yfinance.
    Note: yfinance only allows ~2 years of hourly data, so we use daily for longer periods.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed. Run: pip install yfinance")
        return None

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {symbol} ({yf_symbol}) - {years} years...")

    try:
        ticker = yf.Ticker(yf_symbol)

        # yfinance limits:
        # - Hourly data: max 730 days
        # - Daily data: unlimited
        # For 5 years, we need daily data

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        # Get daily data for 5 years
        df = ticker.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d'  # Daily for full 5 years
        )

        if len(df) == 0:
            print(f"   ❌ No data for {symbol}")
            return None

        # Clean column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        # Ensure required columns
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                print(f"   ❌ Missing column: {col}")
                return None

        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 1.0

        # Reset and normalize index
        df = df.reset_index()
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})
        elif 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'timestamp'})
        elif 'index' in df.columns:
            df = df.rename(columns={'index': 'timestamp'})

        # Convert to UTC and remove timezone
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

        df = df.set_index('timestamp')
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Save with _1d suffix for daily
        filepath = Path(save_dir) / f"{symbol}_1d.csv"
        df.to_csv(filepath)

        print(f"   ✅ Saved {len(df)} rows ({df.index[0]} to {df.index[-1]})")
        return df

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def main():
    print("=" * 70)
    print("DOWNLOAD 5 YEARS OF HISTORICAL DATA")
    print("=" * 70)

    # ============================================================
    # CRYPTO - Try Binance first, then Bybit
    # ============================================================
    print("\n" + "#" * 70)
    print("# CRYPTO (Binance/Bybit)")
    print("#" * 70)

    crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    crypto_success = 0

    for symbol in crypto_symbols:
        # Try Binance first
        result = download_crypto_binance(symbol, days=1825)

        if result is None:
            # Fallback to Bybit
            print(f"   Trying Bybit as fallback...")
            result = download_crypto_bybit(symbol, days=1825)

        if result is not None:
            crypto_success += 1
        else:
            print(f"   ❌ FAILED: Could not download {symbol} from any exchange")
            print(f"   ⚠️  STOPPING - Please check VPN connection")
            return False

        time.sleep(1)

    # ============================================================
    # FOREX
    # ============================================================
    print("\n" + "#" * 70)
    print("# FOREX (yfinance - 5yr daily)")
    print("#" * 70)

    forex_assets = [
        ('EURUSD', 'EURUSD=X'),
        ('GBPUSD', 'GBPUSD=X'),
        ('USDJPY', 'USDJPY=X'),
    ]

    forex_success = 0
    for symbol, yf_symbol in forex_assets:
        result = download_yfinance(symbol, yf_symbol, "data/forex", years=5)
        if result is not None:
            forex_success += 1
        time.sleep(1)

    # ============================================================
    # STOCKS
    # ============================================================
    print("\n" + "#" * 70)
    print("# STOCKS (yfinance - 5yr daily)")
    print("#" * 70)

    stock_assets = [
        ('SPY', 'SPY'),
        ('QQQ', 'QQQ'),
        ('AAPL', 'AAPL'),
    ]

    stock_success = 0
    for symbol, yf_symbol in stock_assets:
        result = download_yfinance(symbol, yf_symbol, "data/stocks", years=5)
        if result is not None:
            stock_success += 1
        time.sleep(1)

    # ============================================================
    # COMMODITIES
    # ============================================================
    print("\n" + "#" * 70)
    print("# COMMODITIES (yfinance - 5yr daily)")
    print("#" * 70)

    commodity_assets = [
        ('GOLD', 'GC=F'),
        ('OIL', 'CL=F'),
    ]

    commodity_success = 0
    for symbol, yf_symbol in commodity_assets:
        result = download_yfinance(symbol, yf_symbol, "data/commodities", years=5)
        if result is not None:
            commodity_success += 1
        time.sleep(1)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print(f"\n  Crypto: {crypto_success}/{len(crypto_symbols)} (5yr hourly from Binance/Bybit)")
    print(f"  Forex: {forex_success}/{len(forex_assets)} (5yr daily from yfinance)")
    print(f"  Stocks: {stock_success}/{len(stock_assets)} (5yr daily from yfinance)")
    print(f"  Commodities: {commodity_success}/{len(commodity_assets)} (5yr daily from yfinance)")

    total = crypto_success + forex_success + stock_success + commodity_success
    total_expected = len(crypto_symbols) + len(forex_assets) + len(stock_assets) + len(commodity_assets)

    if total == total_expected:
        print(f"\n✅ ALL {total} ASSETS DOWNLOADED SUCCESSFULLY!")
        return True
    else:
        print(f"\n⚠️  Downloaded {total}/{total_expected} assets")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
