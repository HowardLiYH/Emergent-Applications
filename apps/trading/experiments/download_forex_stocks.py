#!/usr/bin/env python3
"""
Download Forex and Stocks data for cross-market validation.
Uses yfinance which works without geo-restrictions.
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time

try:
    import yfinance as yf
except ImportError:
    print("❌ yfinance not installed. Run: pip install yfinance")
    sys.exit(1)


def download_asset(symbol: str, yf_symbol: str, save_dir: str, days: int = 400):
    """
    Download hourly data using yfinance.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {symbol} ({yf_symbol})...")

    try:
        ticker = yf.Ticker(yf_symbol)

        # yfinance only allows 730 days of hourly data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=min(days, 729))

        df = ticker.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1h'
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

        # Add volume if missing (forex often has no volume)
        if 'volume' not in df.columns:
            df['volume'] = 1.0

        # Reset index to get timestamp column
        df = df.reset_index()
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'timestamp'})
        elif 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})
        elif 'index' in df.columns:
            df = df.rename(columns={'index': 'timestamp'})

        # Convert to UTC and remove timezone info for consistency
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

        # Set timestamp as index
        df = df.set_index('timestamp')

        # Select only needed columns
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Save
        filepath = Path(save_dir) / f"{symbol}_1h.csv"
        df.to_csv(filepath)

        print(f"   ✅ Saved {len(df)} rows to {filepath}")
        return df

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def main():
    print("=" * 60)
    print("DOWNLOAD FOREX & STOCKS DATA")
    print("=" * 60)

    # Forex pairs (using yfinance format)
    forex_assets = [
        ('EURUSD', 'EURUSD=X'),
        ('GBPUSD', 'GBPUSD=X'),
        ('USDJPY', 'USDJPY=X'),
    ]

    # Stock/ETF assets
    stock_assets = [
        ('SPY', 'SPY'),
        ('QQQ', 'QQQ'),
        ('AAPL', 'AAPL'),
    ]

    # Commodities
    commodity_assets = [
        ('GOLD', 'GC=F'),
        ('OIL', 'CL=F'),
    ]

    print("\n" + "=" * 60)
    print("FOREX")
    print("=" * 60)
    forex_success = 0
    for symbol, yf_symbol in forex_assets:
        result = download_asset(symbol, yf_symbol, "data/forex", days=400)
        if result is not None:
            forex_success += 1
        time.sleep(1)

    print("\n" + "=" * 60)
    print("STOCKS")
    print("=" * 60)
    stock_success = 0
    for symbol, yf_symbol in stock_assets:
        result = download_asset(symbol, yf_symbol, "data/stocks", days=400)
        if result is not None:
            stock_success += 1
        time.sleep(1)

    print("\n" + "=" * 60)
    print("COMMODITIES")
    print("=" * 60)
    commodity_success = 0
    for symbol, yf_symbol in commodity_assets:
        result = download_asset(symbol, yf_symbol, "data/commodities", days=400)
        if result is not None:
            commodity_success += 1
        time.sleep(1)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Forex: {forex_success}/{len(forex_assets)}")
    print(f"  Stocks: {stock_success}/{len(stock_assets)}")
    print(f"  Commodities: {commodity_success}/{len(commodity_assets)}")

    total = forex_success + stock_success + commodity_success
    if total > 0:
        print(f"\n✅ Downloaded {total} assets")
        print("Next: python experiments/run_full_cross_market.py")
        return True
    else:
        print("\n❌ No data downloaded")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
