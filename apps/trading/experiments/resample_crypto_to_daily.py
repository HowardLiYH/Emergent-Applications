#!/usr/bin/env python3
"""
Resample crypto hourly data to daily for fair cross-market comparison.
"""
import pandas as pd
from pathlib import Path


def resample_to_daily(input_file: str, output_file: str):
    """Resample hourly OHLCV to daily."""
    print(f"📊 Resampling {input_file}...")

    df = pd.read_csv(input_file, index_col=0, parse_dates=True)

    # Resample to daily using OHLCV rules
    daily = df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    daily.to_csv(output_file)
    print(f"   ✅ {len(df)} hourly → {len(daily)} daily rows")
    print(f"   Saved to {output_file}")

    return daily


def main():
    print("=" * 60)
    print("RESAMPLE CRYPTO TO DAILY")
    print("=" * 60)

    crypto_dir = Path("data/crypto")

    for hourly_file in crypto_dir.glob("*_1h.csv"):
        symbol = hourly_file.stem.replace("_1h", "")
        daily_file = crypto_dir / f"{symbol}_1d.csv"
        resample_to_daily(str(hourly_file), str(daily_file))

    print("\n✅ All crypto resampled to daily!")


if __name__ == "__main__":
    main()
