#!/usr/bin/env python3
"""
Download macro features for SI correlation analysis.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import time

def download_macro():
    print("\n" + "="*70)
    print("  DOWNLOADING MACRO FEATURES")
    print("="*70)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)
    
    macro_symbols = {
        '^VIX': 'VIX',
        'DX-Y.NYB': 'DXY',
        '^TNX': 'US10Y',
        'GC=F': 'GOLD',
        'CL=F': 'OIL',
        '^GSPC': 'SP500',
        'HYG': 'HY_CREDIT',
        'LQD': 'IG_CREDIT',
    }
    
    output_dir = Path("data/macro")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    
    for symbol, name in macro_symbols.items():
        print(f"  {name:15} ({symbol})...", end=" ", flush=True)
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                print("❌ No data")
                continue
            
            # Clean columns - handle potential MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
            
            # Ensure timezone-naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Save
            filepath = output_dir / f"{name}_1d.csv"
            df.to_csv(filepath)
            
            print(f"✅ {len(df)} days")
            downloaded.append(name)
            
            time.sleep(0.3)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n  Downloaded: {len(downloaded)}/{len(macro_symbols)} macro indicators")
    print(f"  Saved to: {output_dir}")
    print("="*70 + "\n")
    
    return downloaded

if __name__ == "__main__":
    download_macro()
