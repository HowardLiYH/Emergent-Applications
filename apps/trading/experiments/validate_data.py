#!/usr/bin/env python3
"""
Validate all data before analysis.
RUN THIS FIRST!

Usage:
    python experiments/validate_data.py
"""
import sys
sys.path.insert(0, '.')

from src.data.validation import DataValidator, validate_all_markets
from src.data.loader import MultiMarketLoader, MarketType


def main():
    print("="*60)
    print("DATA VALIDATION")
    print("="*60)

    # Define what to validate
    markets_to_validate = {
        'crypto': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],  # Multi-asset ALWAYS
        # Add more as you get data:
        # 'forex': ['EURUSD', 'GBPUSD'],
        # 'stocks': ['SPY', 'QQQ'],
        # 'commodities': ['GOLD'],
    }

    results = validate_all_markets(markets_to_validate)

    # Check if we can proceed
    valid_assets = [k for k, v in results.items() if v['valid']]

    if len(valid_assets) == 0:
        print("\n❌ NO VALID DATA - Cannot proceed!")
        print("   Please fix data issues or download clean data.")
        return False

    print(f"\n✅ {len(valid_assets)} valid assets - Ready to proceed!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
