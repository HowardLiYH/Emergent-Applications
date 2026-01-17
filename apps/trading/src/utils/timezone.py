"""
Timezone utilities - standardize all data to UTC.
"""
import pandas as pd


def standardize_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all timestamps to UTC.

    IMPORTANT: Mixed timezones (crypto=UTC, stocks=ET) cause bugs!
    Always call this after loading data.
    """
    if df.index.tz is None:
        # Assume UTC if no timezone info
        df.index = df.index.tz_localize('UTC')
    else:
        # Convert to UTC
        df.index = df.index.tz_convert('UTC')

    return df


def validate_timezone(df: pd.DataFrame, expected: str = 'UTC') -> bool:
    """Check if data is in expected timezone."""
    if df.index.tz is None:
        print(f"⚠️  WARNING: No timezone info. Assuming {expected}.")
        return True

    actual = str(df.index.tz)
    if actual != expected:
        print(f"⚠️  WARNING: Data is in {actual}, expected {expected}")
        return False

    return True
