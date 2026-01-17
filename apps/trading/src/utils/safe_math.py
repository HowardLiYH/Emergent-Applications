"""
Safe math utilities to handle NaN/Inf/zero.
"""
import numpy as np
import pandas as pd


def safe_divide(a, b, default=0.0):
    """Division with NaN/Inf/zero handling."""
    if b == 0 or np.isnan(b) or np.isinf(b):
        return default
    result = a / b
    if np.isnan(result) or np.isinf(result):
        return default
    return result


def validate_series(s: pd.Series, name: str, min_length: int = 10) -> pd.Series:
    """Validate series before computation."""
    if s.isna().all():
        raise ValueError(f"{name} is all NaN")
    if len(s.dropna()) < min_length:
        raise ValueError(f"{name} has only {len(s.dropna())} non-NaN values (need {min_length})")
    return s


def clip_outliers(s: pd.Series, n_std: float = 5.0) -> pd.Series:
    """Clip outliers beyond n standard deviations."""
    mean, std = s.mean(), s.std()
    lower, upper = mean - n_std * std, mean + n_std * std
    return s.clip(lower, upper)
