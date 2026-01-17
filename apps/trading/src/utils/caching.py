"""
Data caching for faster repeated runs.
"""
import hashlib
import pickle
from pathlib import Path


class DataCache:
    """Cache loaded/processed data to avoid recomputation."""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = str(args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, *args):
        """Get cached data if exists."""
        key = self._get_key(*args)
        filepath = self.cache_dir / f"{key}.pkl"

        if filepath.exists():
            with open(filepath, 'rb') as f:
                print(f"📦 Cache hit: {key[:8]}...")
                return pickle.load(f)

        return None

    def set(self, data, *args):
        """Cache data."""
        key = self._get_key(*args)
        filepath = self.cache_dir / f"{key}.pkl"

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"💾 Cached: {key[:8]}...")

    def get_or_compute(self, compute_fn, *args):
        """Get from cache or compute and cache."""
        cached = self.get(*args)
        if cached is not None:
            return cached

        result = compute_fn()
        self.set(result, *args)
        return result

    def clear(self):
        """Clear all cached data."""
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
        print("🗑️  Cache cleared")
