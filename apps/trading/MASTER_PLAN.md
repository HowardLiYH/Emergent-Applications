# MASTER PLAN: SI Signal Discovery

> **This is the ONLY file you need to execute the project.**
> All code, steps, and decisions are inline. No need to read other files.

---

## 📍 Current Status

```
[x] Phase 0: Planning & Methodology (COMPLETE)
[ ] Phase 1: Pre-Registration & Setup  ← YOU ARE HERE
[ ] Phase 2: Data & Features (incl. validation + multi-market)
[ ] Phase 3: Backtest & SI Computation
[ ] Phase 4: Discovery Pipeline
[ ] Phase 5: Prediction Pipeline
[ ] Phase 6: SI Dynamics Pipeline
[ ] Phase 7: Audits & Validation
[ ] Phase 8: Final Report
[ ] Phase 9: Cross-Market Validation (crypto/forex/stocks/commodities)
```

---

## 🎯 Goal

**Discover what Specialization Index (SI) correlates with in trading.**

Primary Hypothesis: SI correlates with volatility (r > 0.15, p < 0.05)

---

## 🌍 Multi-Market Scope

**Key Insight**: Different markets have different characteristics. SI may work better in some than others.

| Market Type | Assets | Characteristics |
|-------------|--------|-----------------|
| **Crypto** | BTC, ETH, SOL | 24/7, high volatility, no close |
| **Forex** | EUR/USD, GBP/USD | 24/5, lower volatility, macro-driven |
| **US Stocks** | SPY, QQQ, AAPL | 6.5h/day, earnings, dividends |
| **Commodities** | Gold, Oil, Corn | Seasonal, supply/demand driven |

**Testing Order**:
1. Crypto (BTC) - Start here (most data, 24/7)
2. Crypto (ETH, SOL) - Validate within market
3. Forex (EUR/USD) - Cross-market validation
4. Stocks (SPY) - Different market structure
5. Commodities (Gold) - Alternative asset class

**Success Criteria**: SI findings should hold in at least 2 of 4 market types

---

## ⚠️ Contingency Plans

| If This Happens | Then Do This |
|-----------------|--------------|
| Data source unavailable | Use alternative source (listed below) |
| API rate limited | Add sleep, reduce batch size |
| Memory issues | Process data in chunks |
| SI always 0 or 1 | Check agent diversity, adjust parameters |
| All correlations null | Check data quality, try longer SI window |
| Validation fails | Re-examine train findings, check for overfit |

---

## 🛑 Stopping Criteria

Stop and reassess if:

| Phase | Stop If | Action |
|-------|---------|--------|
| Phase 2 | Data validation fails 3+ times | Find different data source |
| Phase 3 | SI has zero variance | Check NichePopulation implementation |
| Phase 4 | All 46 features have \|r\| < 0.05 | Fundamental issue - reassess SI definition |
| Phase 7 | Zero candidates confirmed in validation | Overfitting in train - simplify |
| Any | Stuck 3+ days on one phase | Seek help, reassess approach |

---

## 🎯 Exit Criteria

### Success (Proceed to Future Enhancements)

- [ ] ≥3 features with \|r\| > 0.15, FDR < 0.05, confirmed in val AND test
- [ ] SI Granger-causes at least one prediction target (p < 0.05)
- [ ] Findings replicate in ≥3 of 5 assets

### Partial Success (Proceed with Caution)

- [ ] 1-2 significant features confirmed
- [ ] OR SI correlates but doesn't predict
- [ ] OR works in 1-2 assets only

### Failure (Pivot or Abandon)

- [ ] Zero significant features after full analysis
- [ ] SI doesn't vary meaningfully
- [ ] All findings are noise (fail permutation tests)

---

# PHASE 1: Pre-Registration & Setup

## Step 1.1: Commit Pre-Registration (CRITICAL)

**Why**: Prevents p-hacking. Proves we didn't change hypothesis after seeing results.

```bash
cd /Users/yuhaoli/code/MAS_For_Finance/Emergent-Applications/apps/trading

# The pre-registration file already exists at:
# experiments/pre_registration.json

# Commit it BEFORE any analysis
git add experiments/pre_registration.json
git commit -m "PRE-REGISTRATION: SI hypothesis - $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push origin main
```

**Checkpoint**: Pre-registration committed? ☐ Yes ☐ No

---

## Step 1.2: Create Project Structure

```bash
mkdir -p src/{data,agents,competition,analysis,backtest}
mkdir -p tests
mkdir -p results/si_correlations
mkdir -p data/{crypto,forex,stocks,commodities}
```

---

## Step 1.3: Create Requirements

```python
# requirements.txt
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.4.0
requests>=2.31.0  # For data download
```

```bash
pip install -r requirements.txt
```

**Checkpoint**: Dependencies installed? ☐ Yes ☐ No

---

## Step 1.4: Create Package Structure

```bash
# Create __init__.py files for proper imports
touch src/__init__.py
touch src/data/__init__.py
touch src/agents/__init__.py
touch src/competition/__init__.py
touch src/analysis/__init__.py
touch src/backtest/__init__.py
```

---

## Step 1.5: Minimal Smoke Test (30 min)

Before full execution, verify core logic works:

```python
# experiments/smoke_test.py
"""
Minimal test to verify SI computation works.
Run this BEFORE downloading real data.
"""
import numpy as np
import pandas as pd

def test_si_computation():
    """Test SI with synthetic data."""
    print("=" * 60)
    print("SMOKE TEST: SI Computation")
    print("=" * 60)

    # 1. Create synthetic price data
    np.random.seed(42)
    n = 1000
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n)),
        'low': prices - np.abs(np.random.randn(n)),
        'close': prices + np.random.randn(n) * 0.1,
        'volume': np.random.randint(100, 10000, n),
    }, index=pd.date_range('2024-01-01', periods=n, freq='1h'))

    print(f"✅ Created synthetic data: {len(data)} rows")

    # 2. Create simple strategies
    class SimpleStrategy:
        def __init__(self, bias: float):
            self.bias = bias
        def signal(self, data, idx):
            if idx < 10:
                return 0
            ret = data['close'].iloc[idx] / data['close'].iloc[idx-10] - 1
            return 1.0 if ret > self.bias else -1.0 if ret < -self.bias else 0.0

    strategies = [SimpleStrategy(0.01), SimpleStrategy(0.02), SimpleStrategy(0.03)]
    print(f"✅ Created {len(strategies)} strategies")

    # 3. Create agents with niche affinities
    class SimpleAgent:
        def __init__(self, strategy_idx):
            self.strategy_idx = strategy_idx
            self.niche_affinity = np.ones(3) / 3

        def update(self, regime, won):
            alpha = 0.1
            if won:
                self.niche_affinity[regime] += alpha * (1 - self.niche_affinity[regime])
            else:
                self.niche_affinity[regime] *= (1 - alpha)
            self.niche_affinity /= self.niche_affinity.sum()

    agents = [SimpleAgent(i % 3) for i in range(9)]
    print(f"✅ Created {len(agents)} agents")

    # 4. Run competition
    for idx in range(100, 500):
        regime = idx % 3  # Simple regime rotation

        # Get returns
        returns = []
        for agent in agents:
            signal = strategies[agent.strategy_idx].signal(data, idx-1)
            ret = signal * (data['close'].iloc[idx] / data['close'].iloc[idx-1] - 1)
            returns.append((agent, ret))

        # Winner-take-all
        winner = max(returns, key=lambda x: x[1])[0]

        for agent, ret in returns:
            agent.update(regime, agent == winner)

    print(f"✅ Ran 400 competition rounds")

    # 5. Compute SI
    def compute_si(agents):
        entropies = []
        for agent in agents:
            p = agent.niche_affinity + 1e-10
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(len(p))
            entropies.append(entropy / max_entropy)
        return 1 - np.mean(entropies)

    si = compute_si(agents)
    print(f"✅ Computed SI: {si:.3f}")

    # 6. Validate
    assert 0 <= si <= 1, f"SI out of range: {si}"
    print(f"✅ SI is in valid range [0, 1]")

    # 7. Check agents specialized
    for i, agent in enumerate(agents):
        dominant = np.argmax(agent.niche_affinity)
        print(f"   Agent {i}: dominant regime {dominant}, affinity {agent.niche_affinity.round(2)}")

    print("\n" + "=" * 60)
    print("🎉 SMOKE TEST PASSED!")
    print("=" * 60)

    return True

if __name__ == "__main__":
    success = test_si_computation()
    exit(0 if success else 1)
```

```bash
python experiments/smoke_test.py
```

**Checkpoint**: Smoke test passed? ☐ Yes ☐ No

**If smoke test fails**: Fix before proceeding. Core logic must work.

---

# PHASE 2: Data & Features

## Step 2.1: Multi-Market Data Loader

Create `src/data/loader.py`:

```python
"""
Multi-market data loader for OHLCV data.
Supports: Crypto, Forex, Stocks, Commodities
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from enum import Enum

class MarketType(Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCKS = "stocks"
    COMMODITIES = "commodities"

# Market-specific configurations
MARKET_CONFIG = {
    MarketType.CRYPTO: {
        'data_dir': 'data/crypto',
        'trading_hours': 24,  # 24/7
        'has_volume': True,
        'assets': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT'],
    },
    MarketType.FOREX: {
        'data_dir': 'data/forex',
        'trading_hours': 24,  # 24/5 (closed weekends)
        'has_volume': False,  # Forex volume is unreliable
        'assets': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
    },
    MarketType.STOCKS: {
        'data_dir': 'data/stocks',
        'trading_hours': 6.5,  # 9:30-4:00
        'has_volume': True,
        'assets': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'],
    },
    MarketType.COMMODITIES: {
        'data_dir': 'data/commodities',
        'trading_hours': 23,  # Near 24h with breaks
        'has_volume': True,
        'assets': ['GOLD', 'OIL', 'SILVER', 'CORN', 'NATGAS'],
    },
}


class MultiMarketLoader:
    """Load and prepare data from multiple market types."""

    def __init__(self, market_type: MarketType = MarketType.CRYPTO):
        self.market_type = market_type
        self.config = MARKET_CONFIG[market_type]
        self.data_dir = Path(self.config['data_dir'])

    def load(self, symbol: str,
             start: Optional[str] = None,
             end: Optional[str] = None) -> pd.DataFrame:
        """
        Load OHLCV data for a symbol.

        Returns DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex (hourly)
        """
        filepath = self.data_dir / f"{symbol}_1h.csv"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Data file not found: {filepath}\n"
                f"Available assets for {self.market_type.value}: {self.config['assets']}"
            )

        df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')

        # Filter date range
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]

        # Validate columns
        required = ['open', 'high', 'low', 'close']
        if self.config['has_volume']:
            required.append('volume')

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Add dummy volume for forex if needed
        if not self.config['has_volume'] and 'volume' not in df.columns:
            df['volume'] = 1.0  # Placeholder

        return df[['open', 'high', 'low', 'close', 'volume']]

    def temporal_split(self, df: pd.DataFrame,
                       train_pct: float = 0.70,
                       val_pct: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically (no shuffle).

        Returns: (train, val, test) DataFrames
        """
        n = len(df)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

        return train, val, test

    def get_available_assets(self) -> list:
        """Get list of available assets for this market."""
        return self.config['assets']


# Convenience function for backward compatibility
class DataLoader(MultiMarketLoader):
    """Crypto-specific loader (default)."""
    def __init__(self, data_dir: str = "data/crypto"):
        super().__init__(MarketType.CRYPTO)
        self.data_dir = Path(data_dir)
```

---

## Step 2.2: Data Validation (CRITICAL)

Create `src/data/validation.py`:

```python
"""
Data validation - MUST RUN before any analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path

class DataValidator:
    """Validate data quality before analysis."""

    def __init__(self, strict: bool = True):
        self.strict = strict  # If True, raises errors. If False, just warns.

    def validate(self, df: pd.DataFrame, symbol: str = "unknown") -> Dict:
        """
        Run all validation checks.

        Returns dict with:
            - valid: bool (True if all checks pass)
            - issues: list of issues found
            - stats: data statistics
        """
        issues = []
        warnings = []

        print(f"\n{'='*60}")
        print(f"VALIDATING: {symbol}")
        print('='*60)

        # 1. Check for missing values
        print("\n1. Checking missing values...")
        missing_pct = df.isna().mean() * 100
        for col, pct in missing_pct.items():
            if pct > 5:
                issues.append(f"Column '{col}' has {pct:.1f}% missing values")
            elif pct > 0:
                warnings.append(f"Column '{col}' has {pct:.1f}% missing values")

        if missing_pct.sum() == 0:
            print("   ✅ No missing values")
        else:
            print(f"   ⚠️  Missing values found: {missing_pct[missing_pct > 0].to_dict()}")

        # 2. Check for duplicates
        print("\n2. Checking for duplicate timestamps...")
        n_dups = df.index.duplicated().sum()
        if n_dups > 0:
            issues.append(f"Found {n_dups} duplicate timestamps")
            print(f"   ❌ Found {n_dups} duplicates")
        else:
            print("   ✅ No duplicates")

        # 3. Check for gaps
        print("\n3. Checking for data gaps...")
        if len(df) > 1:
            expected_freq = pd.Timedelta('1h')
            actual_gaps = df.index.to_series().diff()
            large_gaps = actual_gaps[actual_gaps > expected_freq * 2]

            if len(large_gaps) > 0:
                max_gap = large_gaps.max()
                warnings.append(f"Found {len(large_gaps)} gaps > 2 hours (max: {max_gap})")
                print(f"   ⚠️  Found {len(large_gaps)} gaps (max: {max_gap})")
            else:
                print("   ✅ No significant gaps")

        # 4. Check for extreme returns
        print("\n4. Checking for extreme returns...")
        returns = df['close'].pct_change()
        extreme_threshold = 0.5  # 50% in 1 hour
        extreme = (abs(returns) > extreme_threshold).sum()

        if extreme > 0:
            warnings.append(f"Found {extreme} extreme returns (>{extreme_threshold*100}% in 1h)")
            print(f"   ⚠️  Found {extreme} extreme returns")

            # Show the extremes
            extreme_rows = returns[abs(returns) > extreme_threshold]
            for ts, ret in extreme_rows.items():
                print(f"      {ts}: {ret*100:+.1f}%")
        else:
            print("   ✅ No extreme returns")

        # 5. Check data range
        print("\n5. Checking data range...")
        days = (df.index.max() - df.index.min()).days

        if days < 180:
            issues.append(f"Only {days} days of data (recommend 365+)")
            print(f"   ❌ Only {days} days (recommend 365+)")
        elif days < 365:
            warnings.append(f"Only {days} days of data (365+ preferred)")
            print(f"   ⚠️  {days} days (365+ preferred)")
        else:
            print(f"   ✅ {days} days of data")

        # 6. Check for negative prices
        print("\n6. Checking for invalid prices...")
        neg_prices = (df[['open', 'high', 'low', 'close']] <= 0).sum().sum()
        if neg_prices > 0:
            issues.append(f"Found {neg_prices} non-positive prices")
            print(f"   ❌ Found {neg_prices} non-positive prices")
        else:
            print("   ✅ All prices valid")

        # 7. Check OHLC consistency
        print("\n7. Checking OHLC consistency...")
        ohlc_issues = (
            (df['high'] < df['low']).sum() +
            (df['high'] < df['open']).sum() +
            (df['high'] < df['close']).sum() +
            (df['low'] > df['open']).sum() +
            (df['low'] > df['close']).sum()
        )
        if ohlc_issues > 0:
            issues.append(f"Found {ohlc_issues} OHLC inconsistencies")
            print(f"   ❌ Found {ohlc_issues} OHLC inconsistencies")
        else:
            print("   ✅ OHLC consistent")

        # Summary
        print(f"\n{'='*60}")
        valid = len(issues) == 0

        if valid and len(warnings) == 0:
            print("✅ VALIDATION PASSED - Data is clean!")
        elif valid:
            print(f"⚠️  VALIDATION PASSED with {len(warnings)} warnings")
        else:
            print(f"❌ VALIDATION FAILED - {len(issues)} critical issues")
            for issue in issues:
                print(f"   - {issue}")

        # Stats
        stats = {
            'n_rows': len(df),
            'date_start': str(df.index.min()),
            'date_end': str(df.index.max()),
            'days': days,
            'price_min': float(df['close'].min()),
            'price_max': float(df['close'].max()),
            'return_mean': float(returns.mean()),
            'return_std': float(returns.std()),
        }

        result = {
            'valid': valid,
            'issues': issues,
            'warnings': warnings,
            'stats': stats,
        }

        # Raise error if strict mode and issues found
        if self.strict and not valid:
            raise ValueError(
                f"Data validation failed for {symbol}:\n" +
                "\n".join(f"  - {i}" for i in issues)
            )

        return result


def validate_all_markets(markets: Dict[str, List[str]]) -> Dict:
    """
    Validate data for multiple markets and assets.

    Args:
        markets: Dict like {'crypto': ['BTCUSDT', 'ETHUSDT'], 'forex': ['EURUSD']}

    Returns:
        Dict with validation results for each asset
    """
    from .loader import MultiMarketLoader, MarketType

    results = {}
    validator = DataValidator(strict=False)

    for market_name, assets in markets.items():
        market_type = MarketType(market_name)
        loader = MultiMarketLoader(market_type)

        for asset in assets:
            try:
                df = loader.load(asset)
                result = validator.validate(df, f"{market_name}/{asset}")
                results[f"{market_name}/{asset}"] = result
            except FileNotFoundError as e:
                results[f"{market_name}/{asset}"] = {
                    'valid': False,
                    'issues': [str(e)],
                    'warnings': [],
                    'stats': None
                }

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    valid_count = sum(1 for r in results.values() if r['valid'])
    print(f"\nPassed: {valid_count}/{len(results)}")

    for asset, result in results.items():
        status = "✅" if result['valid'] else "❌"
        print(f"  {status} {asset}")

    return results
```

### Run Data Validation

```python
# experiments/validate_data.py
"""
Validate all data before analysis.
RUN THIS FIRST!
"""
import sys
sys.path.append('.')

from src.data.validation import DataValidator, validate_all_markets
from src.data.loader import MultiMarketLoader, MarketType

def main():
    print("="*60)
    print("DATA VALIDATION")
    print("="*60)

    # Define what to validate
    markets_to_validate = {
        'crypto': ['BTCUSDT', 'ETHUSDT'],  # Start with these
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
    exit(0 if success else 1)
```

```bash
# Run data validation BEFORE any analysis
python experiments/validate_data.py
```

**Checkpoint**: Data validation passed? ☐ Yes ☐ No

**If validation fails**: Fix data issues before proceeding!

---

## Step 2.3: Data Sources (Where to Get Data)

| Market | Free Source | Paid Source |
|--------|-------------|-------------|
| **Crypto** | [Binance Data](https://data.binance.vision/), [CryptoDataDownload](https://www.cryptodatadownload.com/) | Bybit API, Binance API |
| **Forex** | [HistData](https://www.histdata.com/), [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) | OANDA, Interactive Brokers |
| **Stocks** | [Yahoo Finance](https://finance.yahoo.com/) via `yfinance` | Alpha Vantage, Polygon.io |
| **Commodities** | [Investing.com](https://www.investing.com/) | Quandl, CME DataMine |

```python
# Example: Download crypto data
import yfinance as yf

# For stocks/ETFs
spy = yf.download("SPY", start="2023-01-01", end="2024-12-31", interval="1h")
spy.to_csv("data/stocks/SPY_1h.csv")

# For crypto (via yfinance)
btc = yf.download("BTC-USD", start="2023-01-01", end="2024-12-31", interval="1h")
btc.to_csv("data/crypto/BTCUSDT_1h.csv")
```

---

## Step 2.2: Feature Calculator

Create `src/analysis/features.py`:

```python
"""
Feature calculator for SI correlation analysis.
46 discovery features + 2 prediction features + 9 SI dynamics features
"""
import pandas as pd
import numpy as np
from typing import Dict, List

class FeatureCalculator:
    """Calculate all features for SI correlation analysis."""

    # Features that use future data (for prediction pipeline only)
    LOOKAHEAD_FEATURES = {'next_day_return', 'next_day_volatility'}

    # Features removed from discovery (circular with SI)
    CIRCULAR_FEATURES = {
        'dsi_dt', 'si_acceleration', 'si_rolling_std',
        'si_1h', 'si_4h', 'si_1d', 'si_1w', 'si_percentile',
        'strategy_concentration', 'niche_affinity_entropy'
    }

    def __init__(self):
        self.features_computed = []

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features."""
        features = pd.DataFrame(index=df.index)

        # Market features (15)
        features = self._add_market_features(df, features)

        # Risk features (10)
        features = self._add_risk_features(df, features)

        # Behavioral features (6)
        features = self._add_behavioral_features(df, features)

        # Liquidity features (4)
        features = self._add_liquidity_features(df, features)

        # Prediction features (2) - lookahead
        features = self._add_prediction_features(df, features)

        self.features_computed = list(features.columns)

        return features

    def _add_market_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add market state features."""
        returns = df['close'].pct_change()

        # Volatility at different windows
        features['volatility_24h'] = returns.rolling(24).std()
        features['volatility_7d'] = returns.rolling(168).std()
        features['volatility_30d'] = returns.rolling(720).std()

        # Trend strength (absolute return / volatility)
        features['trend_strength_7d'] = abs(returns.rolling(168).mean()) / features['volatility_7d']

        # Autocorrelation
        features['return_autocorr_7d'] = returns.rolling(168).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
        )

        # Hurst exponent (simplified)
        features['hurst_exponent'] = self._rolling_hurst(returns, window=168)

        # Return entropy
        features['return_entropy_7d'] = returns.rolling(168).apply(
            self._entropy, raw=True
        )

        # Volume features
        features['volume_24h'] = df['volume'].rolling(24).mean()
        features['volume_volatility_7d'] = df['volume'].rolling(168).std() / df['volume'].rolling(168).mean()

        # Jump frequency (returns > 3 std)
        std_7d = returns.rolling(168).std()
        features['jump_frequency_7d'] = (abs(returns) > 3 * std_7d).rolling(168).mean()

        # Variance ratio
        features['variance_ratio'] = features['volatility_7d'] / features['volatility_24h']

        # Technical indicators
        features['rsi'] = self._rsi(df['close'], 14)
        features['adx'] = self._adx(df, 14)
        features['atr'] = self._atr(df, 14)
        features['bb_width'] = self._bollinger_width(df['close'], 20)

        return features

    def _add_risk_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add risk metrics (using PAST data only)."""
        returns = df['close'].pct_change()

        # Max drawdown (rolling 30d)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.rolling(720).max()
        drawdown = (cumulative - running_max) / running_max
        features['max_drawdown_30d'] = drawdown.rolling(720).min()

        # VaR and CVaR
        features['var_95_30d'] = returns.rolling(720).quantile(0.05)
        features['cvar_95_30d'] = returns.rolling(720).apply(
            lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else np.nan
        )

        # Volatility of volatility
        vol_24h = returns.rolling(24).std()
        features['vol_of_vol_30d'] = vol_24h.rolling(720).std()

        # Tail ratio
        features['tail_ratio_30d'] = abs(features['var_95_30d']) / returns.rolling(720).quantile(0.95)

        # Win rate (positive returns)
        features['win_rate_30d'] = (returns > 0).rolling(720).mean()

        # Profit factor
        gains = returns.clip(lower=0).rolling(720).sum()
        losses = abs(returns.clip(upper=0).rolling(720).sum())
        features['profit_factor_30d'] = gains / losses.replace(0, np.nan)

        # Sharpe and Sortino (rolling)
        features['sharpe_ratio_30d'] = returns.rolling(720).mean() / returns.rolling(720).std() * np.sqrt(8760)
        downside = returns.clip(upper=0).rolling(720).std()
        features['sortino_ratio_30d'] = returns.rolling(720).mean() / downside * np.sqrt(8760)

        return features

    def _add_behavioral_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral/timing features."""
        returns = df['close'].pct_change()

        # Momentum vs mean-reversion returns
        features['momentum_return_7d'] = returns.rolling(168).mean()

        # Mean reversion signal (distance from MA)
        ma_30d = df['close'].rolling(720).mean()
        features['meanrev_signal'] = (df['close'] - ma_30d) / ma_30d

        # Fear/greed proxy (volatility rank)
        vol = returns.rolling(24).std()
        features['fear_greed_proxy'] = vol.rolling(720).rank(pct=True)

        # Regime duration (how long since vol regime change)
        vol_regime = (vol > vol.rolling(168).median()).astype(int)
        features['regime_duration'] = vol_regime.groupby((vol_regime != vol_regime.shift()).cumsum()).cumcount()

        return features

    def _add_liquidity_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity features."""
        returns = df['close'].pct_change()

        # Volume z-score
        features['volume_z'] = (df['volume'] - df['volume'].rolling(720).mean()) / df['volume'].rolling(720).std()

        # Amihud illiquidity (|return| / volume)
        features['amihud_log'] = np.log1p(abs(returns) / df['volume'].replace(0, np.nan) * 1e9)

        # Volume volatility
        features['volume_volatility_24h'] = df['volume'].rolling(24).std() / df['volume'].rolling(24).mean()

        # Spread proxy (high-low range / close)
        features['spread_proxy'] = (df['high'] - df['low']) / df['close']

        return features

    def _add_prediction_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add prediction features (LOOKAHEAD - separate pipeline)."""
        returns = df['close'].pct_change()

        # Next day return (24h forward)
        features['next_day_return'] = returns.shift(-24)

        # Next day volatility
        features['next_day_volatility'] = returns.rolling(24).std().shift(-24)

        return features

    def get_discovery_features(self) -> List[str]:
        """Get features for discovery pipeline (no lookahead, no circular)."""
        return [f for f in self.features_computed
                if f not in self.LOOKAHEAD_FEATURES
                and f not in self.CIRCULAR_FEATURES]

    def get_prediction_features(self) -> List[str]:
        """Get features for prediction pipeline."""
        return list(self.LOOKAHEAD_FEATURES)

    # Helper methods
    def _entropy(self, x):
        """Calculate entropy of return distribution."""
        if len(x) < 10:
            return np.nan
        hist, _ = np.histogram(x, bins=10, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist + 1e-10))

    def _rolling_hurst(self, returns, window):
        """Simplified Hurst exponent estimation."""
        def hurst(x):
            if len(x) < 20:
                return np.nan
            lags = range(2, min(20, len(x) // 2))
            tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
            if len(tau) < 2 or min(tau) <= 0:
                return np.nan
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]
        return returns.rolling(window).apply(hurst, raw=True)

    def _rsi(self, prices, period=14):
        """Relative Strength Index."""
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _adx(self, df, period=14):
        """Average Directional Index (simplified)."""
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(period).mean()

    def _atr(self, df, period=14):
        """Average True Range."""
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _bollinger_width(self, prices, period=20):
        """Bollinger Band width."""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        return (upper - lower) / ma
```

**Checkpoint**: Feature calculator created? ☐ Yes ☐ No

---

## Step 2.3: Download Data

```python
# src/data/download.py
"""
Download historical data from Bybit or other source.
"""
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def download_bybit(symbol: str = "BTCUSDT",
                   days: int = 365,
                   save_path: str = "data/bybit"):
    """
    Download hourly OHLCV data from Bybit.

    Note: You may need to use a different API or data source.
    This is a template.
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Placeholder: Replace with actual API call
    # For now, you can manually download from:
    # - Bybit: https://www.bybit.com/data/basic/spot/history
    # - Binance: https://data.binance.vision/
    # - CryptoDataDownload: https://www.cryptodatadownload.com/

    print(f"Please manually download {symbol} hourly data and save to:")
    print(f"  {save_path}/{symbol}_1h.csv")
    print(f"  Required columns: timestamp, open, high, low, close, volume")

    return None

if __name__ == "__main__":
    download_bybit("BTCUSDT", days=365)
    download_bybit("ETHUSDT", days=365)
```

**Checkpoint**: Data downloaded? ☐ Yes ☐ No

---

# PHASE 3: Backtest & SI Computation

## Step 3.1: Trading Strategies

Create `src/agents/strategies.py`:

```python
"""
Simple trading strategies for specialization testing.
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """Base class for trading strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def signal(self, data: pd.DataFrame, idx: int) -> float:
        """
        Generate trading signal.

        Returns: float in [-1, 1] where:
            -1 = full short
             0 = no position
            +1 = full long
        """
        pass


class MomentumStrategy(BaseStrategy):
    """Buy if price above MA, sell if below."""

    def __init__(self, lookback: int = 168):  # 7 days
        super().__init__(f"Momentum_{lookback}")
        self.lookback = lookback

    def signal(self, data: pd.DataFrame, idx: int) -> float:
        if idx < self.lookback:
            return 0.0

        window = data['close'].iloc[idx-self.lookback:idx]
        ma = window.mean()
        current = data['close'].iloc[idx]

        if current > ma * 1.01:  # 1% above MA
            return 1.0
        elif current < ma * 0.99:  # 1% below MA
            return -1.0
        return 0.0


class MeanReversionStrategy(BaseStrategy):
    """Buy when oversold, sell when overbought."""

    def __init__(self, lookback: int = 24, threshold: float = 2.0):
        super().__init__(f"MeanRev_{lookback}")
        self.lookback = lookback
        self.threshold = threshold

    def signal(self, data: pd.DataFrame, idx: int) -> float:
        if idx < self.lookback:
            return 0.0

        window = data['close'].iloc[idx-self.lookback:idx]
        ma = window.mean()
        std = window.std()
        current = data['close'].iloc[idx]

        z_score = (current - ma) / std if std > 0 else 0

        if z_score < -self.threshold:  # Oversold
            return 1.0
        elif z_score > self.threshold:  # Overbought
            return -1.0
        return 0.0


class BreakoutStrategy(BaseStrategy):
    """Buy on upward breakout, sell on downward."""

    def __init__(self, lookback: int = 48):
        super().__init__(f"Breakout_{lookback}")
        self.lookback = lookback

    def signal(self, data: pd.DataFrame, idx: int) -> float:
        if idx < self.lookback:
            return 0.0

        window = data.iloc[idx-self.lookback:idx]
        current = data['close'].iloc[idx]

        high = window['high'].max()
        low = window['low'].min()

        if current > high:  # Upward breakout
            return 1.0
        elif current < low:  # Downward breakout
            return -1.0
        return 0.0


# Create default strategy set
DEFAULT_STRATEGIES = [
    MomentumStrategy(lookback=168),
    MomentumStrategy(lookback=72),
    MeanReversionStrategy(lookback=24, threshold=2.0),
    MeanReversionStrategy(lookback=48, threshold=1.5),
    BreakoutStrategy(lookback=48),
    BreakoutStrategy(lookback=96),
]
```

---

## Step 3.2: NichePopulation Algorithm

Create `src/competition/niche_population.py`:

```python
"""
NichePopulation algorithm for emergent specialization.
Adapted from Paper 1.
"""
import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass, field

@dataclass
class Agent:
    """Trading agent with niche affinity tracking."""

    agent_id: int
    strategy_idx: int  # Which strategy this agent uses

    # Niche affinities: one per market regime
    niche_affinity: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))

    # Performance tracking
    cumulative_return: float = 0.0
    win_count: int = 0
    total_trades: int = 0

    def update_affinity(self, regime_idx: int, won: bool, alpha: float = 0.1):
        """Update niche affinity based on competition outcome."""
        if won:
            # Increase affinity for this regime
            self.niche_affinity[regime_idx] += alpha * (1 - self.niche_affinity[regime_idx])
        else:
            # Decrease affinity
            self.niche_affinity[regime_idx] *= (1 - alpha)

        # Normalize
        self.niche_affinity /= self.niche_affinity.sum()


class NichePopulation:
    """
    Population of competing agents that develop specialization.
    """

    def __init__(self, strategies: List, n_agents_per_strategy: int = 3):
        self.strategies = strategies
        self.agents: List[Agent] = []

        # Create agents
        agent_id = 0
        for strategy_idx in range(len(strategies)):
            for _ in range(n_agents_per_strategy):
                self.agents.append(Agent(
                    agent_id=agent_id,
                    strategy_idx=strategy_idx,
                    niche_affinity=np.ones(3) / 3  # Start uniform
                ))
                agent_id += 1

        self.history = []

    def classify_regime(self, data: pd.DataFrame, idx: int, lookback: int = 168) -> int:
        """
        Classify current market regime.

        Returns:
            0 = Trending
            1 = Mean-reverting
            2 = High volatility
        """
        if idx < lookback:
            return 0

        returns = data['close'].pct_change().iloc[idx-lookback:idx]

        vol = returns.std()
        trend = abs(returns.mean()) / vol if vol > 0 else 0
        autocorr = returns.autocorr(lag=1)

        # Simple classification
        if vol > returns.rolling(lookback*2).std().iloc[-1] * 1.5:
            return 2  # High volatility
        elif trend > 0.1:
            return 0  # Trending
        else:
            return 1  # Mean-reverting

    def compete(self, data: pd.DataFrame, idx: int) -> Dict:
        """
        Run one competition round.

        Returns:
            Dict with competition results and metrics
        """
        regime = self.classify_regime(data, idx)

        # Get signals and returns for each agent
        returns = []
        next_return = data['close'].pct_change().iloc[idx] if idx < len(data) - 1 else 0

        for agent in self.agents:
            strategy = self.strategies[agent.strategy_idx]
            signal = strategy.signal(data, idx - 1)  # Signal from previous bar
            agent_return = signal * next_return
            returns.append((agent, agent_return))

        # Winner-take-all: best return wins
        returns.sort(key=lambda x: x[1], reverse=True)
        winner = returns[0][0]

        # Update all agents
        for agent, agent_return in returns:
            won = (agent == winner)
            agent.update_affinity(regime, won)
            agent.cumulative_return += agent_return
            agent.total_trades += 1
            if won:
                agent.win_count += 1

        # Record history
        result = {
            'idx': idx,
            'regime': regime,
            'winner_id': winner.agent_id,
            'winner_strategy': winner.strategy_idx,
            'returns': [r[1] for r in returns],
        }
        self.history.append(result)

        return result

    def run(self, data: pd.DataFrame, start_idx: int = 200) -> pd.DataFrame:
        """Run competition over entire dataset."""
        for idx in range(start_idx, len(data)):
            self.compete(data, idx)

        return pd.DataFrame(self.history)

    def compute_si(self) -> float:
        """
        Compute Specialization Index for the population.

        SI = 1 - mean(entropy of niche affinities)

        High SI = agents are specialized (low entropy)
        Low SI = agents are generalists (high entropy)
        """
        entropies = []
        for agent in self.agents:
            # Entropy of niche affinity distribution
            p = agent.niche_affinity + 1e-10  # Avoid log(0)
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(len(p))  # Maximum possible entropy
            normalized_entropy = entropy / max_entropy
            entropies.append(normalized_entropy)

        mean_entropy = np.mean(entropies)
        si = 1 - mean_entropy

        return si

    def compute_si_timeseries(self, data: pd.DataFrame, window: int = 168) -> pd.Series:
        """Compute SI over time using rolling window."""
        if len(self.history) == 0:
            raise ValueError("Run competition first!")

        si_values = []
        indices = []

        for i in range(window, len(self.history)):
            # Reset affinities
            for agent in self.agents:
                agent.niche_affinity = np.ones(3) / 3

            # Replay window
            for j in range(i - window, i):
                record = self.history[j]
                regime = record['regime']
                winner_id = record['winner_id']

                for agent in self.agents:
                    won = (agent.agent_id == winner_id)
                    agent.update_affinity(regime, won)

            si_values.append(self.compute_si())
            indices.append(self.history[i]['idx'])

        return pd.Series(si_values, index=indices, name='si')
```

**Checkpoint**: NichePopulation implemented? ☐ Yes ☐ No

---

## Step 3.3: Run Backtest

Create `experiments/run_backtest.py`:

```python
"""
Run backtest and compute SI.
"""
import sys
sys.path.append('.')

from src.data.loader import DataLoader
from src.analysis.features import FeatureCalculator
from src.agents.strategies import DEFAULT_STRATEGIES
from src.competition.niche_population import NichePopulation
import pandas as pd
import json

def main():
    print("=" * 60)
    print("PHASE 3: BACKTEST & SI COMPUTATION")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data...")
    loader = DataLoader()
    data = loader.load("BTCUSDT")
    print(f"   Loaded {len(data)} rows")

    # 2. Split data
    print("\n2. Splitting data...")
    train, val, test = loader.temporal_split(data)

    # 3. Run competition on TRAIN only
    print("\n3. Running competition on TRAIN set...")
    population = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=3)
    history = population.run(train, start_idx=200)
    print(f"   Completed {len(history)} competition rounds")

    # 4. Compute SI timeseries
    print("\n4. Computing SI timeseries...")
    si = population.compute_si_timeseries(train, window=168)
    print(f"   SI mean: {si.mean():.3f}, std: {si.std():.3f}")

    # 5. Compute features
    print("\n5. Computing features...")
    calc = FeatureCalculator()
    features = calc.compute_all(train)
    print(f"   Computed {len(features.columns)} features")

    # 6. Align SI and features
    print("\n6. Aligning data...")
    common_idx = si.index.intersection(features.index)
    si_aligned = si.loc[common_idx]
    features_aligned = features.loc[common_idx]
    print(f"   Aligned {len(common_idx)} rows")

    # 7. Save results
    print("\n7. Saving results...")
    si_aligned.to_csv("results/si_correlations/si_train.csv")
    features_aligned.to_csv("results/si_correlations/features_train.csv")

    # Save metadata
    metadata = {
        'n_rows': len(common_idx),
        'si_mean': float(si_aligned.mean()),
        'si_std': float(si_aligned.std()),
        'n_features': len(features_aligned.columns),
        'discovery_features': calc.get_discovery_features(),
        'prediction_features': calc.get_prediction_features(),
    }
    with open("results/si_correlations/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE!")
    print(f"SI saved to: results/si_correlations/si_train.csv")
    print(f"Features saved to: results/si_correlations/features_train.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

```bash
python experiments/run_backtest.py
```

**Checkpoint**: Backtest complete? ☐ Yes ☐ No

---

# PHASE 4: Discovery Pipeline (46 Features)

## Step 4.1: Correlation Analyzer

Create `src/analysis/correlations.py`:

```python
"""
Correlation analysis with proper statistical methods.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple

class CorrelationAnalyzer:
    """
    Correlation analysis with:
    - HAC standard errors
    - Block bootstrap CIs
    - FDR correction
    """

    def __init__(self, block_size: int = 24):
        self.block_size = block_size

    def spearman_with_ci(self, x: pd.Series, y: pd.Series,
                         n_bootstrap: int = 1000) -> Dict:
        """
        Spearman correlation with block bootstrap confidence interval.
        """
        # Remove NaN
        mask = ~(x.isna() | y.isna())
        x, y = x[mask], y[mask]

        if len(x) < 50:
            return {'r': np.nan, 'p': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}

        # Point estimate
        r, p = spearmanr(x, y)

        # Block bootstrap
        n = len(x)
        n_blocks = n // self.block_size

        bootstrap_rs = []
        for _ in range(n_bootstrap):
            # Sample blocks with replacement
            block_starts = np.random.choice(n - self.block_size, n_blocks, replace=True)
            indices = np.concatenate([np.arange(s, s + self.block_size) for s in block_starts])
            indices = indices[:n]  # Trim to original length

            boot_r, _ = spearmanr(x.iloc[indices], y.iloc[indices])
            bootstrap_rs.append(boot_r)

        ci_low, ci_high = np.percentile(bootstrap_rs, [2.5, 97.5])

        return {
            'r': r,
            'p': p,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n': len(x)
        }

    def run_discovery(self, si: pd.Series, features: pd.DataFrame,
                      feature_list: List[str]) -> pd.DataFrame:
        """
        Run correlation analysis for discovery pipeline.

        Returns DataFrame with:
        - feature name
        - r (correlation)
        - p (p-value)
        - ci_low, ci_high (95% CI)
        - p_fdr (FDR-corrected p-value)
        - significant (True if p_fdr < 0.05 and |r| > 0.15)
        """
        results = []

        for feature in feature_list:
            if feature not in features.columns:
                continue

            result = self.spearman_with_ci(si, features[feature])
            result['feature'] = feature
            results.append(result)

        df = pd.DataFrame(results)

        # FDR correction
        if len(df) > 0 and not df['p'].isna().all():
            valid_mask = ~df['p'].isna()
            _, p_fdr, _, _ = multipletests(df.loc[valid_mask, 'p'], method='fdr_bh')
            df.loc[valid_mask, 'p_fdr'] = p_fdr
        else:
            df['p_fdr'] = np.nan

        # Significance
        df['significant'] = (df['p_fdr'] < 0.05) & (abs(df['r']) > 0.15)

        # Sort by absolute correlation
        df = df.sort_values('r', key=abs, ascending=False)

        return df
```

---

## Step 4.2: Run Discovery Pipeline

Create `experiments/run_discovery.py`:

```python
"""
Run discovery pipeline: What does SI correlate with?
"""
import sys
sys.path.append('.')

import pandas as pd
import json
from src.analysis.correlations import CorrelationAnalyzer

def main():
    print("=" * 60)
    print("PHASE 4: DISCOVERY PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading SI and features...")
    si = pd.read_csv("results/si_correlations/si_train.csv", index_col=0, parse_dates=True).squeeze()
    features = pd.read_csv("results/si_correlations/features_train.csv", index_col=0, parse_dates=True)

    with open("results/si_correlations/metadata.json") as f:
        metadata = json.load(f)

    discovery_features = metadata['discovery_features']
    print(f"   SI: {len(si)} rows")
    print(f"   Features: {len(discovery_features)} discovery features")

    # 2. Run correlation analysis
    print("\n2. Running correlation analysis...")
    analyzer = CorrelationAnalyzer(block_size=24)
    results = analyzer.run_discovery(si, features, discovery_features)

    # 3. Report results
    print("\n3. Results:")
    print("-" * 60)

    significant = results[results['significant']]
    print(f"\n   SIGNIFICANT CORRELATIONS ({len(significant)}):")
    print(significant[['feature', 'r', 'p_fdr', 'ci_low', 'ci_high']].to_string(index=False))

    print(f"\n   TOP 10 CORRELATIONS (by |r|):")
    print(results.head(10)[['feature', 'r', 'p_fdr', 'significant']].to_string(index=False))

    # 4. Save results
    print("\n4. Saving results...")
    results.to_csv("results/si_correlations/discovery_results.csv", index=False)

    # Save summary
    summary = {
        'n_features_tested': len(results),
        'n_significant': len(significant),
        'top_correlate': results.iloc[0]['feature'] if len(results) > 0 else None,
        'top_r': float(results.iloc[0]['r']) if len(results) > 0 else None,
        'significant_features': significant['feature'].tolist(),
    }
    with open("results/si_correlations/discovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("DISCOVERY COMPLETE!")
    print(f"Results saved to: results/si_correlations/discovery_results.csv")
    print("=" * 60)

    # 5. Decision point
    if len(significant) >= 3:
        print("\n✅ SUCCESS: Found 3+ significant correlations!")
        print("   NEXT: Validate on VAL set, then test on TEST set")
    else:
        print("\n⚠️  CAUTION: Found <3 significant correlations")
        print("   CONSIDER: Check data quality, try different SI variants")

if __name__ == "__main__":
    main()
```

```bash
python experiments/run_discovery.py
```

**Checkpoint**: Discovery pipeline complete? ☐ Yes ☐ No

---

# PHASE 5: Prediction Pipeline (2 Features)

## Step 5.1: Run Prediction Tests

Create `experiments/run_prediction.py`:

```python
"""
Run prediction pipeline: Does SI predict future outcomes?
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import json

def lagged_correlation(si: pd.Series, target: pd.Series, lag: int) -> float:
    """Correlation between SI(t) and target(t+lag)."""
    si_shifted = si.shift(lag)
    mask = ~(si_shifted.isna() | target.isna())
    r, _ = spearmanr(si_shifted[mask], target[mask])
    return r

def signal_decay(si: pd.Series, target: pd.Series, max_lag: int = 168) -> dict:
    """Analyze how SI's predictive power decays over time."""
    lags = list(range(1, max_lag + 1, 6))  # Every 6 hours
    rs = [lagged_correlation(si, target, lag) for lag in lags]

    # Find optimal lag
    best_idx = np.argmax(np.abs(rs))
    optimal_lag = lags[best_idx]

    # Estimate half-life
    peak_r = abs(rs[best_idx])
    half_r = peak_r / 2
    half_life = None
    for i in range(best_idx, len(rs)):
        if abs(rs[i]) < half_r:
            half_life = lags[i] - optimal_lag
            break

    return {
        'lags': lags,
        'correlations': rs,
        'optimal_lag': optimal_lag,
        'peak_r': rs[best_idx],
        'half_life': half_life
    }

def main():
    print("=" * 60)
    print("PHASE 5: PREDICTION PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading SI and features...")
    si = pd.read_csv("results/si_correlations/si_train.csv", index_col=0, parse_dates=True).squeeze()
    features = pd.read_csv("results/si_correlations/features_train.csv", index_col=0, parse_dates=True)

    prediction_targets = ['next_day_return', 'next_day_volatility']

    results = {}

    for target in prediction_targets:
        if target not in features.columns:
            print(f"   WARNING: {target} not found in features")
            continue

        print(f"\n2. Analyzing {target}...")

        # Signal decay
        decay = signal_decay(si, features[target])
        print(f"   Optimal lag: {decay['optimal_lag']} hours")
        print(f"   Peak correlation: {decay['peak_r']:.3f}")
        print(f"   Half-life: {decay['half_life']} hours")

        # Granger causality
        print(f"   Running Granger causality...")
        df = pd.concat([si, features[target]], axis=1).dropna()
        df.columns = ['si', 'target']

        try:
            granger = grangercausalitytests(df[['target', 'si']], maxlag=24, verbose=False)
            p_values = [granger[lag][0]['ssr_ftest'][1] for lag in range(1, 25)]
            min_p = min(p_values)
            best_lag = p_values.index(min_p) + 1

            print(f"   Granger p-value: {min_p:.4f} at lag {best_lag}")
            granger_significant = min_p < 0.05
        except Exception as e:
            print(f"   Granger test failed: {e}")
            granger_significant = False
            min_p = None

        results[target] = {
            'signal_decay': decay,
            'granger_p': min_p,
            'granger_significant': granger_significant
        }

    # 3. Save results
    print("\n3. Saving results...")
    with open("results/si_correlations/prediction_results.json", "w") as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj

        json.dump(results, f, indent=2, default=convert)

    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE COMPLETE!")
    print("=" * 60)

    # Summary
    for target, res in results.items():
        if res['granger_significant']:
            print(f"✅ SI Granger-causes {target} (p={res['granger_p']:.4f})")
        else:
            print(f"❌ SI does NOT Granger-cause {target}")

if __name__ == "__main__":
    main()
```

```bash
python experiments/run_prediction.py
```

**Checkpoint**: Prediction pipeline complete? ☐ Yes ☐ No

---

# PHASE 6: SI Dynamics Pipeline (9 Features)

## Step 6.1: SI Variants Analysis

Create `experiments/run_dynamics.py`:

```python
"""
Run SI dynamics pipeline: How should we use SI?
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import json

def compute_si_variants(si: pd.Series) -> pd.DataFrame:
    """Compute different SI variants."""
    variants = pd.DataFrame(index=si.index)

    variants['si_raw'] = si
    variants['si_1h'] = si.rolling(1).mean()
    variants['si_4h'] = si.rolling(4).mean()
    variants['si_1d'] = si.rolling(24).mean()
    variants['si_1w'] = si.rolling(168).mean()

    # Derivatives
    variants['dsi_dt'] = si.diff()
    variants['si_acceleration'] = variants['dsi_dt'].diff()

    # Stability
    variants['si_std'] = si.rolling(24).std()

    # Percentile
    variants['si_percentile'] = si.rolling(720).rank(pct=True)

    return variants

def main():
    print("=" * 60)
    print("PHASE 6: SI DYNAMICS PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading SI and features...")
    si = pd.read_csv("results/si_correlations/si_train.csv", index_col=0, parse_dates=True).squeeze()
    features = pd.read_csv("results/si_correlations/features_train.csv", index_col=0, parse_dates=True)

    # Use next_day_return as profit proxy
    profit = features['next_day_return'] if 'next_day_return' in features.columns else features.iloc[:, 0]

    # 2. Compute SI variants
    print("\n2. Computing SI variants...")
    variants = compute_si_variants(si)
    print(f"   Created {len(variants.columns)} variants")

    # 3. Compare variants
    print("\n3. Comparing variants with profit...")
    variant_results = {}

    for col in variants.columns:
        mask = ~(variants[col].isna() | profit.isna())
        if mask.sum() < 100:
            continue

        r, p = spearmanr(variants[col][mask], profit[mask])
        variant_results[col] = {'r': r, 'p': p}

    # Sort by absolute correlation
    sorted_variants = sorted(variant_results.items(), key=lambda x: abs(x[1]['r']), reverse=True)

    print("\n   VARIANT PERFORMANCE:")
    for name, res in sorted_variants:
        print(f"   {name:20s} r={res['r']:+.3f}  p={res['p']:.4f}")

    best_variant = sorted_variants[0][0]
    print(f"\n   BEST VARIANT: {best_variant}")

    # 4. Momentum analysis
    print("\n4. Momentum analysis (does dSI/dt matter?)...")
    dsi = variants['dsi_dt'].dropna()
    profit_aligned = profit.loc[dsi.index]

    rising_si = dsi > 0
    profit_rising = profit_aligned[rising_si].mean()
    profit_falling = profit_aligned[~rising_si].mean()

    print(f"   Profit when SI rising: {profit_rising:.4f}")
    print(f"   Profit when SI falling: {profit_falling:.4f}")
    print(f"   Difference: {profit_rising - profit_falling:.4f}")

    momentum_helps = profit_rising > profit_falling

    # 5. Stability analysis
    print("\n5. Stability analysis (does SI volatility matter?)...")
    si_std = variants['si_std'].dropna()
    profit_aligned = profit.loc[si_std.index]

    stable = si_std < si_std.median()
    profit_stable = profit_aligned[stable].mean()
    profit_volatile = profit_aligned[~stable].mean()

    print(f"   Profit when SI stable: {profit_stable:.4f}")
    print(f"   Profit when SI volatile: {profit_volatile:.4f}")

    stability_helps = profit_stable > profit_volatile

    # 6. Extremes analysis
    print("\n6. Extremes analysis (do SI extremes predict reversals?)...")
    si_pct = variants['si_percentile'].dropna()
    profit_aligned = profit.loc[si_pct.index]

    extreme_high = si_pct > 0.9
    extreme_low = si_pct < 0.1
    normal = ~extreme_high & ~extreme_low

    profit_after_high = profit_aligned[extreme_high.shift(24).fillna(False)].mean()
    profit_after_low = profit_aligned[extreme_low.shift(24).fillna(False)].mean()
    profit_normal = profit_aligned[normal].mean()

    print(f"   Profit after extreme high SI: {profit_after_high:.4f}")
    print(f"   Profit after extreme low SI: {profit_after_low:.4f}")
    print(f"   Profit during normal SI: {profit_normal:.4f}")

    # 7. Save results
    results = {
        'best_variant': best_variant,
        'variant_correlations': {k: {'r': v['r'], 'p': v['p']} for k, v in variant_results.items()},
        'momentum_effect': {
            'profit_when_rising': float(profit_rising),
            'profit_when_falling': float(profit_falling),
            'momentum_helps': momentum_helps
        },
        'stability_effect': {
            'profit_when_stable': float(profit_stable),
            'profit_when_volatile': float(profit_volatile),
            'stability_helps': stability_helps
        },
        'extremes_effect': {
            'profit_after_extreme_high': float(profit_after_high) if not np.isnan(profit_after_high) else None,
            'profit_after_extreme_low': float(profit_after_low) if not np.isnan(profit_after_low) else None,
            'profit_normal': float(profit_normal)
        }
    }

    with open("results/si_correlations/dynamics_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SI DYNAMICS COMPLETE!")
    print(f"Best SI variant: {best_variant}")
    print(f"Momentum helps: {momentum_helps}")
    print(f"Stability helps: {stability_helps}")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

```bash
python experiments/run_dynamics.py
```

**Checkpoint**: SI Dynamics complete? ☐ Yes ☐ No

---

# PHASE 7: Validation & Audits

## Step 7.1: Validate on VAL Set

```python
# experiments/run_validation.py
"""
Validate discovery findings on validation set.
"""
import sys
sys.path.append('.')

import pandas as pd
import json
from src.data.loader import DataLoader
from src.analysis.features import FeatureCalculator
from src.agents.strategies import DEFAULT_STRATEGIES
from src.competition.niche_population import NichePopulation
from src.analysis.correlations import CorrelationAnalyzer

def main():
    print("=" * 60)
    print("PHASE 7: VALIDATION")
    print("=" * 60)

    # 1. Load candidates from discovery
    with open("results/si_correlations/discovery_summary.json") as f:
        discovery = json.load(f)

    candidates = discovery['significant_features']
    print(f"\n1. Candidates to validate: {candidates}")

    if len(candidates) == 0:
        print("   No candidates to validate!")
        return

    # 2. Load VAL data
    loader = DataLoader()
    data = loader.load("BTCUSDT")
    _, val, _ = loader.temporal_split(data)

    # 3. Run competition on VAL
    print("\n2. Running competition on VAL set...")
    population = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=3)
    population.run(val, start_idx=200)
    si = population.compute_si_timeseries(val, window=168)

    # 4. Compute features on VAL
    calc = FeatureCalculator()
    features = calc.compute_all(val)

    # 5. Align
    common_idx = si.index.intersection(features.index)
    si = si.loc[common_idx]
    features = features.loc[common_idx]

    # 6. Validate candidates
    print("\n3. Validating candidates...")
    analyzer = CorrelationAnalyzer()

    # Load train results for comparison
    train_results = pd.read_csv("results/si_correlations/discovery_results.csv")

    confirmed = []
    for feature in candidates:
        result = analyzer.spearman_with_ci(si, features[feature])

        # Get train direction
        train_r = train_results[train_results['feature'] == feature]['r'].values[0]

        # Confirm: same direction AND p < 0.05
        same_direction = (result['r'] * train_r) > 0
        significant = result['p'] < 0.05

        status = "✅ CONFIRMED" if (same_direction and significant) else "❌ NOT CONFIRMED"
        print(f"   {feature}: r={result['r']:.3f} (train: {train_r:.3f}) {status}")

        if same_direction and significant:
            confirmed.append(feature)

    # 7. Save results
    validation_results = {
        'candidates_tested': candidates,
        'confirmed': confirmed,
        'confirmation_rate': len(confirmed) / len(candidates) if candidates else 0
    }

    with open("results/si_correlations/validation_results.json", "w") as f:
        json.dump(validation_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"VALIDATION COMPLETE!")
    print(f"Confirmed: {len(confirmed)}/{len(candidates)} candidates")
    print(f"Confirmed features: {confirmed}")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## Step 7.2: Run Audits

Create `experiments/run_audits.py`:

```python
"""
Run all 8 expert-recommended audits.
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import json

def audit_causal(si, feature):
    """Causal inference audit."""
    df = pd.concat([si, feature], axis=1).dropna()
    df.columns = ['si', 'feature']

    results = {}

    # SI → Feature
    try:
        test = grangercausalitytests(df[['feature', 'si']], maxlag=24, verbose=False)
        p_values = [test[lag][0]['ssr_ftest'][1] for lag in range(1, 25)]
        results['si_causes_feature'] = min(p_values) < 0.05
    except:
        results['si_causes_feature'] = None

    # Feature → SI
    try:
        test = grangercausalitytests(df[['si', 'feature']], maxlag=24, verbose=False)
        p_values = [test[lag][0]['ssr_ftest'][1] for lag in range(1, 25)]
        results['feature_causes_si'] = min(p_values) < 0.05
    except:
        results['feature_causes_si'] = None

    # Placebo: random SI
    real_r, _ = spearmanr(si, feature)
    placebo_rs = []
    for _ in range(100):
        fake_si = pd.Series(np.random.randn(len(si)), index=si.index)
        r, _ = spearmanr(fake_si, feature)
        placebo_rs.append(r)

    results['placebo_95_ci'] = list(np.percentile(placebo_rs, [2.5, 97.5]))
    results['real_exceeds_placebo'] = abs(real_r) > np.percentile(np.abs(placebo_rs), 95)

    return results

def audit_permutation(si, feature, n_perm=1000):
    """Permutation test."""
    real_r, _ = spearmanr(si, feature)

    shuffled_rs = []
    for _ in range(n_perm):
        shuffled_si = np.random.permutation(si.values)
        r, _ = spearmanr(shuffled_si, feature)
        shuffled_rs.append(r)

    p = np.mean(np.abs(shuffled_rs) >= np.abs(real_r))

    return {
        'real_r': real_r,
        'permutation_p': p,
        'significant': p < 0.05
    }

def audit_crypto(data, si, feature_name):
    """Crypto-specific audit."""
    feature = data[feature_name] if feature_name in data.columns else None
    if feature is None:
        return {'error': 'Feature not found'}

    results = {}

    # Time of day
    hour = data.index.hour
    for session, (start, end) in [('asian', (0, 8)), ('eu', (8, 16)), ('us', (16, 24))]:
        mask = (hour >= start) & (hour < end)
        if mask.sum() > 100:
            r, p = spearmanr(si[mask], feature[mask])
            results[f'{session}_r'] = r

    # Weekend
    is_weekend = data.index.dayofweek >= 5
    if (~is_weekend).sum() > 100 and is_weekend.sum() > 50:
        r_weekday, _ = spearmanr(si[~is_weekend], feature[~is_weekend])
        r_weekend, _ = spearmanr(si[is_weekend], feature[is_weekend])
        results['weekday_r'] = r_weekday
        results['weekend_r'] = r_weekend
        results['consistent'] = np.sign(r_weekday) == np.sign(r_weekend)

    return results

def main():
    print("=" * 60)
    print("PHASE 7B: AUDITS")
    print("=" * 60)

    # Load data
    si = pd.read_csv("results/si_correlations/si_train.csv", index_col=0, parse_dates=True).squeeze()
    features = pd.read_csv("results/si_correlations/features_train.csv", index_col=0, parse_dates=True)

    # Load confirmed features
    with open("results/si_correlations/validation_results.json") as f:
        validation = json.load(f)

    confirmed = validation.get('confirmed', [])
    if not confirmed:
        confirmed = ['volatility_7d']  # Fallback

    audit_results = {}

    for feature_name in confirmed[:3]:  # Top 3
        print(f"\nAuditing: {feature_name}")
        feature = features[feature_name]

        # Align
        mask = ~(si.isna() | feature.isna())
        si_clean = si[mask]
        feat_clean = feature[mask]

        print("  Running causal audit...")
        audit_results[f'{feature_name}_causal'] = audit_causal(si_clean, feat_clean)

        print("  Running permutation audit...")
        audit_results[f'{feature_name}_permutation'] = audit_permutation(si_clean, feat_clean)

        print("  Running crypto audit...")
        features_with_idx = features.copy()
        features_with_idx['si'] = si
        audit_results[f'{feature_name}_crypto'] = audit_crypto(features_with_idx, si, feature_name)

    # Save
    with open("results/si_correlations/audit_results.json", "w") as f:
        json.dump(audit_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print("\n" + "=" * 60)
    print("AUDITS COMPLETE!")
    print("Results saved to: results/si_correlations/audit_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

```bash
python experiments/run_audits.py
```

**Checkpoint**: All audits complete? ☐ Yes ☐ No

---

# PHASE 8: Final Report

## Step 8.1: Generate Report

Create `experiments/generate_report.py`:

```python
"""
Generate final report.
"""
import json
import pandas as pd

def main():
    print("=" * 60)
    print("FINAL REPORT: SI SIGNAL DISCOVERY")
    print("=" * 60)

    # Load all results
    with open("results/si_correlations/discovery_summary.json") as f:
        discovery = json.load(f)

    with open("results/si_correlations/prediction_results.json") as f:
        prediction = json.load(f)

    with open("results/si_correlations/dynamics_results.json") as f:
        dynamics = json.load(f)

    with open("results/si_correlations/validation_results.json") as f:
        validation = json.load(f)

    with open("results/si_correlations/audit_results.json") as f:
        audits = json.load(f)

    # Report
    print("\n" + "=" * 60)
    print("1. DISCOVERY RESULTS")
    print("=" * 60)
    print(f"Features tested: {discovery['n_features_tested']}")
    print(f"Significant correlations: {discovery['n_significant']}")
    print(f"Top correlate: {discovery['top_correlate']} (r={discovery['top_r']:.3f})")
    print(f"Significant features: {discovery['significant_features']}")

    print("\n" + "=" * 60)
    print("2. VALIDATION RESULTS")
    print("=" * 60)
    print(f"Candidates tested: {len(validation['candidates_tested'])}")
    print(f"Confirmed: {len(validation['confirmed'])}")
    print(f"Confirmation rate: {validation['confirmation_rate']:.0%}")
    print(f"Confirmed features: {validation['confirmed']}")

    print("\n" + "=" * 60)
    print("3. PREDICTION RESULTS")
    print("=" * 60)
    for target, res in prediction.items():
        granger = "✅ SI Granger-causes" if res.get('granger_significant') else "❌ No causality"
        print(f"{target}: {granger}")
        if res.get('signal_decay'):
            print(f"  Optimal lag: {res['signal_decay']['optimal_lag']} hours")
            print(f"  Half-life: {res['signal_decay']['half_life']} hours")

    print("\n" + "=" * 60)
    print("4. SI DYNAMICS RESULTS")
    print("=" * 60)
    print(f"Best SI variant: {dynamics['best_variant']}")
    print(f"Momentum helps: {dynamics['momentum_effect']['momentum_helps']}")
    print(f"Stability helps: {dynamics['stability_effect']['stability_helps']}")

    print("\n" + "=" * 60)
    print("5. AUDIT RESULTS")
    print("=" * 60)
    for audit_name, result in audits.items():
        if 'permutation' in audit_name:
            sig = "✅ PASS" if result.get('significant') else "❌ FAIL"
            print(f"{audit_name}: {sig}")
        elif 'causal' in audit_name:
            si_causes = "✅" if result.get('si_causes_feature') else "❌"
            feat_causes = "✅" if result.get('feature_causes_si') else "❌"
            print(f"{audit_name}: SI→Feature={si_causes}, Feature→SI={feat_causes}")

    print("\n" + "=" * 60)
    print("6. CONCLUSION")
    print("=" * 60)

    # Decision logic
    if len(validation['confirmed']) >= 3:
        print("✅ PRIMARY HYPOTHESIS SUPPORTED")
        print(f"   SI correlates with: {validation['confirmed']}")
    else:
        print("⚠️  PRIMARY HYPOTHESIS PARTIALLY SUPPORTED")
        print(f"   Only {len(validation['confirmed'])} features confirmed")

    if any(res.get('granger_significant') for res in prediction.values()):
        print("✅ SI HAS PREDICTIVE POWER")
    else:
        print("⚠️  SI PREDICTIVE POWER NOT CONFIRMED")

    print("\n" + "=" * 60)
    print("7. NEXT STEPS")
    print("=" * 60)
    if len(validation['confirmed']) > 0:
        print("→ Test on TEST set")
        print("→ Implement trading strategy using SI")
        print("→ Paper writing")
    else:
        print("→ Try different SI variants")
        print("→ Check data quality")
        print("→ Consider alternative hypotheses")

if __name__ == "__main__":
    main()
```

```bash
python experiments/generate_report.py
```

---

# PHASE 9: Cross-Market Validation (CRITICAL)

> **Goal**: Verify SI findings generalize beyond crypto to other market types.

## Step 9.1: Run on Multiple Markets

Create `experiments/run_cross_market.py`:

```python
"""
Cross-market validation: Does SI work beyond crypto?
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import json
from scipy.stats import spearmanr

from src.data.loader import MultiMarketLoader, MarketType
from src.data.validation import DataValidator
from src.analysis.features import FeatureCalculator
from src.agents.strategies import DEFAULT_STRATEGIES
from src.competition.niche_population import NichePopulation
from src.analysis.correlations import CorrelationAnalyzer

# Markets to test
MARKETS = {
    MarketType.CRYPTO: ['BTCUSDT', 'ETHUSDT'],
    MarketType.FOREX: ['EURUSD', 'GBPUSD'],
    MarketType.STOCKS: ['SPY', 'QQQ'],
    MarketType.COMMODITIES: ['GOLD'],
}

def run_single_asset(market_type: MarketType, symbol: str) -> dict:
    """Run full pipeline on a single asset."""
    print(f"\n{'='*60}")
    print(f"Processing: {market_type.value}/{symbol}")
    print('='*60)

    try:
        # 1. Load data
        loader = MultiMarketLoader(market_type)
        data = loader.load(symbol)

        # 2. Validate
        validator = DataValidator(strict=False)
        validation = validator.validate(data, f"{market_type.value}/{symbol}")

        if not validation['valid']:
            return {
                'symbol': symbol,
                'market': market_type.value,
                'status': 'INVALID_DATA',
                'issues': validation['issues']
            }

        # 3. Split
        train, val, test = loader.temporal_split(data)

        # 4. Run competition
        population = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=3)
        population.run(train, start_idx=200)
        si = population.compute_si_timeseries(train, window=168)

        # 5. Compute features
        calc = FeatureCalculator()
        features = calc.compute_all(train)

        # 6. Align
        common_idx = si.index.intersection(features.index)
        si = si.loc[common_idx]
        features = features.loc[common_idx]

        # 7. Run discovery (top 5 only for speed)
        analyzer = CorrelationAnalyzer()
        discovery_features = calc.get_discovery_features()[:20]  # Subset for speed
        results = analyzer.run_discovery(si, features, discovery_features)

        # 8. Summary
        significant = results[results['significant']]

        return {
            'symbol': symbol,
            'market': market_type.value,
            'status': 'SUCCESS',
            'n_rows': len(common_idx),
            'si_mean': float(si.mean()),
            'si_std': float(si.std()),
            'n_significant': len(significant),
            'top_feature': results.iloc[0]['feature'] if len(results) > 0 else None,
            'top_r': float(results.iloc[0]['r']) if len(results) > 0 else None,
            'significant_features': significant['feature'].tolist(),
        }

    except Exception as e:
        return {
            'symbol': symbol,
            'market': market_type.value,
            'status': 'ERROR',
            'error': str(e)
        }


def main():
    print("="*60)
    print("PHASE 9: CROSS-MARKET VALIDATION")
    print("="*60)

    all_results = {}

    for market_type, symbols in MARKETS.items():
        for symbol in symbols:
            result = run_single_asset(market_type, symbol)
            all_results[f"{market_type.value}/{symbol}"] = result

    # Summary
    print("\n" + "="*60)
    print("CROSS-MARKET SUMMARY")
    print("="*60)

    # Group by market
    market_summary = {}
    for key, result in all_results.items():
        market = result['market']
        if market not in market_summary:
            market_summary[market] = {'success': 0, 'total': 0, 'significant': 0}

        market_summary[market]['total'] += 1
        if result['status'] == 'SUCCESS':
            market_summary[market]['success'] += 1
            market_summary[market]['significant'] += result['n_significant']

    print("\nBy Market:")
    for market, stats in market_summary.items():
        print(f"  {market}: {stats['success']}/{stats['total']} assets, "
              f"{stats['significant']} total significant correlations")

    # Find common features
    print("\n" + "="*60)
    print("COMMON FEATURES ACROSS MARKETS")
    print("="*60)

    all_significant = []
    for key, result in all_results.items():
        if result['status'] == 'SUCCESS':
            all_significant.extend(result.get('significant_features', []))

    from collections import Counter
    feature_counts = Counter(all_significant)

    print("\nFeatures significant in multiple assets:")
    for feature, count in feature_counts.most_common(10):
        if count >= 2:
            print(f"  {feature}: {count} assets")

    # Save results
    with open("results/si_correlations/cross_market_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("CROSS-MARKET CONCLUSION")
    print("="*60)

    # Check if SI works across market types
    markets_with_findings = sum(1 for m, s in market_summary.items() if s['significant'] > 0)

    if markets_with_findings >= 3:
        print("✅ SI FINDINGS GENERALIZE ACROSS MARKETS!")
        print(f"   Found significant correlations in {markets_with_findings}/4 market types")
    elif markets_with_findings >= 2:
        print("⚠️  SI PARTIALLY GENERALIZES")
        print(f"   Found significant correlations in {markets_with_findings}/4 market types")
    else:
        print("❌ SI MAY BE MARKET-SPECIFIC")
        print(f"   Only found correlations in {markets_with_findings}/4 market types")

    # Common feature check
    universal_features = [f for f, c in feature_counts.items() if c >= 3]
    if len(universal_features) > 0:
        print(f"\n✅ UNIVERSAL SI CORRELATES: {universal_features}")
    else:
        print("\n⚠️  No features significant across 3+ assets")

if __name__ == "__main__":
    main()
```

```bash
python experiments/run_cross_market.py
```

**Checkpoint**: Cross-market validation complete? ☐ Yes ☐ No

---

## Step 9.2: Interpret Cross-Market Results

| Outcome | Interpretation | Next Step |
|---------|----------------|-----------|
| SI works in 4/4 markets | **Universal signal!** | Paper: "SI as universal market metric" |
| SI works in 2-3 markets | Market-dependent | Focus on working markets |
| SI works only in crypto | Crypto-specific | Reframe scope |
| SI works only in stocks | Equity-specific | Pivot market focus |

---

## Step 9.3: Market-Specific Considerations

| Market | Key Differences | Adaptation Needed |
|--------|-----------------|-------------------|
| **Crypto** | 24/7, no close, high vol | Base case |
| **Forex** | Weekend gaps, macro events | Handle weekend gaps |
| **Stocks** | 6.5h/day, earnings, dividends | Overnight handling |
| **Commodities** | Seasonal, supply shocks | Seasonal features |

```python
# Market-specific feature adaptations
def adapt_features_for_market(features: pd.DataFrame, market_type: MarketType) -> pd.DataFrame:
    """Adapt features based on market characteristics."""

    if market_type == MarketType.STOCKS:
        # Add overnight return feature
        features['overnight_gap'] = ...

        # Remove 24h rolling (doesn't make sense)
        # Use trading-day rolling instead

    elif market_type == MarketType.FOREX:
        # Handle weekend gaps
        features = features[features.index.dayofweek < 5]

        # Volume is unreliable - remove volume features
        vol_cols = [c for c in features.columns if 'volume' in c.lower()]
        features = features.drop(columns=vol_cols)

    elif market_type == MarketType.COMMODITIES:
        # Add seasonal features
        features['month'] = features.index.month
        features['is_roll_period'] = ...  # Contract roll periods

    return features
```

---

# 📋 QUICK REFERENCE

## Commands

```bash
# Phase 1: Pre-Registration
git add experiments/pre_registration.json
git commit -m "PRE-REGISTRATION: SI hypothesis"
git push origin main

# Phase 2: Data Validation (CRITICAL - run first!)
python experiments/validate_data.py

# Phase 3: Backtest
python experiments/run_backtest.py

# Phase 4-6: Analysis Pipelines
python experiments/run_discovery.py
python experiments/run_prediction.py
python experiments/run_dynamics.py

# Phase 7: Validation & Audits
python experiments/run_validation.py
python experiments/run_audits.py

# Phase 8: Report
python experiments/generate_report.py

# Phase 9: Cross-Market
python experiments/run_cross_market.py
```

## Key Files

| File | Purpose |
|------|---------|
| `experiments/pre_registration.json` | Commit first! |
| `src/data/loader.py` | Multi-market data loader |
| `src/data/validation.py` | Data validation |
| `src/analysis/features.py` | 46 features |
| `src/competition/niche_population.py` | SI computation |
| `src/analysis/correlations.py` | Statistical tests |
| `results/si_correlations/*.json` | All results |

## Market Data Locations

| Market | Directory | Example |
|--------|-----------|---------|
| Crypto | `data/crypto/` | `BTCUSDT_1h.csv` |
| Forex | `data/forex/` | `EURUSD_1h.csv` |
| Stocks | `data/stocks/` | `SPY_1h.csv` |
| Commodities | `data/commodities/` | `GOLD_1h.csv` |

---

**END OF MASTER PLAN**
