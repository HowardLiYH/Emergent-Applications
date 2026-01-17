# SI Correlation Test Plan: 70+ Features

**Date**: January 17, 2026
**Purpose**: Systematic plan to discover what SI actually correlates with

---

## 🎯 Goal

Run ONE comprehensive backtest, compute SI and 70+ features, then systematically discover:
1. What SI correlates with MOST strongly
2. Which correlations are statistically significant
3. How to interpret findings and trace to profit

---

## 📊 Phase 1: Data Infrastructure (Day 1-2)

### 1.1 Data Requirements

| Data Type | Source | Timeframe | Granularity |
|-----------|--------|-----------|-------------|
| **Price Data** | Bybit/PopAgent | 6-12 months | 1-hour candles |
| **Assets** | BTC, ETH, SOL | Multi-asset | |
| **Features** | Computed | Rolling windows | 1d, 7d, 30d |

### 1.2 Directory Structure

```
apps/trading/
├── src/
│   ├── data/
│   │   ├── loader.py          # Load price data
│   │   └── features.py        # Compute all features
│   ├── agents/
│   │   ├── base.py            # Agent base class
│   │   ├── strategies.py      # 3 strategies
│   │   └── thompson.py        # Thompson Sampling
│   ├── competition/
│   │   ├── niche_population.py  # NichePopulation algorithm
│   │   └── si_calculator.py     # Compute SI
│   ├── analysis/
│   │   ├── correlation.py     # Correlation analysis
│   │   ├── visualization.py   # Plots
│   │   └── interpretation.py  # Auto-interpret results
│   └── backtest/
│       └── runner.py          # Main backtest loop
├── experiments/
│   └── phase0_si_discovery.py # Main experiment script
├── results/
│   └── si_correlations/       # Output directory
└── data/
    └── bybit/                 # Raw price data
```

### 1.3 Core Classes

```python
# src/data/features.py

class FeatureCalculator:
    """Compute all 70+ features from price data and agent states."""

    def __init__(self, price_data: pd.DataFrame, window_sizes: List[int] = [24, 168, 720]):
        self.price_data = price_data
        self.windows = window_sizes  # 1d, 7d, 30d in hours

    def compute_all_features(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Compute all features at a given timestamp."""
        features = {}

        # Category 1-3: Market State & Microstructure
        features.update(self._compute_market_state_features(timestamp))

        # Category 6: Risk Metrics
        features.update(self._compute_risk_features(timestamp))

        # Category 7: Timing Features
        features.update(self._compute_timing_features(timestamp))

        # Category 8: Behavioral Features
        features.update(self._compute_behavioral_features(timestamp))

        # Category 9: Factor Features
        features.update(self._compute_factor_features(timestamp))

        return features
```

---

## 📊 Phase 2: Feature Computation (Day 3-4)

### 2.1 Feature Categories & Formulas

#### Category 1-3: Market State & Microstructure (15 features)

| Feature | Formula | Window | Hypothesis |
|---------|---------|--------|------------|
| `volatility` | `std(returns) * sqrt(periods)` | 24h, 7d | H1, H7 |
| `trend_strength` | `abs(return) / volatility` | 7d | H1 |
| `return_autocorr` | `corr(r_t, r_{t-1})` | 7d | H2 |
| `hurst_exponent` | R/S analysis | 30d | H2 |
| `return_entropy` | `-sum(p * log(p))` | 7d | H2 |
| `regime_duration` | Days since last vol regime change | N/A | H3 |
| `volume` | `mean(volume)` | 24h | H7 |
| `volume_volatility` | `std(volume) / mean(volume)` | 7d | H7 |
| `jump_frequency` | Count of |r| > 3σ | 7d | H8 |
| `variance_ratio` | `var(r_k) / (k * var(r_1))` | 7d | H8 |
| `adx` | Average Directional Index | 14d | H24 |
| `bb_width` | Bollinger Band width | 20d | H24 |
| `rsi` | Relative Strength Index | 14d | H13 |
| `macd_hist` | MACD histogram | 12/26/9 | H1 |
| `atr` | Average True Range | 14d | H7 |

```python
def _compute_market_state_features(self, timestamp: pd.Timestamp) -> Dict[str, float]:
    features = {}

    for window in self.windows:
        window_data = self._get_window(timestamp, window)
        returns = window_data['close'].pct_change().dropna()

        # Volatility
        features[f'volatility_{window}h'] = returns.std() * np.sqrt(window)

        # Trend Strength
        total_return = (window_data['close'].iloc[-1] / window_data['close'].iloc[0]) - 1
        features[f'trend_strength_{window}h'] = abs(total_return) / (returns.std() + 1e-8)

        # Return Autocorrelation
        if len(returns) > 2:
            features[f'return_autocorr_{window}h'] = returns.autocorr(lag=1)

        # Return Entropy (discretize returns into bins)
        bins = pd.cut(returns, bins=10, labels=False)
        probs = bins.value_counts(normalize=True)
        features[f'return_entropy_{window}h'] = -np.sum(probs * np.log(probs + 1e-8))

        # Volume features
        features[f'volume_{window}h'] = window_data['volume'].mean()
        features[f'volume_volatility_{window}h'] = window_data['volume'].std() / (window_data['volume'].mean() + 1e-8)

        # Jump frequency (|return| > 3 sigma)
        threshold = returns.std() * 3
        features[f'jump_frequency_{window}h'] = (np.abs(returns) > threshold).sum()

    # Hurst Exponent (longer window needed)
    features['hurst_exponent'] = self._compute_hurst(timestamp)

    # Variance Ratio
    features['variance_ratio'] = self._compute_variance_ratio(timestamp)

    # Technical Indicators
    features['adx'] = self._compute_adx(timestamp)
    features['bb_width'] = self._compute_bollinger_width(timestamp)
    features['rsi'] = self._compute_rsi(timestamp)
    features['atr'] = self._compute_atr(timestamp)

    return features
```

#### Category 4-5: Agent Behavior (10 features)

| Feature | Formula | Source | Hypothesis |
|---------|---------|--------|------------|
| `agent_correlation` | Mean pairwise corr of agent returns | Agents | H4 |
| `effective_n` | 1 / sum(w_i^2) | Agents | H4 |
| `winner_consistency` | % same winner in rolling window | Agents | H6 |
| `winner_spread` | Best return - Worst return | Agents | H9 |
| `viable_agent_count` | Agents with positive return | Agents | H9 |
| `agent_confidence_mean` | Mean Thompson posterior width | Agents | H5 |
| `niche_affinity_entropy` | Entropy of niche assignments | Agents | H6 |
| `strategy_concentration` | HHI of strategy usage | Agents | H18 |
| `position_correlation` | Corr of agent positions | Agents | H18 |
| `return_dispersion` | Std of agent returns | Agents | H9 |

```python
def compute_agent_features(self, agents: List[Agent], period_returns: Dict[int, float]) -> Dict[str, float]:
    features = {}

    returns = np.array(list(period_returns.values()))

    # Agent correlation (pairwise)
    if len(agents) >= 2:
        correlations = []
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                # Use historical returns for correlation
                corr = np.corrcoef(agents[i].return_history[-30:],
                                   agents[j].return_history[-30:])[0,1]
                correlations.append(corr)
        features['agent_correlation'] = np.mean(correlations)

    # Effective N (diversification)
    weights = np.array([a.capital / sum(a.capital for a in agents) for a in agents])
    features['effective_n'] = 1.0 / np.sum(weights ** 2)

    # Winner spread
    features['winner_spread'] = returns.max() - returns.min()

    # Viable agents
    features['viable_agent_count'] = (returns > 0).sum()

    # Return dispersion
    features['return_dispersion'] = returns.std()

    # Strategy concentration (HHI)
    strategy_counts = Counter(a.current_strategy for a in agents)
    total = sum(strategy_counts.values())
    shares = [c / total for c in strategy_counts.values()]
    features['strategy_concentration'] = sum(s**2 for s in shares)

    return features
```

#### Category 6: Risk Metrics (10 features)

| Feature | Formula | Window | Hypothesis |
|---------|---------|--------|------------|
| `max_drawdown` | Max peak-to-trough | 30d | H21, H22 |
| `var_95` | 5th percentile of returns | 30d | H21 |
| `cvar_95` | Mean of returns below VaR | 30d | H21 |
| `volatility_of_volatility` | Std of rolling volatility | 30d | H21 |
| `tail_ratio` | 95th percentile / |5th| | 30d | H21 |
| `drawdown_recovery_days` | Days to recover from DD | History | H22 |
| `win_rate` | % of winning trades | 30d | H19 |
| `profit_factor` | Gross profit / Gross loss | 30d | H19 |
| `sharpe_ratio` | Mean / Std of returns | 30d | H23 |
| `sortino_ratio` | Mean / Downside Std | 30d | H23 |

```python
def _compute_risk_features(self, returns: pd.Series, equity_curve: pd.Series) -> Dict[str, float]:
    features = {}

    # Max Drawdown
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    features['max_drawdown'] = drawdowns.min()

    # VaR and CVaR
    features['var_95'] = np.percentile(returns, 5)
    features['cvar_95'] = returns[returns <= features['var_95']].mean()

    # Volatility of Volatility
    rolling_vol = returns.rolling(24).std()
    features['volatility_of_volatility'] = rolling_vol.std()

    # Tail Ratio
    upper_tail = np.percentile(returns, 95)
    lower_tail = abs(np.percentile(returns, 5))
    features['tail_ratio'] = upper_tail / (lower_tail + 1e-8)

    # Win Rate
    features['win_rate'] = (returns > 0).mean()

    # Profit Factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    features['profit_factor'] = gross_profit / (gross_loss + 1e-8)

    # Sharpe Ratio
    features['sharpe_ratio'] = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252 * 24)

    # Sortino Ratio
    downside = returns[returns < 0].std()
    features['sortino_ratio'] = returns.mean() / (downside + 1e-8) * np.sqrt(252 * 24)

    return features
```

#### Category 7-8: Timing & Behavioral (8 features)

| Feature | Formula | Window | Hypothesis |
|---------|---------|--------|------------|
| `next_day_return` | Return at t+1 | Lead | H14, H23 |
| `next_day_volatility` | Volatility at t+1 | Lead | H14, H23 |
| `holding_period_avg` | Average trade duration | History | H25 |
| `momentum_return` | Return of momentum strategy | 7d | H24 |
| `meanrev_return` | Return of mean-rev strategy | 7d | H24 |
| `fear_greed_proxy` | RSI deviation from 50 | 14d | H13, H26 |
| `extreme_high` | SI > 90th percentile | SI | H26 |
| `extreme_low` | SI < 10th percentile | SI | H26 |

#### Category 9: Factor & Cross-Asset (8 features)

| Feature | Formula | Window | Hypothesis |
|---------|---------|--------|------------|
| `momentum_factor` | Long winners, short losers | 30d | H29 |
| `value_factor` | Price vs MA deviation | 30d | H29 |
| `si_btc` | SI for BTC agents | Current | H30 |
| `si_eth` | SI for ETH agents | Current | H30 |
| `si_cross_asset_corr` | Correlation of SI across assets | 7d | H30 |
| `asset_return_corr` | Cross-asset return correlation | 7d | H30 |
| `rotation_signal` | Relative SI change | 7d | H31 |
| `relative_strength` | Asset return vs benchmark | 7d | H31 |

#### Category 10-11: Dynamics & Meta (9 features)

| Feature | Formula | Window | Hypothesis |
|---------|---------|--------|------------|
| `dsi_dt` | SI change (velocity) | 1d | H16 |
| `si_acceleration` | dSI/dt change | 1d | H16 |
| `si_rolling_std` | Stability of SI | 7d | H34 |
| `si_1h` | SI at 1-hour scale | 1h | H38 |
| `si_4h` | SI at 4-hour scale | 4h | H38 |
| `si_1d` | SI at daily scale | 1d | H38 |
| `si_1w` | SI at weekly scale | 1w | H38 |
| `prediction_accuracy` | Model correctness | 30d | H33 |
| `optimal_leverage` | Kelly criterion | 30d | H35 |

---

## 📊 Phase 3: SI Computation (Day 3-4)

### 3.1 SI Calculator

```python
# src/competition/si_calculator.py

class SICalculator:
    """Compute Specialization Index at various granularities."""

    def compute_si(self, agents: List[Agent], niche_assignments: Dict[int, str]) -> float:
        """
        Compute SI as entropy-based specialization measure.

        SI = 1 - H(niche|agent) / H_max

        Where H(niche|agent) is the entropy of niche assignments
        and H_max = log(num_niches)
        """
        # Count niche assignments per agent
        niche_counts = Counter(niche_assignments.values())
        total = sum(niche_counts.values())

        if total == 0:
            return 0.0

        # Compute entropy
        probs = [count / total for count in niche_counts.values()]
        entropy = -sum(p * np.log(p + 1e-8) for p in probs)

        # Normalize by max entropy
        num_niches = len(set(niche_assignments.values()))
        max_entropy = np.log(num_niches) if num_niches > 1 else 1.0

        # SI = 1 means perfect specialization (low entropy)
        # SI = 0 means no specialization (max entropy)
        si = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

        return max(0.0, min(1.0, si))

    def compute_si_timeseries(self,
                              agents: List[Agent],
                              history: List[Dict],
                              window: int = 24) -> pd.Series:
        """Compute SI over rolling window."""
        si_values = []
        timestamps = []

        for i in range(window, len(history)):
            window_data = history[i-window:i]
            # Get niche assignments from window
            niche_assignments = self._aggregate_niches(window_data)
            si = self.compute_si(agents, niche_assignments)
            si_values.append(si)
            timestamps.append(history[i]['timestamp'])

        return pd.Series(si_values, index=timestamps)
```

### 3.2 SI Variants to Compute

| SI Variant | Description | Purpose |
|------------|-------------|---------|
| `si_instant` | SI at each timestep | Raw signal |
| `si_rolling_1d` | 24h rolling average | Smooth signal |
| `si_rolling_7d` | 7d rolling average | Trend |
| `si_velocity` | dSI/dt | H16: Rate of change |
| `si_acceleration` | d²SI/dt² | H16: Momentum |
| `si_std_7d` | Rolling std of SI | H34: Stability |
| `si_percentile` | Current SI percentile in history | H26: Extremes |
| `si_per_asset` | SI computed per asset | H30, H31 |

---

## 📊 Phase 4: Correlation Analysis (Day 5)

### 4.1 Statistical Methods

```python
# src/analysis/correlation.py

class CorrelationAnalyzer:
    """Comprehensive correlation analysis with multiple methods."""

    def analyze_all(self, si: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Run all correlation analyses."""
        results = []

        for col in features.columns:
            feature = features[col].dropna()
            si_aligned = si.loc[feature.index]

            if len(si_aligned) < 30:
                continue

            result = {
                'feature': col,
                # Pearson (linear)
                'pearson_r': self._pearson(si_aligned, feature),
                'pearson_p': self._pearson_pvalue(si_aligned, feature),

                # Spearman (monotonic)
                'spearman_r': self._spearman(si_aligned, feature),
                'spearman_p': self._spearman_pvalue(si_aligned, feature),

                # Mutual Information (nonlinear)
                'mutual_info': self._mutual_info(si_aligned, feature),

                # Granger Causality (temporal)
                'granger_f': self._granger_causality(si_aligned, feature),
                'granger_p': self._granger_pvalue(si_aligned, feature),

                # Lead-Lag
                'optimal_lag': self._find_optimal_lag(si_aligned, feature),
                'lag_corr': self._lagged_correlation(si_aligned, feature),

                # Effect Size
                'cohens_d': self._cohens_d(si_aligned, feature),

                # Sample size
                'n': len(si_aligned)
            }
            results.append(result)

        return pd.DataFrame(results)

    def _pearson(self, x, y):
        return np.corrcoef(x, y)[0, 1]

    def _spearman(self, x, y):
        from scipy.stats import spearmanr
        return spearmanr(x, y)[0]

    def _mutual_info(self, x, y):
        from sklearn.feature_selection import mutual_info_regression
        return mutual_info_regression(x.values.reshape(-1, 1), y.values)[0]

    def _granger_causality(self, x, y, max_lag=5):
        from statsmodels.tsa.stattools import grangercausalitytests
        data = pd.DataFrame({'x': x, 'y': y}).dropna()
        try:
            result = grangercausalitytests(data[['y', 'x']], maxlag=max_lag, verbose=False)
            # Return F-statistic for optimal lag
            f_stats = [result[i+1][0]['ssr_ftest'][0] for i in range(max_lag)]
            return max(f_stats)
        except:
            return np.nan

    def _find_optimal_lag(self, x, y, max_lag=10):
        """Find lag with strongest correlation."""
        best_lag = 0
        best_corr = 0
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(x.iloc[:lag], y.iloc[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(x.iloc[lag:], y.iloc[:-lag])[0, 1]
            else:
                corr = np.corrcoef(x, y)[0, 1]
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        return best_lag
```

### 4.2 Multiple Testing Correction

```python
def apply_corrections(self, results: pd.DataFrame) -> pd.DataFrame:
    """Apply multiple testing corrections."""
    from statsmodels.stats.multitest import multipletests

    # Bonferroni correction
    results['pearson_p_bonferroni'] = results['pearson_p'] * len(results)
    results['pearson_p_bonferroni'] = results['pearson_p_bonferroni'].clip(upper=1.0)

    # FDR (Benjamini-Hochberg)
    _, results['pearson_p_fdr'], _, _ = multipletests(
        results['pearson_p'], method='fdr_bh'
    )

    # Significance flags
    results['sig_raw'] = results['pearson_p'] < 0.05
    results['sig_bonferroni'] = results['pearson_p_bonferroni'] < 0.05
    results['sig_fdr'] = results['pearson_p_fdr'] < 0.05

    return results
```

### 4.3 Output Format

```python
# Results structure
{
    'summary': {
        'total_features': 70,
        'significant_raw': 25,
        'significant_bonferroni': 8,
        'significant_fdr': 12,
        'top_10_features': [...],
        'interpretation': "..."
    },
    'correlations': pd.DataFrame,  # All correlations
    'regime_stratified': {
        'trending': pd.DataFrame,
        'ranging': pd.DataFrame,
        'volatile': pd.DataFrame
    },
    'temporal_analysis': {
        'granger_results': pd.DataFrame,
        'lead_lag_matrix': pd.DataFrame
    },
    'visualizations': {
        'heatmap_path': 'results/si_correlations/heatmap.png',
        'scatter_plots_path': 'results/si_correlations/scatters/',
        'time_series_path': 'results/si_correlations/timeseries.png'
    }
}
```

---

## 📊 Phase 5: Interpretation & Visualization (Day 6)

### 5.1 Auto-Interpretation

```python
# src/analysis/interpretation.py

class Interpreter:
    """Automatically interpret correlation results."""

    def interpret(self, results: pd.DataFrame) -> Dict:
        """Generate human-readable interpretation."""

        # Sort by absolute correlation
        results_sorted = results.sort_values('pearson_r', key=abs, ascending=False)

        # Top positive correlations
        top_positive = results_sorted[results_sorted['pearson_r'] > 0].head(5)

        # Top negative correlations
        top_negative = results_sorted[results_sorted['pearson_r'] < 0].head(5)

        # Significant after correction
        significant = results_sorted[results_sorted['sig_fdr']]

        interpretation = {
            'main_finding': self._generate_main_finding(significant),
            'top_positive': self._describe_correlations(top_positive, 'positive'),
            'top_negative': self._describe_correlations(top_negative, 'negative'),
            'hypothesis_support': self._map_to_hypotheses(significant),
            'recommended_path': self._recommend_path(significant),
            'caveats': self._generate_caveats(results)
        }

        return interpretation

    def _generate_main_finding(self, significant: pd.DataFrame) -> str:
        if len(significant) == 0:
            return "No statistically significant correlations found after FDR correction."

        top = significant.iloc[0]
        direction = "positively" if top['pearson_r'] > 0 else "negatively"

        return f"SI is most strongly {direction} correlated with {top['feature']} (r={top['pearson_r']:.3f}, p={top['pearson_p_fdr']:.4f})"

    def _map_to_hypotheses(self, significant: pd.DataFrame) -> Dict[str, str]:
        """Map significant correlations to our 40 hypotheses."""
        hypothesis_map = {
            'volatility': 'H1, H7',
            'return_entropy': 'H2',
            'regime_duration': 'H3',
            'agent_correlation': 'H4',
            'max_drawdown': 'H21, H22',
            'win_rate': 'H19',
            'next_day_return': 'H14, H23',
            'dsi_dt': 'H16',
            # ... more mappings
        }

        supported = {}
        for _, row in significant.iterrows():
            feature = row['feature']
            for key, hyp in hypothesis_map.items():
                if key in feature.lower():
                    supported[feature] = hyp

        return supported
```

### 5.2 Visualization Suite

```python
# src/analysis/visualization.py

class Visualizer:
    """Generate all visualizations for SI correlation analysis."""

    def generate_all(self, si: pd.Series, features: pd.DataFrame,
                     results: pd.DataFrame, output_dir: str):
        """Generate complete visualization suite."""

        # 1. Correlation Heatmap
        self._plot_heatmap(results, f"{output_dir}/heatmap.png")

        # 2. Top 10 Scatter Plots
        self._plot_top_scatters(si, features, results, f"{output_dir}/scatters/")

        # 3. SI Time Series with Annotations
        self._plot_si_timeseries(si, features, f"{output_dir}/si_timeseries.png")

        # 4. Lead-Lag Heatmap
        self._plot_lead_lag(results, f"{output_dir}/lead_lag.png")

        # 5. Regime-Stratified Correlations
        self._plot_regime_comparison(results, f"{output_dir}/regime_comparison.png")

        # 6. Effect Size Forest Plot
        self._plot_forest(results, f"{output_dir}/effect_sizes.png")

    def _plot_heatmap(self, results: pd.DataFrame, path: str):
        """Correlation heatmap sorted by absolute correlation."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Pivot for heatmap
        sorted_results = results.sort_values('pearson_r', key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(12, 20))

        # Create bar chart of correlations
        colors = ['green' if r > 0 else 'red' for r in sorted_results['pearson_r']]
        bars = ax.barh(range(len(sorted_results)), sorted_results['pearson_r'], color=colors)

        # Add significance markers
        for i, (_, row) in enumerate(sorted_results.iterrows()):
            if row['sig_fdr']:
                ax.annotate('***', (row['pearson_r'], i), fontsize=10)
            elif row['sig_raw']:
                ax.annotate('*', (row['pearson_r'], i), fontsize=10)

        ax.set_yticks(range(len(sorted_results)))
        ax.set_yticklabels(sorted_results['feature'])
        ax.set_xlabel('Pearson Correlation with SI')
        ax.set_title('SI Correlation with All Features\n(*** = FDR significant, * = raw significant)')
        ax.axvline(x=0, color='black', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
```

---

## 📊 Phase 6: Decision & Next Steps (Day 7)

### 6.1 Decision Framework

```python
def make_decision(interpretation: Dict) -> str:
    """Determine next steps based on findings."""

    significant = interpretation['hypothesis_support']

    # Check for direct profit path
    if any('profit' in f.lower() or 'return' in f.lower() for f in significant):
        return "DIRECT_PROFIT_PATH"

    # Check for risk management path
    if any('drawdown' in f.lower() or 'var' in f.lower() or 'risk' in f.lower() for f in significant):
        return "RISK_MANAGEMENT_PATH"

    # Check for diversification path
    if any('correlation' in f.lower() or 'effective_n' in f.lower() for f in significant):
        return "DIVERSIFICATION_PATH"

    # Check for timing path
    if any('next_day' in f.lower() or 'leading' in f.lower() for f in significant):
        return "TIMING_SIGNAL_PATH"

    # Check for regime path
    if any('regime' in f.lower() or 'trend' in f.lower() for f in significant):
        return "REGIME_DETECTION_PATH"

    # No clear path
    return "DEEPER_ANALYSIS_NEEDED"
```

### 6.2 Output Report Template

```markdown
# SI Correlation Discovery Report

## Executive Summary
[Auto-generated main finding]

## Top 10 Correlations

| Rank | Feature | Correlation | P-value (FDR) | Hypothesis Supported |
|------|---------|-------------|---------------|----------------------|
| 1    | ...     | ...         | ...           | ...                  |

## Interpretation

### What SI Measures
Based on the strongest correlations, SI appears to measure: [...]

### Path to Profit
[DIRECT / VIA RISK / VIA DIVERSIFICATION / VIA TIMING / ...]

### Recommended Next Steps
1. [...]
2. [...]
3. [...]

## Caveats
- Multiple testing: X/70 features significant after FDR correction
- Temporal stability: [...]
- Regime dependence: [...]

## Appendix
- Full correlation table
- All visualizations
- Code and reproducibility
```

---

## 📅 Timeline Summary

| Day | Phase | Tasks | Output |
|-----|-------|-------|--------|
| 1 | Setup | Extract data, build infrastructure | `src/` scaffolding |
| 2 | Setup | Implement feature calculators | `FeatureCalculator` class |
| 3 | Compute | Run backtest, compute SI | SI time series |
| 4 | Compute | Compute all 70+ features | Features DataFrame |
| 5 | Analyze | Correlation analysis | Results DataFrame |
| 6 | Interpret | Visualization + interpretation | Report + plots |
| 7 | Decide | Decision + next steps | Path forward |

---

## 🔧 Implementation Checklist

### Day 1-2: Infrastructure
- [ ] Create directory structure
- [ ] Implement `DataLoader`
- [ ] Implement `FeatureCalculator` (15 market state features)
- [ ] Implement `FeatureCalculator` (10 agent features)
- [ ] Implement `FeatureCalculator` (10 risk features)
- [ ] Implement `FeatureCalculator` (8 timing features)
- [ ] Implement `FeatureCalculator` (8 factor features)
- [ ] Implement `FeatureCalculator` (9 dynamics features)

### Day 3-4: Computation
- [ ] Implement 3 strategies (momentum, mean-reversion, breakout)
- [ ] Implement `NichePopulation` algorithm
- [ ] Implement `SICalculator`
- [ ] Run full backtest
- [ ] Compute SI variants (8 types)
- [ ] Compute all 70+ features

### Day 5: Analysis
- [ ] Implement `CorrelationAnalyzer`
- [ ] Pearson correlations
- [ ] Spearman correlations
- [ ] Mutual information
- [ ] Granger causality
- [ ] Lead-lag analysis
- [ ] Multiple testing correction

### Day 6: Visualization
- [ ] Correlation heatmap
- [ ] Top 10 scatter plots
- [ ] SI time series with annotations
- [ ] Lead-lag heatmap
- [ ] Regime-stratified comparison
- [ ] Effect size forest plot

### Day 7: Decision
- [ ] Auto-interpret results
- [ ] Map to hypotheses
- [ ] Generate report
- [ ] Determine path forward
- [ ] Plan Phase 1

---

*This plan ensures systematic discovery of what SI actually measures.*

*Last Updated: January 17, 2026*
