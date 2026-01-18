# Emergent Trading Specialists

> 🎯 **SI Signal Discovery: Cross-Market Validation of Specialization Index as a Trading Signal**

[![Status](https://img.shields.io/badge/Status-Complete-success)](https://github.com/HowardLiYH/Emergent-Applications/tree/main/apps/trading)
[![Implementation](https://img.shields.io/badge/Expert%20Suggestions-23%2F23-brightgreen)](results/implementation_check/status.json)
[![Paper](https://img.shields.io/badge/Paper-LaTeX-blue)](paper/SI_Signal_Discovery_Report.tex)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎉 Results Summary

**ALL 23 EXPERT PANEL SUGGESTIONS IMPLEMENTED** ✅

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Features with \|r\| > 0.15 | ≥3 | **17** | ✅ PASS |
| VAL confirmation rate | >30% | **51%** | ✅ PASS |
| TEST confirmation rate | >30% | **44%** | ✅ PASS |
| Assets validated | ≥3 | **11** | ✅ PASS |
| Markets validated | ≥2 | **4** | ✅ PASS |
| Factor timing success | - | **91%** | ✅ NEW |
| t-SNE clustering | - | **100%** | ✅ NEW |
| OOS R² significant | - | **100%** | ✅ NEW |

## Overview

This project investigates whether the **Specialization Index (SI)**—a metric measuring emergent agent specialization in competitive environments—correlates with meaningful market features and can serve as a trading signal.

**Key Findings:**
- SI correlates with **17 features** across 4 market types
- Top correlates: ADX, Bollinger Band Width, RSI, Volatility
- SI captures **"market readability"**—trending, moderate-volatility conditions
- **Factor timing**: SI helps timing in 91% of assets
- **Regret bound**: O(√T) theorem established for agent learning

## Expert Panel Implementation (4 Rounds)

All 23 suggestions from 4 rounds of expert/professor review have been implemented:

### Round 1 - Foundation (7/7 ✅)
| Item | Description | Status |
|------|-------------|--------|
| R1_A1 | Random agent baseline | ✅ Done |
| R1_B1 | Permutation tests | ✅ Done |
| R1_B5 | FDR justification | ✅ Done |
| R1_C1 | Synthetic data validation | ✅ Done |
| R1_C2 | Alternative SI (Gini, HHI) | ✅ Done |
| R1_H2 | Toy example | ✅ Done |
| R1_H4 | Contribution statement | ✅ Done |

### Round 2 - Robustness (5/5 ✅)
| Item | Description | Status |
|------|-------------|--------|
| R2_1 | Subsample stability | ✅ Done |
| R2_2 | SI persistence/ACF | ✅ Done |
| R2_3 | Convergence analysis | ✅ Done |
| R2_4 | Half-life of SI changes | ✅ Done |
| R2_5 | Falsification criteria | ✅ Done |

### Round 3/4 - High Priority (6/6 ✅)
| Item | Description | Result |
|------|-------------|--------|
| R3_1 | Crisis case study | 2022 Luna/FTX early warning |
| R3_2 | Factor timing | **91% success rate** |
| R3_3 | t-SNE visualization | **100% show clustering** |
| R3_4 | OOS R² with CI | **100% significant** |
| R4_REGRET | Regret bound theorem | **O(√T) proven** |
| R4_LITERATURE | Literature positioning | 7 methods compared |

### Audit & Fixes (5/5 ✅)
- Block bootstrap ✅
- Stationarity tests (ADF/KPSS) ✅
- Parameter sensitivity ✅
- SI risk indicator ✅
- Full NichePopulation SI ✅

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke test
python experiments/smoke_test.py

# Run full analysis
python experiments/run_corrected_analysis.py

# Run 9+ strategy (factor timing, t-SNE, etc.)
python experiments/implement_9plus_strategy.py

# Check implementation status
python experiments/check_all_implementations.py

# Generate figures
python experiments/generate_figures.py
```

## Data Coverage

| Market | Assets | Period | Frequency | Source |
|--------|--------|--------|-----------|--------|
| **Crypto** | BTC, ETH, SOL | 5 years | Daily | Binance |
| **Forex** | EUR/USD, GBP/USD, USD/JPY | 5 years | Daily | Yahoo Finance |
| **Stocks** | SPY, QQQ, AAPL | 5 years | Daily | Yahoo Finance |
| **Commodities** | Gold, Oil | 5 years | Daily | Yahoo Finance |

## Key Results

### Top SI Correlates (All 4 Markets)

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| **ADX** (trend strength) | +0.15 to +0.23 | SI ↑ when trends are clear |
| **Bollinger Band Width** | +0.22 to +0.29 | SI ↑ when volatility is structured |
| **RSI** | +0.20 to +0.30 | SI ↑ when momentum is defined |
| **Volatility** | -0.15 to -0.23 | SI ↓ during extreme volatility |

### Factor Timing Results

| Asset | Improvement | Status |
|-------|-------------|--------|
| ETH | +0.057 | ✅ SI helps |
| OIL | +0.033 | ✅ SI helps |
| AAPL | +0.017 | ✅ SI helps |
| SPY | +0.011 | ✅ SI helps |
| **Overall** | **10/11 (91%)** | ✅ PASS |

### Cross-Market Confirmation Rates

| Market | VAL Rate | TEST Rate |
|--------|----------|-----------|
| Crypto | 37.9% | 24.8% |
| Forex | 55.8% | 52.5% |
| Stocks | 54.0% | 50.0% |
| Commodities | 59.5% | 52.4% |
| **Overall** | **51.1%** | **44.2%** |

### Regime Detection Comparison

| Method | Sign Flip Rate | Recommendation |
|--------|----------------|----------------|
| Rule-based | **5.0%** | ✅ Best for SI |
| GMM | 10.4% | Good alternative |
| HMM | 17.5% | Too smooth |

## Theoretical Contributions

### Regret Bound Theorem

> **THEOREM**: Under the affinity update rule, each agent achieves expected cumulative regret bounded by O(√T log K) where T is rounds and K is number of regimes.

**Implications:**
- Agents learn to specialize optimally over time
- SI emergence is a consequence of no-regret dynamics
- Connects to Multiplicative Weights Update (Arora et al., 2012)

### Literature Positioning

| Method | Interpretable | Emergent | Our Advantage |
|--------|---------------|----------|---------------|
| HMM | Partial | No | SI emerges from dynamics |
| LSTM | No | No | SI is interpretable |
| ABM | Partial | Yes | SI quantifies emergence |
| Factor Models | Yes | No | SI captures agent dynamics |
| GARCH | Yes | No | SI measures specialization |

## Project Structure

```
apps/trading/
├── src/                          # Core Python modules
│   ├── agents/                   # Trading strategies
│   ├── competition/              # NichePopulation algorithm
│   ├── analysis/                 # Feature & correlation analysis
│   ├── data/                     # Data loading & validation
│   ├── backtest/                 # SI-based trading strategy
│   └── utils/                    # Utilities
├── experiments/                  # Runnable scripts
│   ├── implement_9plus_strategy.py  # Factor timing, t-SNE, OOS R²
│   ├── implement_remaining.py       # Regret bound, literature
│   ├── check_all_implementations.py # Verify 23/23 complete
│   └── ... (30+ experiment scripts)
├── paper/                        # LaTeX report
│   ├── SI_Signal_Discovery_Report.tex
│   └── figures/                  # Generated figures
│       └── 9plus/                # t-SNE visualizations
├── results/                      # Analysis outputs
│   ├── 9plus_strategy/           # Factor timing results
│   ├── remaining_items/          # Regret bound & literature
│   ├── implementation_check/     # 23/23 verification
│   └── ... (15+ result directories)
├── data/                         # Market data (5 years)
├── docs/                         # Documentation
├── MASTER_PLAN.md                # Execution plan (complete)
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Methodology

### Pre-Registration
All hypotheses were pre-registered before analysis to prevent p-hacking:
- Report all results including null findings
- No post-hoc hypothesis changes
- Benjamini-Hochberg FDR correction at α = 0.05
- Validate on holdout sets before claiming significance

### Statistical Rigor
- **Spearman correlation** for non-linear relationships
- **FDR correction** for 286 tests (26 features × 11 assets)
- **Block bootstrap** (1,000 iterations) for confidence intervals
- **Permutation tests** for significance validation
- **Stationarity tests** (ADF/KPSS) for time series validity
- **Effect size threshold**: |r| > 0.10 meaningful, |r| > 0.15 strong

### Data Splits
- **Train**: 70% (discovery)
- **Validation**: 15% (confirmation)
- **Test**: 15% (holdout)
- **Purging gap**: 7 days between splits

## Specialization Index (SI)

```
SI = 1 - mean(normalized_entropy of niche_affinities)

High SI (→1): Agents have distinct niches (specialists)
Low SI (→0): Agents are similar (generalists)
```

SI emerges from agent competition:
1. 18 agents (6 strategies × 3 instances) compete daily
2. Winner updates niche affinity for current regime
3. Over time, agents specialize in different regimes
4. SI measures degree of specialization

## Related Papers

| Paper | Focus | Key Finding |
|-------|-------|-------------|
| [NichePopulation](https://arxiv.org/abs/...) | Time-series learners | SI = 0.747, Cohen's d > 20 |
| [Preference Specialization](https://arxiv.org/abs/...) | LLM agents | Competition = 94% of specialization |
| **This Work** | Trading signals | SI correlates with market readability |

## Citation

```bibtex
@techreport{li2026si,
  title={SI Signal Discovery: Cross-Market Validation of Specialization Index as a Trading Signal},
  author={Li, Yuhao},
  institution={University of Pennsylvania},
  year={2026},
  url={https://github.com/HowardLiYH/Emergent-Applications/tree/main/apps/trading}
}
```

## Author

**Yuhao Li**
University of Pennsylvania
📧 li88@sas.upenn.edu

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*23/23 expert suggestions implemented. Factor timing 91%. t-SNE 100%. O(√T) regret bound proven. 4 markets validated.*
