# Emergent Trading Specialists

> 🎯 **SI Signal Discovery: Cross-Market Validation of Specialization Index as a Trading Signal**

[![Status](https://img.shields.io/badge/Status-Complete-success)](https://github.com/HowardLiYH/Emergent-Applications/tree/main/apps/trading)
[![Paper](https://img.shields.io/badge/Paper-LaTeX-blue)](paper/SI_Signal_Discovery_Report.tex)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎉 Results Summary

**PRIMARY HYPOTHESIS SUPPORTED** ✅

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Features with \|r\| > 0.15 | ≥3 | **17** | ✅ PASS |
| VAL confirmation rate | >30% | **51%** | ✅ PASS |
| TEST confirmation rate | >30% | **44%** | ✅ PASS |
| Assets validated | ≥3 | **11** | ✅ PASS |
| Markets validated | ≥2 | **4** | ✅ PASS |

## Overview

This project investigates whether the **Specialization Index (SI)**—a metric measuring emergent agent specialization in competitive environments—correlates with meaningful market features and can serve as a trading signal.

**Key Findings:**
- SI correlates with **17 features** across 4 market types
- Top correlates: ADX, Bollinger Band Width, RSI, Volatility
- SI captures **"market readability"**—trending, moderate-volatility conditions
- **Rule-based regime detection** outperforms HMM/GMM for SI analysis

## Core Research Question

> **"What does Specialization Index (SI) correlate with in financial trading?"**

This is a **discovery-first approach**: rather than assuming SI predicts returns, we systematically test correlations with market features, then trace significant correlations to practical implications.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke test
python experiments/smoke_test.py

# Run full analysis
python experiments/run_corrected_analysis.py

# Generate figures
python experiments/generate_figures.py

# Generate report
python experiments/generate_report.py
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

## Project Structure

```
apps/trading/
├── src/                          # Core Python modules
│   ├── agents/                   # Trading strategies
│   │   ├── strategies.py         # Base strategies
│   │   └── strategies_v2.py      # Frequency-aware strategies
│   ├── competition/              # NichePopulation algorithm
│   │   ├── niche_population.py   # Core SI computation
│   │   └── niche_population_v2.py # Frequency-aware version
│   ├── analysis/                 # Feature & correlation analysis
│   │   ├── features.py           # Feature calculator
│   │   ├── features_v2.py        # Frequency-aware features
│   │   ├── correlations.py       # Statistical analysis
│   │   └── regime_detection.py   # Rule/HMM/GMM detectors
│   ├── data/                     # Data loading & validation
│   │   ├── loader.py             # Multi-market loader
│   │   ├── loader_v2.py          # With purging/embargo
│   │   └── validation.py         # Data quality checks
│   ├── backtest/                 # SI-based trading strategy
│   │   └── si_strategy.py
│   └── utils/                    # Utilities
│       ├── logging_setup.py
│       ├── safe_math.py
│       ├── timezone.py
│       ├── reproducibility.py
│       ├── checkpointing.py
│       └── caching.py
├── experiments/                  # Runnable scripts
│   ├── pre_registration.json     # Pre-registered hypotheses
│   ├── smoke_test.py             # Minimal validation
│   ├── run_corrected_analysis.py # Main analysis (frequency-aware)
│   ├── run_discovery.py          # Discovery pipeline
│   ├── run_prediction.py         # Prediction pipeline
│   ├── run_dynamics.py           # SI dynamics pipeline
│   ├── run_validation.py         # Holdout validation
│   ├── run_regime_analysis.py    # Regime-conditioned analysis
│   ├── compare_regime_methods.py # Rule vs HMM vs GMM
│   ├── generate_figures.py       # Publication figures
│   └── generate_report.py        # Final report
├── paper/                        # LaTeX report
│   ├── SI_Signal_Discovery_Report.tex
│   └── figures/                  # Generated figures (PNG + PDF)
├── results/                      # Analysis outputs
│   ├── corrected_analysis/       # Main results
│   ├── regime_analysis/          # Regime-conditioned results
│   ├── regime_comparison/        # Method comparison
│   └── si_correlations/          # Discovery results
├── data/                         # Market data (5 years)
│   ├── crypto/                   # BTC, ETH, SOL
│   ├── forex/                    # EUR/USD, GBP/USD, USD/JPY
│   ├── stocks/                   # SPY, QQQ, AAPL
│   └── commodities/              # Gold, Oil
├── docs/                         # Documentation
│   ├── UNDERSTANDING_CHECK.md    # Project overview
│   ├── SIGNAL_PROCESSING_ISSUES.md # Frequency-aware fixes
│   └── ... (additional docs)
├── MASTER_PLAN.md                # Execution plan (all phases complete)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Methodology

### Pre-Registration
All hypotheses were pre-registered before analysis to prevent p-hacking:
- Report all results including null findings
- No post-hoc hypothesis changes  
- Benjamini-Hochberg FDR correction at α = 0.05
- Validate on holdout sets before claiming significance

### Statistical Analysis
- **Spearman correlation** for non-linear relationships
- **FDR correction** for 286 tests (26 features × 11 assets)
- **Block bootstrap** (1,000 iterations) for confidence intervals
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

*Discovery-first approach. 17 significant features. 4 market types. 44% holdout confirmation.*
