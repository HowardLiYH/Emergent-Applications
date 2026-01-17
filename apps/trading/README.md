# Emergent Trading Specialists

> 🎯 **Discovering what Specialization Index (SI) measures in financial trading**

## Overview

This application extends the NichePopulation mechanism (Paper 1) to financial trading. Instead of assuming SI leads to profit, we take a **discovery-first approach**: systematically correlating SI with 70+ market features to understand what SI actually measures, then tracing the path to profit.

## Core Research Question

> **"What does Specialization Index (SI) actually correlate with in trading?"**

We don't assume SI → Profit. We discover:
1. What SI correlates with most strongly
2. Whether that leads to profit (directly or indirectly)
3. Alternative value paths (risk reduction, timing, diversification)

## Key Insight: Discovery Before Assumption

```
OLD THINKING: "SI should lead to profit" → Test if true
NEW THINKING: "What does SI measure?" → Find correlations → Trace to profit
```

## 40 Hypotheses Being Tested

| Category | Hypotheses | Example |
|----------|------------|---------|
| **Market State** | H1-H3 | SI = Regime Clarity, Inverse Entropy |
| **Agent Behavior** | H4-H6 | SI = Diversification, Niche Stability |
| **Microstructure** | H7-H8 | SI = Liquidity, Information Flow |
| **Risk** | H19-H22 | SI → Lower Drawdown, Tail Protection |
| **Timing** | H23-H25 | SI → Trading Windows, Time-to-Profit |
| **Behavioral** | H26-H28 | SI → Contrarian Extremes |
| **Factor** | H29-H31 | SI → Cross-Asset Signals |
| **Moonshot** | H36-H40 | SI → Black Swan Detection |

See [docs/SI_EXPLORATION.md](docs/SI_EXPLORATION.md) for all 40 hypotheses.

## Fallback Paths to Value

Even if SI ≠ Profit directly:

| Path | What SI Correlates With | Value |
|------|------------------------|-------|
| **Risk Manager** | Drawdown, VaR | Lower risk, better Sharpe |
| **Diversifier** | Agent correlation | Uncorrelated returns |
| **Regime Detector** | Regime stability | Know when to trade |
| **Timing Signal** | Next-day returns | Position sizing |
| **Anti-Crowding** | Strategy overlap | Avoid crowded trades |

## Quick Start

```bash
# Phase 0: Discovery
python experiments/phase0_si_discovery.py

# Output: Top 10 SI correlations
# Decision: Which path to pursue
```

## 8-Day Test Plan (Three Pipelines)

| Day | Phase | Pipeline | Output |
|-----|-------|----------|--------|
| 0 | Pre-Reg | - | `pre_registration.json` |
| 1-2 | Infrastructure | - | Data + categorized features |
| 3-4 | Computation | - | SI + all features |
| 5 | **Pipeline 1** | Discovery | "SI correlates with X" |
| 6 | **Pipeline 2** | Prediction | "SI predicts Y with lag K" |
| 7 | **Pipeline 3** | SI Dynamics | "Best SI variant is Z" |
| 8 | Report | All | Final conclusions |

See [docs/SI_CORRELATION_TEST_PLAN.md](docs/SI_CORRELATION_TEST_PLAN.md) for full plan.

## Three Separate Pipelines

**Key insight**: Not all features can use the same pipeline!

| Pipeline | Features | Question |
|----------|----------|----------|
| **Discovery** | 46 | What does SI correlate with? |
| **Prediction** | 2 | Does SI predict future outcomes? |
| **SI Dynamics** | 9 | How should we use SI? |

```
⚠️ Circular features (removed from discovery):
- SI-derived: dSI/dt, si_std, si_1h, si_4h...
- Agent features that ARE SI: strategy_concentration, niche_entropy
```

## 70+ Features Being Tested

```python
# Market State (15)
volatility, trend_strength, return_entropy, hurst_exponent...

# Agent Behavior (10)
agent_correlation, winner_spread, effective_n, niche_stability...

# Risk Metrics (10)
max_drawdown, VaR, win_rate, profit_factor, Sharpe...

# Timing (8)
next_day_return, momentum_return, meanrev_return...

# Factor (8)
momentum_factor, cross_asset_corr, rotation_signal...

# Dynamics (9)
dSI/dt, si_stability, si_acceleration...
```

## Folder Structure

```
apps/trading/
├── docs/
│   ├── STRATEGY_PLANNING.md      # Overall strategy
│   ├── SI_EXPLORATION.md         # 40 hypotheses
│   ├── SI_CORRELATION_TEST_PLAN.md # 70+ features test plan
│   ├── SI_TO_PROFIT_INNOVATION.md  # Core thesis
│   ├── TRADING_RESEARCH_PLAN.md    # Expert panel input
│   └── ARCHITECTURE_DECISIONS.md   # Technical choices
├── experiments/
│   └── phase0_si_discovery.py    # Main discovery experiment
├── src/
│   ├── data/                     # Data loading
│   ├── agents/                   # Trading agents
│   ├── competition/              # NichePopulation + SI
│   ├── analysis/                 # Correlation analysis
│   └── backtest/                 # Backtest runner
├── results/
│   └── si_correlations/          # Discovery results
└── data/
    └── bybit/                    # Price data
```

## Decision Framework

```
After correlation analysis:

SI correlates with PROFIT?        → Direct thesis validated
SI correlates with RISK METRICS?  → Pivot: "SI for risk management"
SI correlates with DIVERSIFICATION? → Pivot: "SI for portfolio"
SI correlates with TIMING?        → Pivot: "SI as leading indicator"
SI correlates with NOTHING?       → Deeper analysis or abandon
```

## Success Metrics

| Metric | Target | Priority |
|--------|--------|----------|
| Significant SI correlations | ≥5 after FDR | 1 (Discovery) |
| Path to profit identified | Yes | 2 |
| Net Return | > 0% | 3 (Validation) |
| Sharpe Ratio | > 0.5 | 4 |

## Status

- [x] Research plan documented
- [x] 40 hypotheses defined
- [x] 70+ features identified
- [x] Test plan created
- [x] **Methodology audit: 16 issues fixed**
- [x] **Expert panel review: 18 recommendations incorporated**
- [x] **Feature pipeline audit: 3 pipelines designed**
- [x] **8 additional audits: All designed**
- [x] **Pre-registration file created**
- [ ] **NEXT: Commit pre-registration to GitHub**
- [ ] Build infrastructure
- [ ] Run discovery experiment
- [ ] Interpret results
- [ ] Choose path forward

## Methodology Rigor

**Total improvements: 48**

| Audit Phase | Issues | Status |
|-------------|--------|--------|
| Initial Methodology Audit | 16 | ✅ Fixed |
| Expert Panel Review | 18 | ✅ Incorporated |
| Feature Pipeline Audit | 6 | ✅ 3 pipelines created |
| **8 Additional Expert Audits** | 8 | ✅ All designed |

### 8 Expert-Recommended Audits

| # | Audit | Priority | Status |
|---|-------|----------|--------|
| 1 | Pre-Registration | 🔴 | ✅ `experiments/pre_registration.json` |
| 2 | Implementation Tests | 🔴 | ✅ Unit tests designed |
| 3 | Causal Inference | 🔴 | ✅ Granger, placebo, permutation |
| 4 | Strategy Validity | 🔴 | ✅ Benchmarks, costs, parameters |
| 5 | Reproducibility | 🟡 | ✅ Manifest, seeds, versions |
| 6 | Crypto-Specific | 🟡 | ✅ Time-of-day, weekend, liquidity |
| 7 | Multi-Asset | 🟡 | ✅ 5 assets, time periods, regimes |
| 8 | Adversarial | 🟡 | ✅ Devil's advocate, permutation |

### Key Safeguards

| Safeguard | Implementation |
|-----------|----------------|
| Train/Val/Test split | 70/15/15 temporal |
| Autocorrelation | HAC std errors + block bootstrap |
| Multiple testing | Pre-registration + FDR |
| Circular reasoning | Predictive (lagged) correlations |
| Rolling validation | 5-fold cross-validation |
| Negative controls | Random noise, shuffled SI |
| Liquidity control | Amihud, volume as confounders |
| Signal decay | Half-life estimation |
| Transaction costs | Sensitivity analysis |

## Expert Panel

14 experts consulted across domains:
- 🧠 Information Theorist
- 🌀 Complexity Scientist
- 🏦 Quant Strategist
- 📊 Risk Manager
- 💼 Portfolio Manager
- 🔮 Regime Detection Specialist
- ... and 8 more

## Related Papers

- [Paper 1: NichePopulation](https://github.com/HowardLiYH/NichePopulation) - Foundational specialization
- [Paper 2: Emergent Preference Specialization](https://github.com/HowardLiYH/Emergent-Preference-Specialization-in-LLM-Agent-Populations) - LLM preferences
- [Paper 3: Emergent Tool Specialization](https://github.com/HowardLiYH/Emergent-Tool-Specialization) - Real tools

---

*Discovery-first. 40 hypotheses. 70+ features. Let's find out what SI really measures.*
