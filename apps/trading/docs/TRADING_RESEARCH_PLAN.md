# Trading Application: Research Plan & Expert Analysis

**Date**: January 17, 2026
**Status**: Research & Planning Phase
**Based On**: NichePopulation (Paper 1) mechanism extension

---

## Executive Summary

This document captures the research findings and expert analysis for building a trading application based on the NichePopulation specialization mechanism. The key insight is that **profit must be validated first**, before testing whether specialization (SI) adds value.

---

## 🎯 Core Thesis

> **Can competitive specialization among trading agents produce better risk-adjusted returns than individual strategies or naive ensembles?**

### The Causal Chain We Must Validate

```
Individual Strategy Profitability
         ↓ (must prove first)
Strategy Diversity (low correlation)
         ↓ (must prove second)
Ensemble Improvement (Sharpe > individual)
         ↓ (must prove third)
NichePopulation Mechanism Adds Value
         ↓ (final validation)
SI Emerges AND Leads to Profit
```

**Critical Insight**: Each step must be validated before proceeding. If individual strategies aren't profitable, no amount of specialization will help.

---

## 🔬 Key Research Findings

### Finding 1: SI Does NOT Guarantee Profit

| Question | Answer |
|----------|--------|
| Does high SI guarantee profit? | **NO** — SI measures diversity, not profitability |
| Can high SI + unprofitable strategies = profit? | **NO** — combining unprofitable strategies doesn't create profit |
| Can high SI + profitable strategies = better profit? | **YES** — this is the only valid path |
| What should we test first? | **Individual strategy profitability**, then SI |

### Finding 2: Thompson Sampling Status

Thompson Sampling is NOT outdated, but has limitations for trading:

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| Non-stationary rewards | Markets change over time | Use sliding windows, decay old observations |
| Delayed feedback | P&L takes time to realize | Use proxy rewards, multi-step returns |
| High-dimensional state | Market state is complex | Use feature extraction first |

**Recommendation**: Start with Thompson Sampling, upgrade later if validated.

### Finding 3: Modern Alternatives to Thompson Sampling

| Architecture | Description | Pros | Cons | When to Use |
|--------------|-------------|------|------|-------------|
| **Thompson Sampling** | Bayesian bandit with posterior sampling | Simple, theoretically grounded | Assumes stationary, simple rewards | Phase 0-1 (baseline) |
| **Neural Contextual Bandits** | Neural network estimates rewards per context | Handles complex state | Needs more data | Phase 2+ if TS works |
| **Soft Actor-Critic (SAC)** | Continuous RL with entropy regularization | Natural exploration | Complex, data-hungry | Production upgrade |
| **Decision Transformer** | Sequence prediction for RL | Leverages offline data | Computationally expensive | Advanced research |
| **Mixture of Experts (MoE)** | Gating network routes to specialists | Natural specialization | Need to train gating | Production upgrade |
| **Population-Based Training** | Evolves hyperparameters during training | Adapts to change | Expensive | Advanced research |

### Finding 4: Regimes Are Correlated

Real market regimes are NOT independent:

- Transitions follow patterns (P(crisis | high_vol) >> P(crisis | low_vol))
- Correlations spike during stress (all assets move together)
- Volatility and correlation regimes are linked

**Recommendation**: Use soft regime assignments (probabilities), not hard labels.

### Finding 5: What Hedge Funds Actually Use

| Approach | Used By | Description |
|----------|---------|-------------|
| Bayesian Model Averaging | AQR, Two Sigma | Weight strategies by posterior probability |
| Online Convex Optimization | Renaissance, Jump | Adaptive weights with regret guarantees |
| Mixture of Experts | Citadel | Neural gating selects strategies |
| Hierarchical RL | DE Shaw | High-level policy selects strategies |
| Regime-Switching Models | Most macro funds | DCC-GARCH, Markov-Switching |

---

## 📊 Success Metrics

### Primary Metrics (Profit-Focused)

| Metric | Definition | Target |
|--------|------------|--------|
| **Net Return** | Total profit after costs | > 0% |
| **Sharpe Ratio** | Risk-adjusted return | > 0.5 (baseline), > 1.0 (good) |
| **Max Drawdown** | Worst peak-to-trough loss | < 25% |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 |

### Secondary Metrics (Diagnostic)

| Metric | Definition | Target |
|--------|------------|--------|
| **Specialization Index (SI)** | Entropy-based diversity measure | > 0.4 |
| **Strategy Correlation** | Pairwise return correlation | < 0.3 |
| **Regime Coverage** | Fraction of time with confident regime | > 70% |

### Order of Evaluation

1. **Profit** → Must be positive
2. **Drawdown** → Must be acceptable
3. **Sharpe** → Must beat baseline
4. **SI** → Nice to have, not required

---

## 🚀 Phased Implementation Plan

### Phase 0a: Individual Strategy Profitability (Week 1)

**Objective**: Prove base strategies work independently

| Strategy | Description | Data | Success Criteria |
|----------|-------------|------|------------------|
| Momentum | Buy if 20d return > 0 | BTC, ETH, SPY | Sharpe > 0.3 |
| Mean Reversion | Buy if price < 20d MA by 2% | BTC, ETH, SPY | Sharpe > 0.3 |
| Volatility | Reduce position if vol > 2x average | BTC, ETH, SPY | Lower drawdown |

**Cost**: $0 (free data from Yahoo Finance)
**Time**: 1 week
**Go/No-Go**: If NO strategy is profitable → STOP

### Phase 0b: Ensemble Improvement (Week 2)

**Objective**: Prove combining strategies adds value

| Test | Comparison | Success Criteria |
|------|------------|------------------|
| Equal-weight ensemble vs Best individual | Sharpe ratio | Ensemble > Best |
| Ensemble vs Individual | Max drawdown | Ensemble lower |
| Strategy correlation | Pairwise correlation | < 0.3 |

**Cost**: $0
**Time**: 3-5 days
**Go/No-Go**: If ensemble doesn't beat best individual → STOP

### Phase 1: NichePopulation Mechanism (Week 3)

**Objective**: Prove the specialization mechanism adds value

| Test | Comparison | Success Criteria |
|------|------------|------------------|
| NichePopulation vs Equal-weight | Sharpe ratio | NichePopulation > Equal |
| NichePopulation vs Best individual | Total return | NichePopulation > Best |
| Winner analysis | Which agent wins when | Different agents win in different periods |

**Architecture**:
- 5 agents with learnable parameters
- Thompson Sampling for strategy selection
- Winner-take-all competition per week
- Fitness = rolling 30-day Sharpe

**Cost**: $0
**Time**: 1 week
**Go/No-Go**: If NichePopulation doesn't beat equal-weight → reassess mechanism

### Phase 2: SI Validation (Week 4)

**Objective**: Confirm specialization emerges and correlates with profit

| Metric | Target | Validation |
|--------|--------|------------|
| SI | > 0.4 | Agents have differentiated parameters |
| Regime-matching | Visual | Different agents dominate different market conditions |
| Profit attribution | Analysis | SI periods correlate with higher returns |

**Cost**: $0
**Time**: 3-5 days

### Phase 3: Advanced Architecture (Month 2+)

**Only if Phase 2 succeeds**

- Replace Thompson Sampling with Neural Contextual Bandits
- Add regime detection features (volatility, correlation, trend)
- Implement soft regime assignments
- Test on more assets (10-20 liquid stocks/ETFs)

**Cost**: ~$100-500 (compute, optional data)
**Time**: 4-6 weeks

---

## 🛠 Technical Architecture

### Phase 0-2: Simple Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DATA LAYER                         │
│  - Yahoo Finance API (free)                         │
│  - Daily OHLCV for BTC, ETH, SPY                   │
│  - 3-5 years of history                            │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                     │
│  - Returns (1d, 5d, 20d)                           │
│  - Volatility (20d rolling)                        │
│  - Momentum indicators                              │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                  AGENT LAYER                        │
│  - 5 agents with learnable parameters              │
│  - Each has: lookback, threshold, stop-loss        │
│  - Thompson Sampling for action selection          │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               COMPETITION LAYER                      │
│  - Weekly evaluation period                         │
│  - Fitness = Sharpe ratio                          │
│  - Winner-take-all update                          │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              EVALUATION LAYER                        │
│  - Profit metrics (return, Sharpe, drawdown)       │
│  - SI measurement                                   │
│  - Regime analysis                                  │
└─────────────────────────────────────────────────────┘
```

### Phase 3+: Advanced Architecture

```
┌─────────────────────────────────────────────────────┐
│               REGIME DETECTION                       │
│  - DCC-GARCH / Markov-Switching                    │
│  - Neural feature extraction                        │
│  - Soft regime assignments (probabilities)         │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│           NEURAL CONTEXTUAL BANDITS                  │
│  - Context = market features + regime probs        │
│  - Neural network estimates rewards                 │
│  - Uncertainty-aware action selection              │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              SPECIALIST AGENTS                       │
│  - Trend-following specialist                       │
│  - Mean-reversion specialist                        │
│  - Volatility-targeting specialist                 │
│  - Defensive/hedging specialist                    │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
apps/trading/
├── docs/
│   ├── TRADING_RESEARCH_PLAN.md (this file)
│   ├── ARCHITECTURE_DECISIONS.md
│   └── RESULTS_ANALYSIS.md
├── experiments/
│   ├── phase0/
│   │   ├── test_individual_strategies.py
│   │   └── test_ensemble.py
│   ├── phase1/
│   │   └── test_niche_population.py
│   └── phase2/
│       └── validate_si.py
├── src/
│   ├── agents/
│   │   ├── base_agent.py
│   │   └── thompson_agent.py
│   ├── strategies/
│   │   ├── momentum.py
│   │   ├── mean_reversion.py
│   │   └── volatility.py
│   ├── competition/
│   │   └── niche_population.py
│   └── evaluation/
│       ├── metrics.py
│       └── visualization.py
└── README.md
```

---

## ⚠️ Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Individual strategies unprofitable | Medium | High (blocks all progress) | Test multiple simple strategies first |
| Overfitting to historical data | High | High | Use walk-forward validation, out-of-sample tests |
| Specialization doesn't emerge | Medium | Medium | Try different fitness functions, competition rules |
| Transaction costs kill profits | Medium | High | Include realistic costs from Phase 0 |
| Regime transitions too noisy | Medium | Medium | Use soft assignments, longer evaluation periods |

---

## 📚 References

### Academic Papers
- Markowitz, H. (1952). Portfolio Selection
- Thompson, W. R. (1933). On the Likelihood that One Unknown Probability Exceeds Another
- Two Sigma: "A Machine Learning Approach to Regime Modeling"

### Modern Approaches
- NeuralUCB, NeuralTS (2020+)
- Decision Transformer (Chen et al., 2021)
- FinRL: Deep Reinforcement Learning for Finance

### Reviewed Research (January 2026)
- **AgentEvolver** (arxiv 2511.10395): Self-evolving agents with self-attributing mechanism
  - Useful for Phase 2+: Credit attribution for understanding specialist success
  - Not for Phase 0-1: Over-engineered for basic validation

### Our Prior Work
- Paper 1: NichePopulation (SI=0.747 across 6 domains)
- Paper 2: Emergent Preference Specialization in LLM Agents
- Paper 3: Emergent Tool Specialization

---

## 💡 The SI → Profit Innovation

### Core Thesis Refinement

Our contribution isn't just "SI emerges in trading" — it's:

> **"Emergent specialization (SI) PREDICTS and CAUSES better trading performance"**

This requires proving a CAUSAL link:

```
Competition → Specialists Emerge (SI > 0.4)
     ↓
Specialists Win in Different Conditions
     ↓
Ensemble Outperforms (Sharpe improvement)
     ↓
SI Correlates with Profit (r > 0.3, p < 0.05)
```

### Novel Contributions for Trading

| Contribution | Description | Validated In |
|--------------|-------------|--------------|
| **Emergent Regime Discovery** | Specialists define regimes by WHAT WORKS, not arbitrary features | Phase 1 |
| **SI as Profit Predictor** | Show statistical link between SI and returns | Phase 1 |
| **Attribution-Guided Specialization** | Use credit assignment to improve specialist learning | Phase 2 |
| **Self-Evolving Trading Population** | System improves without manual intervention | Phase 3 |

### Differentiation from Existing Work

| Existing Approach | Our Approach | Key Difference |
|-------------------|--------------|----------------|
| Pre-define regimes, assign strategies | Let regimes EMERGE from competition | No manual labeling |
| Train router to select models | Competition produces specialists | No supervised training |
| RL for single trading agent | Population-based competition | Diversity built-in |
| Thompson Sampling for exploration | Thompson Sampling + Niche Affinity | Specialization, not just exploration |

---

## 🎯 Next Steps

1. [ ] Create Phase 0a backtesting scripts
2. [ ] Download historical data (BTC, ETH, SPY)
3. [ ] Implement basic strategies (momentum, mean-reversion)
4. [ ] Run Phase 0a tests
5. [ ] Document results and make go/no-go decision
6. [ ] If Phase 0 succeeds: Measure SI-Profit correlation
7. [ ] If Phase 1 succeeds: Add attribution mechanism

---

*Last Updated: January 17, 2026*
