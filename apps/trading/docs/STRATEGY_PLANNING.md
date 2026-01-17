# Strategy Planning: Trading Application

**Date**: January 17, 2026  
**Purpose**: Keep development on track and aligned with core thesis

---

## 🎯 Our Core Thesis (DO NOT LOSE SIGHT)

> **Competitive specialization among agents produces emergent specialists that, when combined, outperform individual strategies or naive ensembles.**

This is the thesis from our research series:
- **Paper 1 (NichePopulation)**: Proved SI=0.747 across 6 domains with rule-based learners
- **Paper 2 (Preference Specialization)**: Proved LLM agents develop stable preferences
- **Paper 3 (Tool Specialization)**: Proved LLM agents specialize in tools (+83% advantage)

**Trading Application Goal**: Extend this thesis to financial trading.

---

## 📍 Where We Are

### Repositories in Our Research Series

| Repo | Paper | Status | Focus |
|------|-------|--------|-------|
| [NichePopulation](https://github.com/HowardLiYH/NichePopulation) | Paper 1 | ✅ Published/arXiv ready | Rule-based learners, time series |
| [Emergent-Preference-Specialization](https://github.com/HowardLiYH/Emergent-Preference-Specialization-in-LLM-Agent-Populations) | Paper 2 | ✅ arXiv ready | LLM agents, synthetic rules |
| [Emergent-Tool-Specialization](https://github.com/HowardLiYH/Emergent-Tool-Specialization) | Paper 3 | 🔄 In progress | LLM agents, real MCP tools |
| [Emergent-Applications](https://github.com/HowardLiYH/Emergent-Applications) | Applications | 🔄 In progress | Practical applications |

### Current Task

Building the **Trading Application** in `Emergent-Applications/apps/trading/`

---

## 🔗 PopAgent: Integration Assessment

[PopAgent (MAS_For_Finance)](https://github.com/HowardLiYH/MAS_For_Finance) is a SEPARATE project that shares some mechanisms.

### What PopAgent Has That We Can Use

| Component | PopAgent Location | Useful For Us? | How to Integrate |
|-----------|-------------------|----------------|------------------|
| **Thompson Sampling** | `trading_agents/` | ✅ YES | Copy algorithm, adapt to our agents |
| **Data pipeline (Bybit)** | `data_pipeline/` | ✅ YES | Reuse for crypto data |
| **Multi-asset features** | `trading_agents/` | ✅ YES | Cross-asset signals |
| **Backtesting framework** | `trading_agents/` | ✅ YES | Adapt for our experiments |
| **Feature-Aligned Learning** | `trading_agents/` | ⚠️ MAYBE | Consider for Phase 2+ |
| **LLM agents** | `trading_agents/` | ❌ NO | Our thesis works without LLM |
| **Method inventory (15+ methods)** | `trading_agents/` | ⚠️ MAYBE | Start simpler (3 methods) |
| **Dashboard** | `dashboard/` | ⚠️ MAYBE | Nice to have, not priority |

### What We Should NOT Import

- LLM dependency (expensive, may not add value for trading)
- Complexity before validation (Feature-Aligned Learning before basics work)
- Their specific agent roles (Analyst, Researcher, etc.)

---

## 📋 Our Plan (Staying on Track)

### Phase 0: Validate Core Thesis (Week 1-2)

**Question**: Does our NichePopulation mechanism improve trading?

**What to build**:
1. 3 simple trading strategies (momentum, mean-reversion, volatility)
2. NichePopulation competition mechanism (from Paper 1)
3. Simple backtest

**What to reuse from PopAgent**:
- Data pipeline (Bybit CSVs)
- Thompson Sampling implementation
- Basic feature calculations

**Success criteria**:
- NichePopulation ensemble > Single best strategy
- NichePopulation ensemble > Equal-weight ensemble
- Positive returns after costs

### Phase 1: Confirm with Multiple Assets (Week 3-4)

**If Phase 0 succeeds**:
- Extend to BTC, ETH, SOL
- Add transaction costs
- Measure SI (do agents specialize?)

### Phase 2: Add Sophistication (Month 2+)

**Only after Phase 1 validates**:
- Consider PopAgent's Feature-Aligned Learning
- Consider more methods in inventory
- Consider cross-asset features

---

## ⚠️ Guardrails (Don't Get Lost Again)

### Always Ask:

1. **"Does this validate our thesis?"** - If not, deprioritize
2. **"Is this the simplest test?"** - Complexity comes later
3. **"Do we have profit results?"** - No architecture without results
4. **"Are we extending OUR work or building something new?"** - Stay focused

### Red Flags:

- ❌ Building complex systems before basic validation
- ❌ Adding LLM when rule-based suffices
- ❌ Copying PopAgent wholesale instead of extracting useful pieces
- ❌ Forgetting that profit is the success metric, not SI

---

## 🔄 Integration Plan: PopAgent → Emergent-Applications

### Step 1: Extract Data Pipeline

```bash
# Copy only what we need
cp -r MAS_For_Finance/data/bybit/ Emergent-Applications/apps/trading/data/
cp MAS_For_Finance/data_pipeline/bybit_loader.py Emergent-Applications/apps/trading/src/data/
```

### Step 2: Extract Thompson Sampling

```bash
# Adapt Thompson Sampling for our agent structure
# Source: MAS_For_Finance/trading_agents/
# Target: Emergent-Applications/apps/trading/src/agents/
```

### Step 3: Build OUR Competition Mechanism

Use Paper 1's NichePopulation algorithm, NOT PopAgent's architecture:
- Winner-take-all selection
- Niche affinity tracking
- Fitness sharing (optional)

### Step 4: Simple Backtest

Build a minimal backtest that answers:
- Does NichePopulation beat single strategy?
- Does NichePopulation beat equal-weight?
- Are returns positive?

---

## 📊 Success Metrics (In Priority Order)

1. **Profit** - Net return after costs (PRIMARY)
2. **Sharpe** - Risk-adjusted return
3. **Drawdown** - Maximum loss
4. **SI** - Specialization Index (SECONDARY - validates mechanism)

---

## 📅 Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Extract PopAgent data + Thompson Sampling | `src/data/`, `src/agents/` |
| 1 | Build 3 simple strategies | `src/strategies/` |
| 2 | Build NichePopulation mechanism | `src/competition/` |
| 2 | Run Phase 0 backtest | Results: Sharpe, return, drawdown |
| 3 | Go/No-Go decision | If positive → Phase 1 |
| 3-4 | Multi-asset validation | BTC, ETH, SOL results |
| 5+ | Add sophistication (if validated) | Feature-Aligned Learning, etc. |

---

## 🎯 Current Status

- [x] Created trading folder structure
- [x] Documented research plan
- [x] Documented architecture decisions
- [x] Created this strategy planning doc
- [ ] Extract data pipeline from PopAgent
- [ ] Extract Thompson Sampling from PopAgent
- [ ] Build simple strategies
- [ ] Build NichePopulation mechanism
- [ ] Run Phase 0 backtest

---

## 📝 Key Reminders

1. **We are extending Papers 1-3 to trading, not building a new system**
2. **PopAgent is a resource to borrow from, not a project to complete**
3. **Profit validates our thesis, not architecture complexity**
4. **Start simple, add complexity only after validation**

---

*Last Updated: January 17, 2026*
