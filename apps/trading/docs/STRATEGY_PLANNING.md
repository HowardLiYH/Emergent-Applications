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

### ⚡ NEW APPROACH: Discover What SI Measures FIRST

**Old thinking**: Assume SI → Profit, test if true
**New thinking**: Find what SI correlates with, THEN trace to profit

> **"What does SI actually measure? What is it related to?"**

See `SI_EXPLORATION.md` for the full list of 18 wild hypotheses.

---

### Phase 0: Discovery Phase (Week 1)

**THE QUESTION**: What is SI actually measuring?

**The Discovery Protocol**:
1. **Setup**: 5 agents, 3 strategies, BTC/ETH data
2. **Run**: NichePopulation competition, compute SI time series
3. **Correlate SI with EVERYTHING**:

```python
# Market State
correlate(SI, volatility)           # Does SI emerge in calm markets?
correlate(SI, trend_strength)       # Does SI need trends?
correlate(SI, return_entropy)       # Is SI about predictability?
correlate(SI, regime_stability)     # Is SI about regime persistence?

# Risk Metrics
correlate(SI, max_drawdown)         # Does SI reduce risk?
correlate(SI, volatility_of_vol)    # Does SI like stable conditions?

# Agent Behavior
correlate(SI, agent_correlation)    # Does SI = diversification?
correlate(SI, winner_spread)        # Does SI = competitive intensity?

# Profit (one of many tests, not THE test)
correlate(SI, profit)               # Direct relationship?

# SI as Predictor
correlate(SI_today, profit_tomorrow)    # Leading indicator?
correlate(SI_today, volatility_tomorrow) # Warning signal?

# SI Dynamics
correlate(dSI/dt, profit)           # Is SI change more important?
```

**Output**: Top 10 strongest SI correlations

---

### Phase 0 Decision Tree

```
After correlation analysis:

┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  SI correlates strongly with PROFIT directly?               │
│     → Original thesis validated → Proceed                   │
│                                                              │
│  SI correlates with RISK METRICS (drawdown, vol)?           │
│     → Pivot: "SI for risk management"                       │
│     → Value: Lower risk, better Sharpe                      │
│                                                              │
│  SI correlates with DIVERSIFICATION (agent correlation)?    │
│     → Pivot: "SI for portfolio construction"                │
│     → Value: Uncorrelated alpha streams                     │
│                                                              │
│  SI correlates with PREDICTABILITY (entropy, autocorr)?     │
│     → Pivot: "SI as market regime indicator"                │
│     → Value: Meta-signal for when to trade                  │
│                                                              │
│  SI correlates with REGIME STABILITY?                       │
│     → Pivot: "SI as regime detector"                        │
│     → Value: Position sizing, risk allocation               │
│                                                              │
│  SI predicts NEXT-DAY returns/vol?                          │
│     → Pivot: "SI as leading indicator"                      │
│     → Value: Timing signal                                  │
│                                                              │
│  SI correlates with NOTHING significant?                    │
│     → Try: Nonlinear tests, regime-stratified analysis     │
│     → Or: SI might not be useful for trading                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### Fallback Paths to Value

Even if SI ≠ Profit directly, SI might still be valuable:

| Fallback | What SI Correlates With | Value Proposition |
|----------|-------------------------|-------------------|
| **Risk Manager** | Drawdown, VaR | Lower risk, better Sharpe |
| **Diversifier** | Agent correlation | Uncorrelated returns |
| **Regime Detector** | Regime stability | Know when to trade |
| **Leading Indicator** | Next-day returns | Timing signal |
| **Anti-Crowding** | Strategy overlap | Avoid crowded trades |
| **Stability Metric** | System health | Monitor trading system |

**What to reuse from PopAgent**:
- Data pipeline (Bybit CSVs)
- Thompson Sampling implementation
- Basic feature calculations

---

### Phase 1: Confirm Robustness (Week 2-3)

**Only if Phase 0 passes ALL criteria**:

1. **Multi-asset**: Add SOL, extend time period
2. **Regime analysis**: When does SI help most? (volatile vs calm)
3. **Ablation**: What if no competition? (baseline comparison)

**Success Criteria**:
| Test | Success |
|------|---------|
| SI→Profit holds on SOL | r > 0, p < 0.05 |
| SI→Profit holds in different periods | Consistent across windows |
| NichePopulation > Equal-weight | Sharpe improvement > 10% |
| NichePopulation > Single-best | Sharpe improvement > 5% |

---

### Phase 2: Add Sophistication (Month 2+)

**Only after Phase 1 validates across assets/periods**:
- Consider PopAgent's Feature-Aligned Learning
- Consider more methods in inventory
- Consider attribution-guided learning (from AgentEvolver)

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

| Week | Task | Deliverable | Gate |
|------|------|-------------|------|
| 1 Day 1-2 | Extract PopAgent data + Thompson Sampling | `src/data/`, `src/agents/` | |
| 1 Day 3-4 | Build 3 strategies + NichePopulation | `src/strategies/`, `src/competition/` | |
| 1 Day 5 | Run backtest, collect SI + ALL features | Raw data dump | |
| 1 Day 6 | **Correlation analysis** | **Top 10 SI correlates** | |
| **Week 1 End** | **Interpretation: What does SI measure?** | **Discovery report** | **PIVOT DECISION** |
| 2 | Follow strongest path (profit/risk/signal) | Targeted validation | |
| 3 | Multi-asset robustness | BTC, ETH, SOL | |
| 4+ | Add sophistication based on findings | Attribution, etc. | |

---

## 🎯 Current Status

**Priority: DISCOVER what SI measures, THEN trace to profit**

- [x] Created trading folder structure
- [x] Documented research plan  
- [x] Documented architecture decisions
- [x] Created this strategy planning doc
- [x] Created SI→Profit innovation doc
- [x] Created SI Exploration doc (18 wild hypotheses)
- [ ] **NEXT: Extract data pipeline from PopAgent**
- [ ] Extract Thompson Sampling from PopAgent
- [ ] Build simple strategies (momentum, mean-reversion, breakout)
- [ ] Build NichePopulation mechanism
- [ ] Run backtest + collect all features
- [ ] **CRITICAL: Correlation analysis - What does SI relate to?**
- [ ] Interpretation + pivot decision

---

## 📝 Key Reminders

1. **We are extending Papers 1-3 to trading, not building a new system**
2. **PopAgent is a resource to borrow from, not a project to complete**
3. **Profit validates our thesis, not architecture complexity**
4. **Start simple, add complexity only after validation**

---

## 🔬 Research Integration: External Ideas

### Reviewed Papers

| Paper | Source | Key Concept | Relevance |
|-------|--------|-------------|-----------|
| **AgentEvolver** | arxiv 2511.10395 | Self-evolving agents with self-questioning, self-navigating, self-attributing | ⚠️ Phase 3+ |

### AgentEvolver: What's Useful (and What's Not)

**NOT Useful for Phase 0-1**:
- Self-Questioning (task generation) — We have clear tasks already
- Complex LLM-based agents — Over-engineered for trading
- Context-Managing Templates — We don't need long-horizon reasoning

**POTENTIALLY Useful for Phase 2+**:
- **Self-Attributing**: Assign credit to individual trading decisions
  - Problem: Trade today, profit/loss realized later. Which decision mattered?
  - Solution: Attribution analysis to understand WHY specialists win
  - Application: Help specialists learn BETTER, not just compete

### Creative Integration: SI → Profit Architecture

The key insight is linking our Specialization Index (SI) directly to profitability:

```
┌─────────────────────────────────────────────────────────────┐
│                    SI → PROFIT PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PHASE 0-1: VALIDATE BASICS                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │Strategy │ → │NichePop │ → │ Profit  │                   │
│  │  Pool   │    │Competition│   │ Check   │                 │
│  └─────────┘    └─────────┘    └─────────┘                 │
│       ↓              ↓              ↓                       │
│   3 strategies   Winners emerge   Is profit > baseline?    │
│                                                              │
│  PHASE 2: ADD ATTRIBUTION (from AgentEvolver)               │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │Specialist│ → │ Trade   │ → │ Attribute│                 │
│  │ Trades  │    │ Result  │    │ Credit  │                 │
│  └─────────┘    └─────────┘    └─────────┘                 │
│       ↓              ↓              ↓                       │
│   Agent A trades   +5% return    Entry was good,           │
│   BTC long                       Exit timing helped         │
│                                                              │
│  PHASE 3: FULL SELF-EVOLUTION                               │
│  ┌─────────────────────────────────────────┐               │
│  │ Specialists learn from attribution      │               │
│  │ → Improve strategy parameters           │               │
│  │ → Better SI → Better Profit            │               │
│  └─────────────────────────────────────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### The Core Innovation: SI as Profit Predictor

Our thesis isn't just "SI emerges" — it's **"SI leads to profit"**.

To prove this, we need to show:

| Hypothesis | Test | Success Criteria |
|------------|------|------------------|
| H1: Specialists emerge | Measure SI | SI > 0.4 |
| H2: Specialists are profitable | Measure profit per specialist | Each specialist profitable in their niche |
| H3: Ensemble beats individuals | Compare ensemble vs single-best | Ensemble Sharpe > Best individual |
| H4: SI correlates with profit | Regression: SI → Profit | Positive, significant correlation |

**H4 is the key innovation**: We're not just showing SI exists, we're showing it PREDICTS profitability.

---

## 💡 Creative Architecture Ideas

### Idea 1: Profit-Weighted Competition (Novel)

Instead of pure winner-take-all, weight competition by BOTH performance AND specialization:

```python
# Traditional: Winner takes all
winner = argmax(returns)

# Novel: Profit-weighted with diversity bonus
scores = returns + alpha * diversity_contribution
winner = argmax(scores)
```

This incentivizes specialists to be BOTH profitable AND different.

### Idea 2: Attribution-Guided Learning (From AgentEvolver)

After Phase 1 validation, add attribution:

```python
# For each trade sequence
for trade in specialist_trades:
    # Decompose profit into components
    entry_contribution = attribute_to_entry(trade)
    timing_contribution = attribute_to_timing(trade)
    sizing_contribution = attribute_to_sizing(trade)

    # Update specialist based on what worked
    specialist.learn_from_attribution(entry_contribution, ...)
```

### Idea 3: Niche Affinity as Market Regime Detector

Our specialists don't just pick strategies—they DISCOVER market regimes:

```
Specialist A wins in: High volatility, trending
Specialist B wins in: Low volatility, mean-reverting
Specialist C wins in: Choppy, uncertain

→ The PATTERN of who wins tells us the current regime
→ We didn't define regimes; they EMERGED from competition
```

This is powerful because:
1. No manual regime labeling needed
2. Regimes are defined by what WORKS, not arbitrary features
3. Adapts automatically as markets change

---

## 🎯 Revised Phase Plan (With Research Integration)

### Phase 0: Pure Validation (No Research Integration)
- 3 strategies, NichePopulation, basic backtest
- Success = Profit > Baseline
- **Do NOT add attribution, self-evolution, etc.**

### Phase 1: Multi-Asset + SI Measurement
- Extend to BTC, ETH, SOL
- Measure SI and profit correlation
- Success = SI emerges AND correlates with profit

### Phase 2: Attribution Integration (From AgentEvolver)
- Add credit attribution to understand WHY specialists win
- Use attribution to improve specialist learning
- Success = Specialists improve over time

### Phase 3: Full Self-Evolution
- Specialists adapt their strategies based on performance
- Continuous learning from market feedback
- Success = System improves without manual intervention

---

## 📚 Research To Revisit Later

When we reach Phase 2+, revisit these concepts:

1. **AgentEvolver's Self-Attributing**: For credit assignment
2. **Feature-Aligned Learning** (from PopAgent): For regime-aware updates
3. **Neural Contextual Bandits**: For complex state handling
4. **Mixture of Experts**: For learned routing

But NOT NOW. Phase 0 first.

---

*Last Updated: January 17, 2026*
