# Understanding Check: Emergent Trading Specialists

**Date**: January 17, 2026
**Purpose**: Document my understanding of this project for review
**GitHub**: https://github.com/HowardLiYH/Emergent-Applications/tree/main/apps/trading

---

## 🎯 Core Research Question

> **"What does Specialization Index (SI) actually correlate with in financial trading, and does that correlation lead to profit?"**

This is a **discovery-first approach**, NOT an assumption-first approach. We don't assume SI → Profit; we discover what SI measures first, then trace to profit.

---

## 🧬 Research Foundation

This project extends findings from three prior papers in the Emergent Specialization research series:

| Paper | Focus | Key Finding | Status |
|-------|-------|-------------|--------|
| **Paper 1: NichePopulation** | Learner populations (time series) | SI=0.747, Cohen's d > 20 | ✅ Published (arXiv) |
| **Paper 2: Preference Specialization** | Synthetic rules (LLM agents) | Competition alone = 94% of specialization | ✅ Published (arXiv) |
| **Paper 3: Tool Specialization** | Real MCP APIs | Early experiments show specialist advantage | 🔄 In Progress |

> **Note**: Paper 3 is in a separate repo (`Emergent-Tool-Specialization`) and is still experimental. The "+83.3% specialist advantage" was an early finding but not yet fully validated.

The core discovery: **Specialization emerges spontaneously from competition alone** — no explicit training, role assignment, or gradient updates needed.

---

## 🔬 What is Specialization Index (SI)?

SI measures how specialized agents are in a population:

```
SI = 1 - mean(normalized_entropy of niche_affinities)

High SI (→1): Agents have distinct niches (specialists)
Low SI (→0): Agents are similar (generalists)
```

**Primary SI variant for analysis**: `si_rolling_7d` (7-day rolling window)
- This is specified in `pre_registration.json` as the main variant to test
- Other variants (1h, 4h, 1d) tested in Pipeline 3 (SI Dynamics)

### The NichePopulation Mechanism

1. Agents compete for rewards (profit in trading)
2. Winner-take-all selection each period
3. Agents track which "niches" (market regimes) they win in
4. Over time, agents develop different niche affinities
5. SI emerges as agents specialize

---

## 📊 The Three Pipelines

### Pipeline 1: Discovery (46 features)
**Question**: What does SI correlate with?

Features by category (from `FEATURE_PIPELINE_AUDIT.md`):

| Category | Count | Examples |
|----------|-------|----------|
| Market | 15 | volatility, trend_strength, Hurst, entropy, ADX, RSI, ATR |
| Agent (safe) | 6 | agent_correlation, winner_spread, viable_agent_count |
| Risk | 10 | max_drawdown, VaR, CVaR, Sharpe, Sortino, profit_factor |
| Cross-Asset | 5 | asset_return_corr, relative_strength, rotation_signal |
| Behavioral | 6 | fear_greed_proxy, momentum_return, regime_duration |
| Liquidity | 4 | volume_z, amihud_log, spread_proxy |

**Removed (circular)**: `strategy_concentration`, `niche_affinity_entropy`, all SI-derived

### Pipeline 2: Prediction (2 features)
**Question**: Does SI predict future outcomes?

Targets:
- `next_day_return` (24h forward)
- `next_day_volatility`

Methods: Lagged correlation, Granger causality, signal decay analysis

### Pipeline 3: SI Dynamics (9 features)
**Question**: How should we use SI?

Variants tested:
- SI level: `si_1h`, `si_4h`, `si_1d`, `si_1w`
- SI derivatives: `dSI/dt`, `si_acceleration`
- SI stability: `si_std`, `si_percentile`

---

## 🧪 40 Hypotheses Being Tested

Organized into 11 categories:

| Category | Key Hypotheses |
|----------|---------------|
| **Market State** | H1: SI = Regime Clarity, H2: SI = Inverse Entropy, H3: SI = Regime Persistence |
| **Agent Behavior** | H4: SI = Strategy Orthogonality, H5: SI = Learning Convergence, H6: SI = Niche Stability |
| **Microstructure** | H7: SI = Liquidity State, H8: SI = Information Flow Clarity |
| **Meta/Abstract** | H9-H12: Competitive Intensity, Complexity Matching, Alpha Persistence, Fractal Self-Similarity |
| **Wild Ideas** | H13-H18: Market Mood, Leading Indicator, SI Velocity, Crash Warning, Anti-Crowding |
| **Risk** | H19-H22: Signal-to-Noise, Loss Avoidance, Tail Protection, Drawdown Recovery |
| **Timing** | H23-H25: Trading Windows, Momentum/MeanRev Switch, Time-to-Profit |
| **Behavioral** | H26-H28: Contrarian at Extremes, Retail vs Institutional, Sentiment Momentum |
| **Factor** | H29-H31: Factor Tilt, Cross-Asset Signals, Sector Rotation |
| **Operational** | H32-H35: Resource Allocation, Model Calibration, SI Stability, Leverage Optimization |
| **Moonshot** | H36-H40: Macro Prediction, Black Swan Detection, Multi-Timescale, Ensemble SI, Network Effects |

---

## 📋 Methodology Safeguards

| Safeguard | Implementation |
|-----------|----------------|
| **Train/Val/Test split** | 70/15/15 temporal (no shuffle) |
| **Autocorrelation handling** | HAC standard errors + block bootstrap |
| **Multiple testing** | Pre-registration + FDR correction (Benjamini-Hochberg) |
| **Circular reasoning prevention** | Predictive (lagged) correlations only |
| **Validation** | 5-fold rolling cross-validation |
| **Negative controls** | Random noise SI, shuffled SI permutation tests |
| **Causality** | Granger causality, placebo tests |
| **Transaction costs** | Sensitivity analysis (noted as known limitation) |

### ⚠️ Critical Additions from Expert Panel Review (MUST KNOW)

| Fix | Details |
|-----|---------|
| **Circular Features Removed** | SI-derived features (si_1h, dSI/dt, etc.) were moved to Pipeline 3 ONLY. Discovery Pipeline uses 46 external market features only. Mixing SI with SI-derived correlations would be circular. |
| **Data-Driven Block Size** | Block size for bootstrap is NOT fixed at 24. Uses `optimal_block_size()` based on autocorrelation structure. |
| **Effective N Adjustment** | Sample size adjusted for autocorrelation: `effective_n = n / (1 + 2*sum(acf))`. Prevents overestimating significance. |
| **Regime-Conditioned Analysis** | Correlations checked WITHIN each regime. A correlation can flip sign in different regimes, making aggregate correlation meaningless. |
| **Timezone Standardization** | ALL data converted to UTC before analysis. Crypto = already UTC, Forex = session-aware, Stocks = market hours only. |
| **Data Validation (Phase 2.1)** | CRITICAL step added: Check for missing values, duplicate timestamps, gaps >1hr, extreme returns (>30%), negative prices, OHLC consistency BEFORE any analysis. |

---

## 🌍 Multi-Market Scope

| Market Type | Assets | Characteristics |
|-------------|--------|-----------------|
| **Crypto** | BTC, ETH, SOL | 24/7, high volatility, start here |
| **Forex** | EUR/USD, GBP/USD | 24/5, lower vol, unreliable volume |
| **Stocks** | SPY, QQQ, AAPL | 6.5h/day, earnings, overnight gaps |
| **Commodities** | Gold, Oil, Corn | Seasonal, supply/demand driven |

### Testing Order (IMPORTANT)

1. Crypto (BTC) ← **Start here** (most data, 24/7)
2. Crypto (ETH, SOL) ← Validate within market
3. Forex (EUR/USD) ← Cross-market validation
4. Stocks (SPY) ← Different market structure
5. Commodities (Gold) ← Alternative asset class

**Success criteria**: SI findings must hold in at least 2 of 4 market types

---

## 📍 Current Project Status

```
[x] Phase 0: Planning & Methodology — COMPLETE
[ ] Phase 1: Pre-Registration & Setup — YOU ARE HERE
    [ ] Commit pre_registration.json to GitHub
    [ ] Create src/ directory structure
    [ ] Run smoke test
[ ] Phase 2: Data & Features (incl. validation + multi-market)
[ ] Phase 2.1: DATA VALIDATION (CRITICAL - cannot skip)
[ ] Phase 3: Backtest & SI Computation
[ ] Phase 4: Discovery Pipeline
[ ] Phase 5: Prediction Pipeline
[ ] Phase 6: SI Dynamics Pipeline
[ ] Phase 7: Audits & Validation
[ ] Phase 8: Final Report
[ ] Phase 9: Cross-Market Validation
```

### ⏱️ Phase Time Budgets (ENFORCED)

| Phase | Max Time | Checkpoint |
|-------|----------|------------|
| Phase 1 | 2 hours | src/ exists, smoke test passes |
| Phase 2 | 4 hours | Data downloaded, validated |
| Phase 3 | 3 hours | SI time series computed |
| Phase 4 | 4 hours | Top correlations identified |
| Phase 5 | 3 hours | Granger tests complete |
| Phase 6 | 2 hours | SI variants compared |
| Phase 7 | 3 hours | Validation complete |
| Phase 8 | 2 hours | Report written |
| Phase 9 | 8 hours | Cross-market tested |

### 🚨 Contingency Plans

| Problem | Action |
|---------|--------|
| Binance API fails | Fallback to yfinance/CryptoCompare |
| No significant correlations | Check methodology, try different lags, document null result |
| Insufficient data | Reduce lookback to 180 days minimum |
| Computational timeout | Sample 10% of data for quick validation |
| SI computation NaN | Check for division by zero, log edge cases |

### 🛑 Stopping Criteria

| Condition | Action |
|-----------|--------|
| Data validation fails | STOP - fix data before proceeding |
| Smoke test fails | STOP - debug core algorithm |
| <500 hours of data | STOP - insufficient sample size |
| All 46 correlations ≈ 0 (|r| < 0.05) | Document null result, consider pivot |

### ✅ Exit Criteria

| Outcome | Definition |
|---------|------------|
| **SUCCESS** | ≥3 features |r| > 0.15, FDR < 0.05, replicates in ≥2 markets |
| **PARTIAL SUCCESS** | ≥1 feature significant, limited to one market |
| **NULL RESULT** | No significant correlations (still publishable) |
| **FAILURE** | Methodology flaw discovered, results unreliable |

### What Exists
- `experiments/pre_registration.json` — Pre-registered hypotheses (not yet committed)
- `MASTER_PLAN.md` — **THE PRIMARY EXECUTION DOCUMENT** (read this first!)
- `docs/` — Comprehensive research documentation (14+ files)
- `future/` — Parked enhancements (live trading, Polymarket, regime prediction)

### What's NOT Built Yet (⚠️ CRITICAL - NOTHING EXISTS!)

**Current state of the codebase:**
- `src/` — **COMPLETELY EMPTY** - all code must be extracted from MASTER_PLAN.md
- `experiments/phase0/` — **EMPTY**
- `experiments/phase1/` — **EMPTY**
- `experiments/phase2/` — **EMPTY**
- `experiments/pre_registration.json` — ✅ EXISTS (commit this first!)

**What needs to be created (in order):**

1. **Utility modules** (`src/utils/`):
   - `logging_setup.py`, `safe_math.py`, `timezone.py`
   - `reproducibility.py`, `checkpointing.py`, `caching.py`

2. **Data modules** (`src/data/`):
   - `loader.py`, `validation.py`

3. **Core modules**:
   - `src/analysis/features.py`, `src/analysis/correlations.py`
   - `src/agents/strategies.py`
   - `src/competition/niche_population.py`

4. **Experiment scripts** (`experiments/`):
   - `smoke_test.py`, `validate_data.py`, `run_backtest.py`
   - `run_discovery.py`, `run_prediction.py`, `run_dynamics.py`
   - `run_validation.py`, `run_audits.py`, `generate_report.py`
   - `run_cross_market.py`

5. **Data directories and files**:
   - `data/crypto/BTCUSDT_1h.csv`, `ETHUSDT_1h.csv`, `SOLUSDT_1h.csv`
   - (Later: `data/forex/`, `data/stocks/`, `data/commodities/`)

**⚠️ The Quick Reference commands will NOT work until these files are created!**

### 📂 File Structure Explanation

| File | Purpose | Read Priority |
|------|---------|---------------|
| `MASTER_PLAN.md` | **MAIN EXECUTION PLAN** - contains everything | 🔴 FIRST |
| `docs/UNDERSTANDING_CHECK.md` | This file - orientation | 🔴 HIGH |
| `experiments/pre_registration.json` | Hypotheses (commit before analysis) | 🟡 MEDIUM |
| `docs/SI_EXPLORATION.md` | 40 hypotheses detailed | 🟡 REFERENCE |
| `docs/SI_CORRELATION_TEST_PLAN.md` | 7-day test plan | 🟡 REFERENCE |
| `docs/METHODOLOGY_AUDIT.md` | 16 issues fixed | 🟢 IF NEEDED |
| `docs/EXPERT_PANEL_*.md` | Expert feedback (4 files) | 🟢 IF NEEDED |
| `docs/FEATURE_PIPELINE_AUDIT.md` | Why 3 pipelines (critical reading) | 🟡 MEDIUM |
| `docs/STRATEGY_PLANNING.md` | High-level strategy | 🟢 IF NEEDED |
| `docs/TRADING_RESEARCH_PLAN.md` | Original research plan | 🟢 IF NEEDED |
| `docs/METHODOLOGY_AUDIT.md` | 16 issues and fixes | 🟢 IF NEEDED |
| `docs/ARCHITECTURE_DECISIONS.md` | Technical design choices | 🟢 IF NEEDED |
| `future/*.md` | Parked ideas (DO NOT START) | ⚫ LATER |

**Full docs/ folder (15 files)**: `ARCHITECTURE_DECISIONS.md`, `COMPREHENSIVE_AUDIT_IMPLEMENTATION.md`, `EXPERT_PANEL_ADDITIONAL_AUDIT.md`, `EXPERT_PANEL_FINAL_REVIEW.md`, `EXPERT_PANEL_MASTER_PLAN_REVIEW.md`, `FEATURE_PIPELINE_AUDIT.md`, `FINAL_EXPERT_REVIEW_BEFORE_EXECUTION.md`, `FINAL_READINESS_AUDIT.md`, `METHODOLOGY_AUDIT.md`, `SI_CORRELATION_TEST_PLAN.md`, `SI_EXPLORATION.md`, `SI_TO_PROFIT_INNOVATION.md`, `STRATEGY_PLANNING.md`, `TRADING_RESEARCH_PLAN.md`, `UNDERSTANDING_CHECK.md`

### ⚠️ README.md Discrepancies (Outdated)

The main `README.md` has some outdated information:
- Says "70+ features" but should be **57** (after circular removal)
- Says success is "≥5 after FDR" but `pre_registration.json` says **≥3**
- References `experiments/phase0_si_discovery.py` which **doesn't exist**

**Trust `MASTER_PLAN.md` over `README.md` for accurate information.**

**Key Principle**: When in doubt, check `MASTER_PLAN.md`. It's the single source of truth.

### 📝 Execution Log (To Fill During Execution)

MASTER_PLAN.md has an empty execution log at the top. **Fill this in as you go:**

| Date | Phase | Status | Notes |
|------|-------|--------|-------|
| | | | |
| | | | |

This provides an audit trail of what was done and when.

---

## ✅ Success Criteria

### Primary Success
- ≥3 features with |r| > 0.15, FDR < 0.05
- Confirmed in BOTH validation AND test sets
- SI Granger-causes at least one prediction target (p < 0.05)
- Replicates in ≥3 of 5 assets

### Fallback Paths (If SI ≠ Profit Directly)

| If SI correlates with... | Pivot to... | Value |
|--------------------------|-------------|-------|
| Risk metrics | SI for risk management | Lower drawdowns, better Sharpe |
| Agent correlation | SI for diversification | Uncorrelated alpha sources |
| Regime stability | SI as regime detector | Know when to trade aggressively |
| Next-day returns | SI as timing signal | Position sizing based on SI |
| Nothing | Deeper analysis or abandon | Null result is also a contribution |

---

## 🔑 Key Insights I've Gathered

### 1. Discovery-First Philosophy
The project deliberately avoids assuming SI → Profit. Instead, it correlates SI with **57 features** (46 discovery + 2 prediction + 9 dynamics) to discover what SI actually measures, then traces any correlation to profit.

> **Note**: Originally 70+ features were proposed, but after the FEATURE_PIPELINE_AUDIT, circular features were removed, leaving 57.

### 2. Rigorous Pre-Registration
The `pre_registration.json` must be committed BEFORE any data analysis to prevent p-hacking accusations. This is a critical scientific safeguard.

**Commitments in pre_registration.json:**
- ✅ Report ALL results (including nulls)
- ✅ No post-hoc hypothesis changes
- ✅ Document all deviations from plan
- ✅ **Publish regardless of outcome** (negative result is still a contribution)

### 3. Known Limitations (Acknowledged)
The MASTER_PLAN explicitly states this is **signal testing, not production trading**:
- Simple strategies (fine for SI testing)
- No market impact (not trading yet)
- No transaction costs (will need for live trading)
- Equal agent sizing (testing SI, not strategies)

### 4. Expert Panel Consultation
The methodology incorporated feedback from **multiple expert panels at different stages**:

| Stage | Panel Size | Output | File |
|-------|------------|--------|------|
| Hypothesis brainstorming | 14 experts | 40 hypotheses across 11 categories | `SI_EXPLORATION.md` |
| Methodology audit | 8 experts | 18 recommendations | `EXPERT_PANEL_FINAL_REVIEW.md` |
| MASTER_PLAN review | 12 experts | 17 issues, 25 recommendations | `EXPERT_PANEL_MASTER_PLAN_REVIEW.md` |

Experts included: Information Theorist, Complexity Scientist, Market Microstructure Expert, Behavioral Finance Researcher, Systems Dynamics Expert, Regime Detection Specialist, Quant Strategist, Portfolio Manager, Risk Manager, Execution Specialist, ML Researcher, Macro Strategist, Latency Specialist, Statistician

### 5. Future Enhancements (Parked in `future/` folder)
These are explicitly **NOT to be started until SI is validated**:

| File | Enhancement | Start When |
|------|-------------|------------|
| `REGIME_PREDICTION.md` | Markov Chain forecasting | SI shows significant correlation with regimes |
| `POLYMARKET_STRATEGY.md` | Betting market arbitrage | Regime prediction works + want new domain |
| `LIVE_TRADING.md` | Real money implementation | Backtests profitable + risk management ready |

**Decision Framework** (from `future/README.md`):
- If SI correlates with nothing → Pivot or abandon (don't proceed)
- If SI correlates with volatility/risk → Proceed to Regime Prediction
- If SI + Regime Prediction works → Consider Polymarket or Live Trading

### 6. ⚠️ CRITICAL USER PREFERENCE: Multi-Coin Only
**The user ONLY wants multi-coin/multi-asset backtests, NEVER single-coin backtests.**
- Always use `--symbols BTC,ETH,SOL` (multiple coins)
- NEVER run backtest on single asset
- This is a hard requirement

### 7. MASTER_PLAN.md is Self-Contained
The `MASTER_PLAN.md` file contains:
- All execution steps in order
- Inline Python code ready to extract
- All statistical methodology
- All checkpoints and decision points

**Do NOT look for code elsewhere.** Extract code directly from MASTER_PLAN.md into `src/`.

### 8. Three Separate Pipelines (NOT One)
Features cannot share one pipeline due to circularity:
- **Discovery Pipeline (46 features)**: SI vs external market features
- **Prediction Pipeline (2 features)**: SI(t) vs future returns/volatility
- **SI Dynamics Pipeline (9 features)**: Which SI variant works best

If you mix SI-derived features into Discovery, you get circular correlations (SI correlates with SI).

---

## 📝 My Understanding of the Value Proposition

### If This Works
1. **Novel trading signal** — SI as an emergent indicator that captures market "readability"
2. **Zero manual regime engineering** — Regimes emerge from competition, not arbitrary thresholds
3. **Self-adapting system** — New specialists emerge as market structure changes
4. **Cross-market validation** — Generalizable beyond crypto

### What Makes This Different from Traditional Approaches

| Traditional | This Approach |
|-------------|---------------|
| Define regimes manually | Regimes emerge from competition |
| Assign strategies to regimes | Specialists discover their own niches |
| Retrain when markets change | Self-adapts through ongoing competition |
| Requires domain expertise | Requires only fitness function (profit) |

---

## ❓ Questions/Clarifications Needed

1. **Data source**: Which specific API/source should be used for downloading OHLCV data? (MASTER_PLAN mentions Bybit, Binance, yfinance)
   > **ANSWER**: Use `ccxt` for crypto (Binance/Bybit), `yfinance` for stocks, `yfinance` for forex. See `src/data/loader.py` in MASTER_PLAN.

2. **First asset priority**: Should BTC be the first asset tested, or run all crypto simultaneously?
   > **ANSWER**: ALWAYS multi-asset. Use `--symbols BTC,ETH,SOL` minimum. Never single-coin.

3. **Computational resources**: How long is the backtest expected to run? Any GPU requirements?
   > **ANSWER**: CPU only. Estimated ~10-30 min for 365 days × 3 assets. No GPU needed.

4. **Pre-registration commit**: Should this be committed to a specific branch or main?
   > **ANSWER**: Commit to `main` with timestamp in commit message. This is the pre-registration anchor.

5. **Timeline expectations**: Is the 8-20 day estimate realistic given current progress?
   > **ANSWER**: Yes, with the caveat that data validation may reveal issues requiring fixes.

---

## 🎯 Immediate Next Steps (My Understanding)

### ⚠️ Reality Check: Almost NOTHING exists yet!

The `src/` folder is empty. The experiment scripts don't exist. You must create everything from MASTER_PLAN.md.

---

### Step 1: Commit pre-registration (CRITICAL - prevents p-hacking claims)
```bash
cd /Users/yuhaoli/code/MAS_For_Finance/Emergent-Applications/apps/trading
git add experiments/pre_registration.json
git commit -m "PRE-REGISTRATION: SI hypothesis - $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push origin main
```
✅ `pre_registration.json` EXISTS - this step can be done now!

---

### Step 2: Create directory structure
```bash
mkdir -p src/{data,agents,competition,analysis,backtest,utils}
mkdir -p tests
mkdir -p results/si_correlations
mkdir -p data/{crypto,forex,stocks,commodities}
```

### Step 3: Create `__init__.py` files
```bash
touch src/__init__.py
touch src/data/__init__.py
touch src/agents/__init__.py
touch src/competition/__init__.py
touch src/analysis/__init__.py
touch src/backtest/__init__.py
touch src/utils/__init__.py
```

### Step 4: Create `requirements.txt`
Copy from MASTER_PLAN.md Step 1.3:
```
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1
... (see MASTER_PLAN.md for full list)
```

### Step 5: Extract code from MASTER_PLAN.md into files
- `src/utils/logging_setup.py` ← Step 1.3b
- `src/utils/safe_math.py` ← Step 1.3b
- `src/utils/timezone.py` ← Step 1.3b
- `src/utils/reproducibility.py` ← Step 1.3b
- `src/utils/checkpointing.py` ← Step 1.3b
- `src/utils/caching.py` ← Step 1.3b
- `experiments/smoke_test.py` ← Step 1.5
- `src/data/loader.py` ← Step 2.1
- `src/data/validation.py` ← Step 2.2
- ... (continue following MASTER_PLAN.md)

### Step 6: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 7: Run smoke test (MUST PASS)
```bash
python experiments/smoke_test.py
```
If smoke test fails → STOP and debug

### Step 8: Download data (BTC, ETH, SOL - ALWAYS multi-asset!)
⚠️ NEVER single-coin backtests

### Step 9: Run data validation
```bash
python experiments/validate_data.py
```
If validation fails → STOP and fix data

---

---

## 📝 Amendments by Previous Session (January 17, 2026)

The following items were **added by the previous conversation window** after reviewing this understanding check:

### Added: Critical Methodology Safeguards Table
- Circular Features Removed
- Data-Driven Block Size
- Effective N Adjustment
- Regime-Conditioned Analysis
- Timezone Standardization
- Data Validation (Phase 2.1)

### Added: Phase Time Budgets
Each phase has a maximum time budget to prevent scope creep.

### Added: Contingency Plans
What to do when things go wrong (API fails, no correlations, etc.)

### Added: Stopping and Exit Criteria
When to halt and how to define success/failure.

### Added: Key Insight #6 - Multi-Coin Only
Critical user preference: NEVER run single-coin backtests.

### Added: Key Insight #7 - MASTER_PLAN is Self-Contained
All code is in MASTER_PLAN.md. Extract directly.

### Added: Key Insight #8 - Three Separate Pipelines
Pipelines must not be mixed due to circularity issues.

### Added: File Structure Explanation
Priority guide for which files to read first.

### Added: Answers to Clarification Questions
Resolved all 5 questions with concrete answers.

---

## 📝 Second Review Amendments (January 17, 2026)

The following items were **corrected/added** during the second thorough check:

### Corrected: Feature Count
- Changed "70+ features" to accurate **57 features** (46 discovery + 2 prediction + 9 dynamics)
- Note: Circular features were removed after FEATURE_PIPELINE_AUDIT

### Corrected: Expert Panel Details
- Added table showing **3 separate expert panels** at different stages
- Hypothesis brainstorming: 14 experts
- Methodology audit: 8 experts with 18 recommendations
- MASTER_PLAN review: 12 experts with 17 issues

### Added: Detailed Feature Categories
- Added table with feature counts per category (Market: 15, Agent: 6, Risk: 10, etc.)
- Listed which features were removed as circular

### Added: Future Folder Decision Framework
- Added the decision tree from `future/README.md`
- When to proceed to each enhancement based on SI results

### Added: Full docs/ File List
- Listed all 15 documentation files for reference

---

## 📝 Third Review Amendments (January 17, 2026)

The following **critical issues** were discovered during the third thorough check:

### 🚨 CRITICAL: No Code Exists Yet!

- **`src/` is COMPLETELY EMPTY** - zero files
- **`experiments/phase0/`, `phase1/`, `phase2/` are ALL EMPTY** - zero files
- Only `experiments/pre_registration.json` exists
- **The Quick Reference commands will fail** until code is extracted from MASTER_PLAN.md

### Added: Detailed "What's NOT Built Yet" Section

- Listed ALL files that need to be created
- Emphasized the order of creation
- Added warning that commands won't work yet

### Added: README.md Discrepancies Warning

- README says "70+ features" but should be 57
- README says "≥5 after FDR" but pre_registration.json says ≥3
- README references scripts that don't exist

### Key Takeaway for New Agent

**Phase 1 is NOT just "commit pre-registration"**. It also requires:
1. Commit pre_registration.json ✅ (exists)
2. Create `src/` directory structure ❌ (empty)
3. Create `requirements.txt` ❌ (doesn't exist)
4. Extract ALL code from MASTER_PLAN.md into files ❌ (nothing done)
5. Run smoke test ❌ (script doesn't exist)

---

## 📝 Fourth Review Amendments (January 17, 2026)

The following items were **added** during the fourth thorough check:

### Added: Primary SI Variant
- Specified `si_rolling_7d` as the primary SI variant from `pre_registration.json`
- Other variants tested in Pipeline 3

### Added: Pre-Registration Commitments
- Report ALL results (including nulls)
- No post-hoc hypothesis changes
- **Publish regardless of outcome**

### Verification Complete ✅
- All key details from `pre_registration.json` are now captured
- All key details from `MASTER_PLAN.md` are now captured
- The Understanding Check is now complete

---

## ⚠️ Known Limitations (From MASTER_PLAN)

This is **signal testing**, not production trading:

| Limitation | Impact | Acceptable Because |
|------------|--------|-------------------|
| Simple strategies | May not reflect real trading | Fine for SI testing |
| No market impact | Frictionless execution | Not trading yet |
| Equal agent sizing | Ignores conviction-based sizing | Testing SI, not strategies |
| No transaction costs | Overstates returns | Not trading yet |
| Hourly data only | May miss intraday patterns | Standard for signal research |

---

## 📋 Quick Reference Commands

### ⚠️ NOTE: Most scripts below DON'T EXIST YET and must be created from MASTER_PLAN.md!

```bash
# Step 1: Pre-Registration (✅ CAN DO NOW)
cd /Users/yuhaoli/code/MAS_For_Finance/Emergent-Applications/apps/trading
git add experiments/pre_registration.json
git commit -m "PRE-REGISTRATION: SI hypothesis"
git push origin main

# Step 2-5: Create files from MASTER_PLAN.md (❌ MUST CREATE FIRST)
# See "Immediate Next Steps" above for detailed instructions

# After files are created:
# Phase 2: Data Validation (CRITICAL - run first!)
python experiments/validate_data.py

# Smoke test
python experiments/smoke_test.py

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

### Script Existence Status

| Script | Exists? | Create From |
|--------|---------|-------------|
| `pre_registration.json` | ✅ YES | - |
| `smoke_test.py` | ❌ NO | MASTER_PLAN Step 1.5 |
| `validate_data.py` | ❌ NO | MASTER_PLAN Step 2.2 |
| `run_backtest.py` | ❌ NO | MASTER_PLAN Step 3 |
| `run_discovery.py` | ❌ NO | MASTER_PLAN Step 4 |
| `run_prediction.py` | ❌ NO | MASTER_PLAN Step 5 |
| `run_dynamics.py` | ❌ NO | MASTER_PLAN Step 6 |
| `run_validation.py` | ❌ NO | MASTER_PLAN Step 7 |
| `run_audits.py` | ❌ NO | MASTER_PLAN Step 7 |
| `generate_report.py` | ❌ NO | MASTER_PLAN Step 8 |
| `run_cross_market.py` | ❌ NO | MASTER_PLAN Step 9 |

---

**End of Understanding Check**

*Reviewed and amended by previous session. Ready for execution.*
