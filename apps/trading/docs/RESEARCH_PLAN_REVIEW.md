# Research Plan Review

**Date:** January 17, 2026  
**Status:** Evaluating proposed 3-phase plan

---

## Current Status (What We've Done)

| Task | Status | Finding |
|------|--------|---------|
| SI-ADX correlation | ✅ Done | r=+0.137, 9/9 consistent |
| SI-Volatility correlation | ✅ Done | r=-0.153, 9/9 consistent |
| SI-RSI correlation | ✅ Done | r=+0.152, 7/9 consistent |
| Granger causality | ✅ Done | SI→ADX 100% |
| Lead-lag (SI vs features) | ✅ Done | SI leads ADX |
| Trading strategy | ✅ Done | 4-5/9 validated |
| Block bootstrap CIs | ✅ Done | Autocorrelation-robust |
| Stationarity tests | ✅ Done | 9/9 stationary |

---

## Phase 1: Deep Correlation Exploration

| Task | Status | Priority | Comment |
|------|--------|----------|---------|
| **1.1 All pairwise correlations** | ⚠️ Partial | HIGH | Only tested 3 features (ADX, Vol, RSI). Should expand to 20+ features |
| **1.2 Lead-lag for all features** | ⚠️ Partial | HIGH | Only Granger-tested 3 features. Need systematic lead-lag matrix |
| **1.3 Conditional correlations** | ❌ Not done | MEDIUM | Does SI-ADX change in different regimes? Critical for understanding |
| **1.4 Stability analysis** | ❌ Not done | HIGH | Are correlations stable over time? Rolling window analysis needed |

**Verdict:** Phase 1 is 30% complete. High-value additions remain.

---

## Phase 2: Mathematical Formalization

| Task | Status | Priority | Comment |
|------|--------|----------|---------|
| **2.1 Prove SI-ADX entropy equivalence** | ❌ Not done | **CRITICAL** | This is the NOVEL CONTRIBUTION. Must formalize. |
| **2.2 Derive SI-RSI relationship** | ❌ Not done | MEDIUM | Why does SI correlate with RSI? |
| **2.3 Generalized SI formula** | ❌ Not done | LOW | Can wait |
| **2.4 SI-Enhanced ADX** | ❌ Not done | LOW | Application, not core research |

**Verdict:** Phase 2 is 0% complete. Task 2.1 is the KEY DIFFERENTIATOR for publication.

---

## Phase 3: Derived Applications

| Task | Status | Priority | Comment |
|------|--------|----------|---------|
| **3.1 Systematically derive from Phase 1-2** | ⏳ Waiting | MEDIUM | Depends on Phase 2 |
| **3.2 Prioritize by theoretical strength** | ⏳ Waiting | MEDIUM | Depends on Phase 2 |
| **3.3 Test rigorously** | ⏳ Waiting | HIGH | Must follow proper validation |

**Verdict:** Phase 3 should wait until Phase 1-2 are stronger.

---

## CRITICAL ADDITIONS NEEDED

### A. Missing from Phase 1

1. **Cross-asset correlation analysis**
   - Does high SI in BTC predict high SI in ETH?
   - Market contagion via SI

2. **Macro feature correlations**
   - SI vs VIX
   - SI vs DXY (dollar index)
   - SI vs Fed Funds rate changes

3. **Higher moments**
   - SI vs skewness
   - SI vs kurtosis (tail risk)

### B. Missing from Phase 2

1. **FORMAL THEOREM (Most Important)**
   ```
   Theorem: Under competitive dynamics with affinity updates,
   SI converges to a function of market entropy H(returns).
   
   Proof sketch:
   - When returns are directional (low entropy), winners specialize
   - Affinity updates: a_k += α(1-a_k) for winning regime k
   - This creates peaked distribution → low agent entropy
   - SI = 1 - mean(agent_entropy) → HIGH
   
   Connection: SI ≈ f(ADX) because:
   - ADX = |+DI - -DI| / (+DI + -DI) measures directional imbalance
   - Imbalance → consistent winners → specialization → high SI
   ```

2. **Regret bound connection**
   - Affinity updates resemble Hedge algorithm
   - Should prove O(√T) regret bound

### C. Missing from Phase 3

1. **SI as factor timing signal**
   - When SI high → increase momentum exposure
   - When SI low → reduce momentum exposure

2. **SI-based volatility forecasting**
   - SI negatively correlates with volatility
   - Can SI predict volatility spikes?

3. **SI-based regime detection**
   - Use SI for HMM state inference
   - Compare to traditional regime detection

---

## RECOMMENDED EXECUTION ORDER

### Priority 1: Complete Phase 1 (1-2 days)
1. ✅ **1.1** Expand features to 15-20 (momentum, mean-rev, volume, macro proxies)
2. ✅ **1.2** Systematic lead-lag matrix for all features
3. ✅ **1.3** Regime-conditional correlations
4. ✅ **1.4** Rolling 1-year stability analysis

### Priority 2: Phase 2.1 - THE THEOREM (1 day)
- This is the PUBLICATION DIFFERENTIATOR
- Formal proof of SI-ADX entropy equivalence
- Connect to information theory and game theory

### Priority 3: Derived Applications (1 day)
- Only pursue applications with theoretical backing
- SI for factor timing (has theoretical justification)
- SI for volatility forecasting (has correlation evidence)

---

## FINAL RECOMMENDATION

**Proceed with Phase 1 expansion first.**

Reason: We have strong empirical foundation but:
1. Only 3 features tested - need 15-20 for completeness
2. No regime-conditional analysis yet
3. No stability analysis (are correlations time-varying?)

After Phase 1 is complete, Phase 2.1 (the theorem) becomes the priority.

---

## Estimated Timeline

| Phase | Time | Deliverable |
|-------|------|-------------|
| Phase 1 expansion | 1-2 days | Comprehensive correlation matrix |
| Phase 2.1 theorem | 1 day | Formal SI-ADX theorem |
| Phase 3 applications | 1 day | 2-3 validated applications |
| **Total** | **3-4 days** | Publication-ready package |
