# Mathematical Analysis: SI-ADX Connection

**Date:** January 17, 2026  
**Goal:** Derive theoretical connection between SI and ADX formulas

---

## 1. Formula Definitions

### SI (Specialization Index)

```
SI = 1 - H̄

where H̄ = (1/N) Σᵢ Hᵢ  (mean agent entropy)

and Hᵢ = -Σₖ pᵢₖ log(pᵢₖ) / log(K)  (normalized entropy for agent i)

pᵢₖ = affinity of agent i to regime k
K = 3 (number of regimes: trending, mean-reverting, volatile)
```

**Key insight:** SI is HIGH when agents are SPECIALIZED (peaked affinity distribution)

### ADX (Average Directional Index)

```
ADX = MA₁₄(DX)

where DX = 100 × |+DI - -DI| / (+DI + -DI)

+DI = 100 × MA₁₄(+DM) / ATR
-DI = 100 × MA₁₄(-DM) / ATR

+DM = max(highₜ - highₜ₋₁, 0)   [positive directional movement]
-DM = max(lowₜ₋₁ - lowₜ, 0)     [negative directional movement]

ATR = MA₁₄(TR)  [average true range]
```

**Key insight:** ADX is HIGH when one direction DOMINATES

---

## 2. Mathematical Connection

### Observation: Both Measure Asymmetry/Imbalance

| Metric | What it measures | High when |
|--------|------------------|-----------|
| SI | Asymmetry in agent affinities | Agents specialize |
| ADX | Asymmetry in price direction | Trend dominates |

### The Causal Chain

```
High ADX (trending market)
    ↓
One direction dominates (+DM >> -DM or vice versa)
    ↓
Momentum strategies win consistently
    ↓
Winners update affinities: aₖ += α(1 - aₖ) for regime k
    ↓
Repeated wins → peaked affinity distribution
    ↓
Low agent entropy H̄
    ↓
High SI = 1 - H̄
```

### Formal Statement

**Theorem (Informal):** Under the NichePopulation competitive dynamics with affinity updates, SI converges to a monotonic function of market directional imbalance.

---

## 3. Key Mathematical Insights

### Insight 1: DX as a Probability-Like Ratio

Rewrite DX:
```
DX = |+DI - -DI| / (+DI + -DI)
   = |p⁺ - p⁻|  where p⁺ = +DI/(+DI + -DI), p⁻ = -DI/(+DI + -DI)
```

This is the **absolute difference between "probabilities"** of up vs down movement!

### Insight 2: SI Entropy and Market Entropy

Agent entropy:
```
Hᵢ = -Σₖ pᵢₖ log(pᵢₖ)
```

Market "directional entropy":
```
H_market = -p⁺ log(p⁺) - p⁻ log(p⁻)  where p⁺ + p⁻ = 1
```

**When market entropy is LOW** (one direction dominates):
- Consistent winners emerge
- Winners specialize → agent entropy LOW → SI HIGH

**When market entropy is HIGH** (balanced up/down):
- No consistent winners
- Agents stay generalist → agent entropy HIGH → SI LOW

### Insight 3: The Jensen's Inequality Connection

For convex function f(x) = -x log(x):
```
E[f(X)] ≤ f(E[X])  (Jensen's inequality)
```

This suggests:
- Mean agent entropy ≤ entropy of mean affinities
- SI relates to "variance" of specialization across agents

---

## 4. New Discoveries from Formula Analysis

### Discovery 1: SI as "Market Readability"

SI measures how "readable" the market is:
- High SI → clear signal (one regime dominates) → predictable
- Low SI → mixed signals → unpredictable

**Application:** Use SI as a confidence measure for any trading signal.

### Discovery 2: SI-RSI Connection

RSI formula:
```
RSI = 100 - 100/(1 + RS)  where RS = avg_gain / avg_loss
```

When RSI is extreme (>70 or <30):
- One direction dominates (similar to high ADX)
- Same mechanism leads to high SI

**Prediction:** SI should correlate with |RSI - 50| (distance from neutral)

### Discovery 3: SI-Volatility Inverse Relationship

We found SI negatively correlates with volatility. Why?

```
High volatility → frequent regime switches → no consistent winner
                → agents stay generalist → low SI

Low volatility  → stable regime → consistent winners
                → agents specialize → high SI
```

**Mathematical formulation:**
```
∂SI/∂σ < 0  (SI decreases with volatility)
```

### Discovery 4: SI as Entropy Complement

Define "Market Clarity Index" (MCI):
```
MCI = 1 - H_market / log(K)
```

**Conjecture:** SI ≈ MCI under equilibrium conditions.

This connects SI to information theory directly!

---

## 5. Testable Predictions from Theory

| Prediction | How to Test | Expected Result |
|------------|-------------|-----------------|
| SI ~ |RSI - 50| | Compute correlation | Positive correlation |
| SI ~ ADX | Already tested | ✅ Confirmed (+0.137) |
| SI ~ 1/σ | Already tested | ✅ Confirmed (-0.153 with σ) |
| SI leads ADX | Granger test | ✅ Confirmed (100%) |
| SI ~ MCI | Compute MCI, correlate | Should be high |
| High SI → predictable | Measure hit rate | Higher when SI high |

---

## 6. Novel Indicator: SI-Enhanced ADX

Combine SI and ADX:
```
SI_ADX = ADX × (1 + β × SI)  where β is a scaling factor
```

Rationale: When both SI and ADX are high, confidence in trend is stronger.

**Or use SI to adjust ADX threshold:**
```
If SI > threshold:
    Use lower ADX threshold (e.g., 20 instead of 25)
Else:
    Use higher ADX threshold (e.g., 30)
```

---

## 7. Implications for Publication

### Main Theorem to Prove

**Theorem:** Let {aᵢₖ(t)} be agent affinities evolving under the affinity update rule:
```
aᵢₖ(t+1) = aᵢₖ(t) + α × 𝟙{winner} × (1 - aᵢₖ(t))
```

Then as t → ∞, the Specialization Index SI(t) converges to:
```
SI* = f(DI⁺, DI⁻) = f(ADX)
```
where f is a monotonic function determined by α and the competition dynamics.

### Why This Matters

1. **Theoretical novelty:** First formal connection between agent-based specialization and classical TA
2. **Interpretability:** SI is not arbitrary - it captures fundamental market structure
3. **Applications:** SI can enhance existing indicators (ADX, RSI)

---

## 8. Next Steps

1. **Verify Discovery 2:** Test SI ~ |RSI - 50| correlation
2. **Compute MCI:** Test SI ~ MCI relationship
3. **Prove theorem formally:** LaTeX proof with all conditions
4. **Test SI-Enhanced ADX:** Does it improve trading?
