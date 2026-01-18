# SI Deep Exploration Plan

## Goal
Understand **WHY** SI correlates with market features, not just **THAT** it correlates.

---

## Part 1: Mathematical Dissection

### 1.1 SI Formula

```
SI = 1 - H̄

where H̄ = (1/N) Σᵢ Hᵢ  (mean normalized entropy across agents)

and Hᵢ = -Σₖ pᵢₖ log(pᵢₖ) / log(K)  (normalized entropy for agent i)

pᵢₖ = niche_affinity of agent i for regime k
K = 3 (number of regimes: trending, mean-reverting, volatile)
```

**Key insight**: SI measures how "peaked" the affinity distributions are.
- High SI → agents have concentrated affinities (specialists)
- Low SI → agents have uniform affinities (generalists)

---

### 1.2 Correlated Features - Their Formulas

#### ADX (Average Directional Index)

```
ADX = SMA₁₄(DX)

where DX = |+DI - -DI| / (+DI + -DI) × 100

+DI = SMA₁₄(+DM) / ATR₁₄ × 100
-DI = SMA₁₄(-DM) / ATR₁₄ × 100

+DM = max(High_t - High_{t-1}, 0)  if > -DM, else 0
-DM = max(Low_{t-1} - Low_t, 0)    if > +DM, else 0
```

**Structure**: ADX measures the **ratio of directional movement to total range**.

#### RSI (Relative Strength Index)

```
RSI = 100 - 100/(1 + RS)

where RS = SMA₁₄(gains) / SMA₁₄(losses)

gains = max(close_t - close_{t-1}, 0)
losses = max(close_{t-1} - close_t, 0)
```

**Structure**: RSI measures the **ratio of up-moves to down-moves**.

#### Bollinger Band Width

```
BB_Width = (Upper - Lower) / Middle

where Upper = SMA₂₀ + 2σ₂₀
      Lower = SMA₂₀ - 2σ₂₀
      Middle = SMA₂₀

Simplified: BB_Width = 4σ₂₀ / SMA₂₀ = 4 × CV (coefficient of variation)
```

**Structure**: BB Width measures **normalized volatility**.

#### Volatility

```
σ = √(Σ(rₜ - r̄)² / (n-1))

where rₜ = log(close_t / close_{t-1})
```

**Structure**: Standard deviation of returns.

---

### 1.3 Mathematical Connection Hypothesis

#### Observation: All correlated features measure "clarity" or "structure"

| Feature | What It Measures | Structural Similarity to SI |
|---------|------------------|------------------------------|
| ADX | Trend clarity (directional consistency) | SI measures affinity clarity |
| RSI | Momentum consistency | SI measures winner consistency |
| BB Width | Volatility structure | SI measures regime structure |
| Volatility | Return dispersion | SI measures affinity dispersion |

#### Hypothesis: SI and ADX Share Entropy-Like Structure

**ADX decomposition**:
```
DX = |+DI - -DI| / (+DI + -DI)

This is structurally similar to:

DX ≈ |p₊ - p₋| / (p₊ + p₋) = |p₊ - p₋|  (if normalized)

where p₊ = probability of up-move, p₋ = probability of down-move
```

**SI decomposition**:
```
SI = 1 - H̄ 

For binary case (K=2): H = -p log p - (1-p) log(1-p)

When p is extreme (0.9 or 0.1): H is LOW, SI is HIGH
When p is balanced (0.5): H is HIGH, SI is LOW
```

**Connection**: Both ADX and SI are HIGH when there's **asymmetry/imbalance**.

---

### 1.4 Formal Mathematical Relationship

#### Theorem (Proposed): SI-ADX Structural Equivalence

Let:
- p₊ = proportion of "up" regimes won by an agent
- p₋ = proportion of "down" regimes won

Then:
```
SI ∝ 1 - H(p₊, p₋, p₀)  (entropy of regime wins)

ADX ∝ |p₊ - p₋|  (imbalance of directional moves)
```

**Claim**: Under stationary markets, high |p₊ - p₋| implies low H, thus high SI.

**Proof sketch**:
1. When market trends strongly (high ADX), one direction dominates
2. Agents winning in that direction update affinities toward that regime
3. This creates peaked affinity distributions (low entropy)
4. Therefore SI is high

**This explains the positive SI-ADX correlation mathematically, not just empirically.**

---

## Part 2: Lead-Lag Analysis

### 2.1 Questions to Answer

| Question | Method | Expected Insight |
|----------|--------|------------------|
| Does SI lead ADX? | Cross-correlation at lags | SI as leading indicator |
| Does ADX lead SI? | Cross-correlation at lags | SI as lagging indicator |
| What's the optimal lag? | Maximize correlation | Prediction horizon |
| Is relationship symmetric? | Compare lead vs lag | Causality hints |

### 2.2 Implementation

```python
def lead_lag_analysis(si, feature, max_lag=30):
    """
    Compute cross-correlation at various lags.
    Positive lag: SI leads feature
    Negative lag: feature leads SI
    """
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corr = si[:-lag].corr(feature[lag:])
        elif lag < 0:
            corr = si[-lag:].corr(feature[:lag])
        else:
            corr = si.corr(feature)
        correlations.append({'lag': lag, 'correlation': corr})
    return pd.DataFrame(correlations)
```

---

## Part 3: Interaction Effects

### 3.1 Two-Way Interactions

| Interaction | Question |
|-------------|----------|
| SI × ADX | Does high SI + high ADX predict returns better? |
| SI × Volatility | Does SI work differently in high vs low vol? |
| SI × RSI | Can SI confirm/deny RSI signals? |

### 3.2 Conditional Analysis

```python
def conditional_returns(si, adx, returns):
    """
    Analyze returns in each SI-ADX quadrant.
    """
    si_high = si > si.median()
    adx_high = adx > adx.median()
    
    quadrants = {
        'high_SI_high_ADX': returns[si_high & adx_high].mean(),
        'high_SI_low_ADX': returns[si_high & ~adx_high].mean(),
        'low_SI_high_ADX': returns[~si_high & adx_high].mean(),
        'low_SI_low_ADX': returns[~si_high & ~adx_high].mean(),
    }
    return quadrants
```

**Expected finding**: High SI + High ADX should have best returns.

---

## Part 4: Threshold Analysis

### 4.1 Non-Linear Effects

| Question | Method |
|----------|--------|
| At what SI level do correlations strengthen? | Quantile analysis |
| Is there a "critical" SI threshold? | Breakpoint detection |
| Do extreme SI values have different behavior? | Tail analysis |

### 4.2 Implementation

```python
def threshold_analysis(si, feature, returns, n_quantiles=5):
    """
    Analyze relationship at different SI levels.
    """
    si_quantiles = pd.qcut(si, n_quantiles, labels=False)
    
    results = []
    for q in range(n_quantiles):
        mask = si_quantiles == q
        corr = si[mask].corr(feature[mask])
        ret = returns[mask].mean()
        results.append({
            'quantile': q,
            'si_range': f'{si[mask].min():.2f}-{si[mask].max():.2f}',
            'correlation': corr,
            'mean_return': ret,
        })
    return pd.DataFrame(results)
```

---

## Part 5: Derived Applications

Based on exploration, derive new applications:

### 5.1 Application Derivation Map

| Exploration Finding | Derived Application |
|---------------------|---------------------|
| SI leads volatility by 3 days | → Volatility forecasting |
| High SI + High ADX = best returns | → Combined signal |
| SI drops before drawdowns | → Drawdown early warning |
| SI threshold at 0.7 matters | → Binary regime indicator |
| SI stable → regime stable | → Regime persistence signal |

### 5.2 Priority Applications to Test

| Priority | Application | Based On |
|----------|-------------|----------|
| 1 | **Vol forecasting** | SI-volatility lead-lag |
| 2 | **SI × ADX signal** | Interaction effects |
| 3 | **Drawdown warning** | SI threshold analysis |
| 4 | **Regime persistence** | SI stability analysis |
| 5 | **Cross-asset SI** | Multi-market SI divergence |
| 6 | **SI momentum** | SI change rate |
| 7 | **SI mean reversion** | SI extremes |
| 8 | **Portfolio SI** | Aggregate SI across assets |

---

## Part 6: Implementation Plan

### Phase 1: Mathematical Analysis (2 hours)
1. Derive formal connection between SI and ADX entropy structures
2. Write proofs/derivations in LaTeX
3. Identify other structural connections (RSI, BB Width)

### Phase 2: Lead-Lag Analysis (1 hour)
1. Compute cross-correlations for all 31 assets
2. Identify which features SI leads vs lags
3. Determine optimal prediction horizons

### Phase 3: Interaction Effects (1 hour)
1. Build SI × feature quadrant analysis
2. Test 2-way and 3-way interactions
3. Identify strongest combinations

### Phase 4: Threshold Analysis (1 hour)
1. Quantile-based correlation analysis
2. Breakpoint detection for SI thresholds
3. Tail behavior analysis

### Phase 5: Application Development (2 hours)
1. Implement top 3 derived applications
2. Backtest each application
3. Compare to baseline methods

### Phase 6: Documentation (1 hour)
1. Document all findings
2. Update thesis with mathematical insights
3. Create visualizations

**Total estimated time: 8 hours**

---

## Expected Outcomes

### If Successful:
1. **Mathematical proof** that SI and ADX share entropy structure
2. **Lead-lag relationships** for prediction
3. **Optimal thresholds** for trading signals
4. **3-5 new applications** derived from first principles
5. **Deeper thesis**: "SI is a behavioral entropy measure that captures market structure"

### If Partially Successful:
1. Better understanding of SI limitations
2. Clearer positioning of where SI works/doesn't work
3. Honest documentation of exploration

---

## New Thesis (If Exploration Succeeds)

> **"The Specialization Index (SI) is a behavioral entropy measure that emerges from agent competition. We prove mathematically that SI shares structural properties with established technical indicators (ADX, RSI), explaining their empirical correlation. This understanding enables novel applications: volatility forecasting (SI leads vol by 3 days), drawdown prediction (SI drops precede crashes), and optimized factor timing (High SI × High ADX quadrant yields 2x returns)."**

This is much stronger than our current thesis because it explains **WHY**, not just **WHAT**.
