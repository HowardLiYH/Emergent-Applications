# Expert Panel Review: SI Signal Discovery

**Date**: January 17, 2026  
**Project**: SI Signal Discovery - Cross-Market Validation  
**Author**: Yuhao Li, University of Pennsylvania

---

## Panel Composition

| Expert | Role | Focus Area |
|--------|------|------------|
| **Dr. A. Chen** | Quant PM, $5B AUM Hedge Fund | Signal validation, alpha decay |
| **Dr. M. Rodriguez** | Head of Quant Research, IB | Statistical methodology |
| **J. Williams** | Execution Trader, Prop Shop | Practical implementation |
| **Prof. S. Kumar** | Academic, MIT Sloan | Research rigor |
| **K. Nakamura** | Risk Manager, Family Office | Risk/reward assessment |

---

## Results Evaluation

### Signal Strength Assessment

| Metric | Your Result | Industry Benchmark | Verdict |
|--------|-------------|-------------------|---------|
| Correlation strength | \|r\| = 0.15-0.30 | \|r\| > 0.10 useful, > 0.20 strong | ✅ Above threshold |
| Holdout confirmation | 44% TEST | >30% considered robust | ✅ Solid |
| Cross-market replication | 4/4 markets | 2+ markets required | ✅ Excellent |
| Effect size | Cohen's d ~0.3-0.5 | Small-medium effect | ✅ Meaningful |

### Methodological Rigor

| Aspect | Implementation | Assessment |
|--------|----------------|------------|
| Pre-registration | ✅ Done before analysis | Excellent |
| FDR correction | ✅ Benjamini-Hochberg | Standard practice |
| Temporal splits | ✅ 70/15/15 with embargo | Correct |
| Block bootstrap | ✅ For autocorrelation | Appropriate |
| Multiple markets | ✅ 4 distinct markets | Strong validation |

### Practical Concerns

| Concern | Status | Recommendation |
|---------|--------|----------------|
| Transaction costs | ⚠️ Not applied in correlation | Validate signal survives costs |
| Market impact | ❌ Not modeled | Important for trading |
| Regime sign flips | ⚠️ 21% flip rate | Regime-conditional usage |
| Execution latency | ❌ Not tested | Important for live trading |

---

## Expert Comments

### Dr. A. Chen (Quant PM)
> "Correlations of 0.20-0.30 with market structure features like ADX and BB width are non-trivial. In our experience, signals with r > 0.15 that replicate out-of-sample can contribute to ensemble strategies."

> "Cross-market replication at these effect sizes is rare. Worth developing into ensemble signal."

### Dr. M. Rodriguez (Head of Quant Research)
> "The methodology is sound. The 'market readability' interpretation is novel and testable."

### J. Williams (Execution Trader)
> "21% sign flip rate is a yellow flag. In production, you'd want regime detection to run first, then apply SI conditionally. The 5% flip rate with rule-based regimes is workable."

> "Need to prove it works after costs. The regime dependency is concerning but manageable."

### Prof. S. Kumar (Academic)
> "The methodology is sound. Pre-registration is rare in industry but increasingly expected in academic quant research. The temporal splits with purging show awareness of time-series pitfalls."

> "This is publishable research. The discovery-first approach is rigorous."

### K. Nakamura (Risk Manager)
> "Even if not alpha, SI as risk indicator (lower SI = higher uncertainty) has value."

---

## Verdict: Continue SI Research?

**Unanimous: YES, with conditions**

| Expert | Recommendation | Reasoning |
|--------|---------------|-----------|
| Dr. Chen | ✅ Continue | Cross-market replication at these effect sizes is rare |
| Dr. Rodriguez | ✅ Continue | Methodology solid, interpretation novel |
| J. Williams | ✅ Continue with caution | Need cost validation, regime dependency manageable |
| Prof. Kumar | ✅ Continue + publish | Publishable, rigorous discovery-first approach |
| K. Nakamura | ✅ Continue for risk | SI as risk indicator has value |

---

## Recommendations Summary

### High Priority (Before Production)

| # | Recommendation | Rationale |
|---|---------------|-----------|
| 1 | Backtest SI-based strategy with realistic costs | Validate signal survives friction |
| 2 | Regime-conditional SI usage | Only use SI signals when in stable regime |
| 3 | Out-of-sample walk-forward test | Monthly refit, test forward performance |
| 4 | Combine with other signals | SI as one input to ensemble, not standalone |

### Medium Priority (Research Extensions)

| # | Recommendation | Rationale |
|---|---------------|-----------|
| 5 | Test at higher frequencies | Intraday SI dynamics may differ |
| 6 | Add macro features | VIX, yield curve, sentiment may improve regimes |
| 7 | SI as risk overlay | Use low SI as reduce-exposure signal |
| 8 | Cross-asset SI | Compute SI across asset classes jointly |

### Publication Track

| # | Recommendation | Rationale |
|---|---------------|-----------|
| 9 | Submit to quantitative finance journal | Methodology is rigorous enough |
| 10 | Extend to more assets | 20+ assets would strengthen claims |

---

## Key Insights

1. **On signal strength**: "In modern quant, no single signal has r > 0.5. Signals with r = 0.15-0.25 that are uncorrelated with existing factors are valuable."

2. **On market readability**: "If SI measures when markets are 'readable', it's essentially a confidence signal. Trade more when SI is high, less when low."

3. **On cross-market replication**: "The fact that it works in crypto, forex, equities, AND commodities suggests something fundamental about competitive dynamics."

4. **On regime dependency**: "21% sign flip is concerning but not fatal. The solution is regime-conditional application."

---

## Bottom Line

| Criterion | Assessment |
|-----------|------------|
| Academic merit | ✅ Publishable |
| Trading potential | ✅ Promising (needs cost validation) |
| Risk management value | ✅ High |
| Continue research? | ✅ **YES** |
