# Emergent Trading Specialists

> 🎯 **Applying NichePopulation specialization to financial trading**

## Overview

This application extends the NichePopulation mechanism (Paper 1) to financial trading. The core hypothesis is that competitive specialization among trading agents can produce better risk-adjusted returns than individual strategies or naive ensembles.

## Key Insight

**Profit must be validated first, before testing specialization.**

The validation chain:
1. ✅ Individual strategies are profitable
2. ✅ Combining strategies improves Sharpe ratio
3. ✅ NichePopulation mechanism beats equal-weight ensemble
4. ✅ Specialization Index (SI) correlates with profit

If step 1 fails, stop. No amount of specialization can make unprofitable strategies profitable.

## Quick Start

```bash
# Phase 0a: Test individual strategies
python experiments/phase0/test_individual_strategies.py

# Phase 0b: Test ensemble
python experiments/phase0/test_ensemble.py

# Phase 1: Test NichePopulation
python experiments/phase1/test_niche_population.py
```

## Research Plan

See [docs/TRADING_RESEARCH_PLAN.md](docs/TRADING_RESEARCH_PLAN.md) for the complete research plan including:
- Expert panel analysis
- Modern alternatives to Thompson Sampling
- Phased implementation plan
- Success metrics and go/no-go criteria

## Folder Structure

```
apps/trading/
├── docs/           # Research plans and analysis
├── experiments/    # Phased experiments
│   ├── phase0/    # Individual strategy testing
│   ├── phase1/    # NichePopulation testing
│   └── phase2/    # SI validation
└── src/           # Core implementation
```

## Success Metrics

| Metric | Target | Priority |
|--------|--------|----------|
| Net Return | > 0% | 1 (Primary) |
| Sharpe Ratio | > 0.5 | 2 |
| Max Drawdown | < 25% | 3 |
| SI | > 0.4 | 4 (Secondary) |

## Status

- [x] Research plan documented
- [ ] Phase 0a: Individual strategy testing
- [ ] Phase 0b: Ensemble testing
- [ ] Phase 1: NichePopulation mechanism
- [ ] Phase 2: SI validation

## Related Papers

- [Paper 1: NichePopulation](https://github.com/HowardLiYH/NichePopulation) - Foundational specialization mechanism
- [Paper 2: Emergent Preference Specialization](https://github.com/HowardLiYH/Emergent-Preference-Specialization-in-LLM-Agent-Populations) - LLM agent specialization
- [Paper 3: Emergent Tool Specialization](https://github.com/HowardLiYH/Emergent-Tool-Specialization) - Real tool specialization
