# Architecture Decisions: Trading Application

**Date**: January 17, 2026
**Status**: Research Phase

---

## Decision 1: Thompson Sampling vs Modern Alternatives

### Background

Thompson Sampling (TS) was used in Paper 1 (NichePopulation). The question is whether TS is still appropriate for trading or if modern alternatives should be used.

### Research Findings

| Architecture | Maturity | Complexity | Interpretability | Data Needs | Best For |
|--------------|----------|------------|------------------|------------|----------|
| **Thompson Sampling** | High | Low | High | Low | Simple bandits, clear rewards |
| **UCB (Upper Confidence Bound)** | High | Low | High | Low | When exploration bonus matters |
| **Neural Contextual Bandits** | Medium | Medium | Medium | Medium | Complex state, non-linear rewards |
| **Soft Actor-Critic (SAC)** | Medium | High | Low | High | Continuous actions, complex environments |
| **Decision Transformer** | Low | High | Medium | Very High | Offline RL, sequence prediction |
| **Mixture of Experts (MoE)** | High | Medium | High | Medium | Natural specialization |

### Decision

**Phase 0-2**: Use Thompson Sampling
- Simple, interpretable, proven in Paper 1
- Validate that the mechanism works before adding complexity

**Phase 3+**: Consider upgrading to:
- Neural Contextual Bandits (if state is complex)
- MoE with learned gating (if specialization works)

### Rationale

1. **Don't optimize before validating**: If basic TS doesn't produce profitable specialization, advanced architectures won't help
2. **Interpretability matters**: We need to understand WHY agents specialize
3. **Data efficiency**: Trading data is limited; complex architectures overfit easily

---

## Decision 2: Regime Detection Approach

### Background

Market regimes (trending, volatile, calm) are correlated and transition between each other. How should we model this?

### Options Considered

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Hard labels** | Discrete regime assignment | Simple | Ignores transitions, overlap |
| **Soft assignments** | Probability distribution over regimes | Captures uncertainty | More complex |
| **Hidden Markov Model** | Latent states with transitions | Models transitions | Assumes stationarity |
| **DCC-GARCH** | Dynamic correlations + volatility | Captures correlation changes | Complex to implement |
| **No explicit regimes** | Let agents discover regimes | Most flexible | May not find anything |

### Decision

**Phase 0-1**: No explicit regime detection
- Let specialization emerge naturally
- Agents compete on raw returns/Sharpe

**Phase 2+**: Add soft regime features
- Compute volatility, correlation, trend indicators
- Include as context for agents
- Use soft assignments (probabilities)

### Rationale

1. **The thesis is about EMERGENT specialization**: Pre-defining regimes defeats the purpose
2. **Start simple**: Prove the mechanism works without regime labels
3. **Add regimes as features, not labels**: Let agents learn what matters

---

## Decision 3: Competition Mechanism

### Background

How should agents compete? What determines "winning"?

### Options Considered

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Winner-take-all** | Best agent wins entire period | Strong selection pressure | May converge too fast |
| **Proportional** | Reward proportional to performance | Balanced | Weak selection |
| **Fitness sharing** | Penalize crowded niches | Promotes diversity | May hurt best performers |
| **Multi-objective** | Optimize profit AND diversity | Balances goals | Complex |

### Decision

**Phase 0-2**: Winner-take-all (same as Paper 1)
- Consistent with proven mechanism
- Strong selection pressure for specialization

**If specialization fails**: Try fitness sharing or proportional

### Rationale

1. **Paper 1 validated winner-take-all**: It worked for time series prediction
2. **Start with what works**: Only change if results are poor

---

## Decision 4: Fitness Function

### Background

What should agents optimize? This is crucial—wrong fitness = wrong specialization.

### Options Considered

| Fitness | Definition | Pros | Cons |
|---------|------------|------|------|
| **Raw return** | Total profit | Simple, direct | Ignores risk |
| **Sharpe ratio** | Return / volatility | Risk-adjusted | Can be gamed with low vol |
| **Sortino ratio** | Return / downside vol | Penalizes losses more | More complex |
| **Calmar ratio** | Return / max drawdown | Protects from ruin | Sensitive to single events |
| **Profit factor** | Gross profit / gross loss | Intuitive | Ignores magnitude |

### Decision

**Phase 0-2**: Sharpe ratio (rolling 30-day)
- Standard industry metric
- Balances return and risk
- Well-understood

**Phase 3+**: Consider multi-objective (Sharpe + drawdown constraint)

### Rationale

1. **Industry standard**: Sharpe is widely used, comparable to benchmarks
2. **Risk-adjusted**: Raw returns encourage reckless behavior
3. **Rolling window**: Adapts to changing conditions

---

## Decision 5: Evaluation Period

### Background

How often should we evaluate agents and update specialization?

### Options Considered

| Period | Description | Pros | Cons |
|--------|-------------|------|------|
| **Daily** | Evaluate every day | Fast adaptation | Too noisy |
| **Weekly** | Evaluate every week | Balance speed/noise | May miss fast changes |
| **Monthly** | Evaluate every month | Stable | Slow adaptation |
| **Adaptive** | Based on regime detection | Optimal theoretically | Complex |

### Decision

**Phase 0-2**: Weekly evaluation
- Balance between adaptation speed and noise reduction
- ~50 evaluation periods per year (statistically meaningful)

### Rationale

1. **Weekly is standard**: Many hedge funds rebalance weekly
2. **Enough samples**: 3-year backtest = 150+ evaluation periods
3. **Not too noisy**: Daily would be dominated by randomness

---

## Decision 6: Data and Assets

### Background

What data should we use for Phase 0 testing?

### Decision

**Phase 0**: Free data from Yahoo Finance
- BTC-USD: High volatility, regime variety
- ETH-USD: Correlated but different from BTC
- SPY: Traditional equity, lower volatility

**Why these assets**:
1. **Free and accessible**: No data costs
2. **Diverse**: Crypto + equity
3. **Regime variety**: Have experienced bull, bear, volatile periods
4. **Liquid**: Realistic execution assumptions

**Time period**: 2021-2024 (captures bull, crash, recovery)

---

## Decision 7: Transaction Costs

### Background

How should we model trading costs?

### Decision

**Phase 0-2**: Simple fixed costs
- Crypto: 0.1% per trade (realistic for major exchanges)
- Equity: 0.05% per trade (realistic for retail)

**Phase 3+**: Add slippage model
- Market impact proportional to position size
- Bid-ask spread modeling

### Rationale

1. **Include costs from day 1**: Many strategies look good without costs
2. **Conservative estimates**: Better to underestimate profits
3. **Simple first**: Complex cost models can be added later

---

## Summary of Key Decisions

| Decision | Choice | Phase |
|----------|--------|-------|
| Selection algorithm | Thompson Sampling | 0-2 |
| Regime detection | None (emergent) | 0-2 |
| Competition | Winner-take-all | 0-2 |
| Fitness function | Sharpe ratio (30d) | 0-2 |
| Evaluation period | Weekly | 0-2 |
| Data | BTC, ETH, SPY (Yahoo) | 0 |
| Transaction costs | 0.1% crypto, 0.05% equity | 0-2 |

---

## Future Upgrades (Phase 3+)

If Phase 0-2 succeeds, consider:

1. **Neural Contextual Bandits**: Replace TS with learned reward estimation
2. **Soft regime features**: Add volatility, correlation as context
3. **More assets**: Expand to 20+ liquid assets
4. **Real-time data**: Polygon.io or similar
5. **Paper trading**: Test with real market execution

---

*Last Updated: January 17, 2026*
