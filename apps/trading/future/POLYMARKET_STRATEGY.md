# Future Enhancement: Polymarket Betting Strategy

> ⚠️ **DO NOT START UNTIL**:
> 1. SI testing complete
> 2. Regime prediction works
> 3. You're ready for a separate HFT project

---

## Prerequisites

- [ ] SI testing complete and validated
- [ ] Regime prediction implemented and tested
- [ ] Comfortable with high-frequency trading concepts
- [ ] Ready to invest in infrastructure (VPS, etc.)

---

## The Idea

Apply SI + Regime Prediction to **prediction markets** like Polymarket.

**Specifically**: The "BTC 15-minute UP/DOWN" market.

---

## Why This is a SEPARATE Project

| Current Project | Polymarket Project |
|-----------------|-------------------|
| Signal discovery | Arbitrage execution |
| Python | Rust (performance) |
| Hourly/daily | Sub-second |
| Research focus | Infrastructure focus |
| Low cost | Requires VPS, etc. |

---

## Referenced Strategy Summary

**Source**: @the_smart_ape on X

### Strategy: Two-Leg Arbitrage

```
1. OBSERVE: Monitor price for "crash" (rapid drop)
2. LEG 1: Buy the side that crashed (e.g., DOWN at $0.30)
3. WAIT: Until opposite side is cheap enough
4. LEG 2: Buy opposite side (e.g., UP at $0.65)
5. PROFIT: Paid $0.95 total, guaranteed $1.00 payout
```

### Key Parameters

| Parameter | Description | Conservative | Aggressive |
|-----------|-------------|--------------|------------|
| `sumTarget` | Max combined price | 0.95 | 0.60 |
| `movePct` | Crash threshold | 15% | 1% |
| `windowMin` | Entry window | 2 min | 15 min |

### Backtest Results

- Conservative: **+86% ROI**
- Aggressive: **-50% ROI**

---

## How SI Could Enhance This (Hypothesis)

```
STANDARD ARBITRAGE:
  Wait for crash → Buy → Hedge → Profit

SI-ENHANCED:
  SI predicts volatility → Position BEFORE crash → Better entry

REGIME-ENHANCED:
  HMM predicts regime change → Anticipate crash direction
```

### Potential SI Applications

| SI Signal | Polymarket Action |
|-----------|-------------------|
| High SI + volatile regime | Expect larger swings, wider arbitrage |
| Low SI | Market calm, fewer opportunities |
| SI rising | Volatility coming, prepare for crash |
| Regime transition predicted | Pre-position for direction |

---

## Technical Requirements

### Infrastructure
- [ ] VPS near Polymarket servers
- [ ] Dedicated Polygon RPC node
- [ ] Low-latency networking

### Tech Stack
- [ ] Rust (rewrite from Python/JS)
- [ ] WebSocket streaming
- [ ] Custom data recorder

### Data
- [ ] Real-time Best Bid/Ask
- [ ] Order book depth
- [ ] Tick-level recording (6GB/4 days per their data)

---

## Estimated Effort

| Task | Time |
|------|------|
| Learn Polymarket API | 1 week |
| Build data recorder | 1 week |
| Implement bot (Python) | 2 weeks |
| Rewrite in Rust | 2-4 weeks |
| Infrastructure setup | 1 week |
| Backtesting | 2 weeks |
| Paper trading | 2 weeks |
| Live trading | Ongoing |
| **Total** | **3-6 months** |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Execution latency | High | Rust + VPS |
| Market impact | High | Small position sizes |
| API changes | Medium | Monitor updates |
| Regulatory | Medium | Check jurisdiction |
| Capital loss | High | Paper trade first |

---

## Decision Framework

```
Should you do this?

IF you want HFT experience: YES (but expect 3-6 months)
IF you want quick wins: NO (stick with SI testing)
IF you have infrastructure budget: Helps
IF you're risk-averse: NO (this is high-risk)
```

---

## Alternative: Simpler Polymarket Application

Instead of HFT arbitrage, could apply SI to:

1. **Daily resolution markets** (less HFT, more signal-based)
2. **Longer-term markets** (election, events)
3. **Volatility-based positioning** (if SI predicts vol)

This would be easier and more aligned with current work.

---

## Folder Structure (If Started)

```
apps/
├── trading/           ← Current (SI testing)
└── polymarket/        ← Future (separate project)
    ├── src/
    │   ├── data/      # WebSocket, recorder
    │   ├── strategy/  # Two-leg arbitrage
    │   └── execution/ # Order management
    ├── rust/          # Performance-critical code
    ├── infra/         # VPS, RPC setup
    └── backtests/
```

---

*This is a SEPARATE PROJECT. Do not mix with SI testing. Start only after current work is validated and you're ready for 3-6 month commitment.*
