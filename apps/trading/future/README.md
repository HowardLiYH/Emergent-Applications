# Future Enhancements (After SI is Validated)

> ⚠️ **DO NOT START THESE UNTIL SI TESTING IS COMPLETE**
>
> Current focus: Validate that SI correlates with something useful.
> These ideas are parked here for later.

---

## 📍 Current Priority

```
NOW:     Test SI → Find correlations → Validate across markets
LATER:   Everything in this folder
```

---

## 📁 Contents

| File | Enhancement | Depends On |
|------|-------------|------------|
| `REGIME_PREDICTION.md` | Markov Chain regime forecasting | SI validated |
| `POLYMARKET_STRATEGY.md` | Betting market arbitrage | SI validated + regime prediction |
| `LIVE_TRADING.md` | Real money implementation | All above |

---

## When to Start These

| Enhancement | Start When |
|-------------|------------|
| Regime Prediction | SI shows significant correlation with regimes |
| Polymarket | Regime prediction works + want new domain |
| Live Trading | Backtests profitable + risk management ready |

---

## Decision Framework

```
IF SI correlates with nothing:
   → Pivot or abandon (don't proceed to future enhancements)

IF SI correlates with volatility/risk:
   → Proceed to Regime Prediction

IF SI + Regime Prediction works:
   → Consider Polymarket or Live Trading
```

---

*Last Updated: January 17, 2026*
