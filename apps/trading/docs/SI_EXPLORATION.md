# SI Exploration: What Is Specialization Index Actually Measuring?

**Date**: January 17, 2026
**Purpose**: Before assuming SI → Profit, let's discover what SI actually correlates with

---

## 🎯 The New Approach

**Old thinking**: "SI should lead to profit" (assumption)
**New thinking**: "What does SI actually measure? What is it related to?" (discovery)

```
┌────────────────────────────────────────────────────────────┐
│                    DISCOVERY APPROACH                       │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: Run backtest, compute SI time series              │
│                                                             │
│  Step 2: Correlate SI with EVERYTHING:                     │
│          - Market features (vol, trend, volume...)         │
│          - Risk metrics (drawdown, VaR, tail...)           │
│          - Agent behavior (correlation, turnover...)       │
│          - Meta features (regime, entropy, complexity...)  │
│                                                             │
│  Step 3: Find strongest correlations                       │
│          "SI is most related to X"                         │
│                                                             │
│  Step 4: Ask "Does X relate to profit?"                    │
│          If yes → We found the path!                       │
│          If no → SI might have different value             │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## 🔬 Expert Panel: What Might SI Actually Measure?

### Panel (Expanded)

| Expert | Domain | Key Questions They Can Answer |
|--------|--------|-------------------------------|
| 🧠 Information Theorist | Entropy, mutual information | Is SI measuring predictability? Information content? |
| 🌀 Complexity Scientist | Emergence, chaos, fractals | Is SI about system complexity? Edge of chaos? |
| 📈 Market Microstructure Expert | Liquidity, order flow | Does SI correlate with execution environment? |
| 🎲 Behavioral Finance Researcher | Biases, sentiment | Is SI capturing crowd psychology? |
| ⚡ Systems Dynamics Expert | Feedback loops, stability | Is SI about system stability? Convergence? |
| 🔮 Regime Detection Specialist | HMM, regime switching | Does SI identify regime persistence? |
| 🏦 Quant Strategist | Alpha, factor investing | Does SI predict alpha persistence? Crowding? |
| 💼 Portfolio Manager | Allocation, diversification | Does SI measure portfolio diversification benefit? |
| 📊 Risk Manager | VaR, drawdowns, tail risk | Does SI reduce risk? Predict crashes? |
| 🛡️ Execution Specialist | Slippage, timing | Does SI improve execution quality? |
| 🤖 ML Researcher | Calibration, ensembles | Can SI be used for meta-learning? |
| 🌐 Macro Strategist | Cross-asset, policy | Does SI have cross-asset predictive power? |
| ⏱️ Latency Specialist | Speed, turnover | Does SI correlate with time-to-profit? |
| 🔢 Statistician | Causality, mediation | How do we avoid p-hacking? Establish causality? |

---

## 💡 Wild Hypotheses: What SI Might Correlate With

### Category 1: Market State / Readability

#### H1: SI = Market Regime Clarity

**The Idea**: High SI emerges when the market has a CLEAR regime. Low SI when the market is confused/transitioning.

```
Clear trending market → Momentum specialist dominates → High SI
Choppy, unclear market → No one wins consistently → Low SI
```

🌀 **Complexity Scientist**: "This would mean SI is measuring the market's 'decidability' - how well-defined the current state is. This is related to entropy of the market state."

**What to correlate with**:
- Regime confidence (if using HMM/GMM)
- Autocorrelation (high autocorr = clear trend = high SI?)
- Hurst exponent (measure of trendiness)

---

#### H2: SI = Inverse of Market Entropy

**The Idea**: SI measures the "orderedness" of the market.

```
High SI = Low market entropy = Predictable patterns
Low SI = High market entropy = Random, unpredictable
```

🧠 **Information Theorist**: "Compute the Shannon entropy of returns. If SI negatively correlates with return entropy, then SI is measuring predictability."

**What to correlate with**:
- Entropy of return distribution
- Mutual information between past and future returns
- Predictability (simple AR model R²)

---

#### H3: SI = Regime Persistence

**The Idea**: High SI when the current regime is STABLE. Low SI when regimes are changing.

```
Regime stable for weeks → Specialists lock in → High SI
Regime changes frequently → No one adapts fast enough → Low SI
```

🔮 **Regime Detection Specialist**: "This is testable! Measure regime transition probabilities. High SI should correlate with low transition probability (regime is sticky)."

**What to correlate with**:
- Regime transition probability
- Days since last regime change
- Regime duration

---

### Category 2: Agent Behavior

#### H4: SI = Strategy Orthogonality

**The Idea**: SI measures how DIFFERENT the winning strategies are from each other.

```
High SI = Winners use genuinely different approaches
Low SI = Winners are doing similar things
```

💼 **Portfolio Manager**: "This is about return correlation. Compute pairwise correlation between agent returns. High SI should mean low average correlation."

**What to correlate with**:
- Average pairwise correlation between agents
- Principal component analysis - how many components explain variance?
- Effective N (diversification ratio)

---

#### H5: SI = Learning Convergence

**The Idea**: SI might measure whether agents have "figured out" the market.

```
Early in training: Agents exploring → Low SI
Later in training: Agents specialized → High SI
→ SI is a measure of LEARNING PROGRESS
```

🧠 **Information Theorist**: "This means SI is about the agents, not the market. Track SI over training time. If SI increases as agents learn, it's measuring convergence."

**What to correlate with**:
- Training iteration
- Agent confidence (Thompson Sampling posteriors)
- Win rate stability

---

#### H6: SI = Niche Stability

**The Idea**: High SI when niches are STABLE (same agents keep winning in same conditions).

```
Agent A always wins in volatility → Stable niche → High SI
Winners are random each period → No stable niches → Low SI
```

⚡ **Systems Dynamics Expert**: "Track which agent wins over time. High SI should correlate with low entropy of the winner distribution per condition."

**What to correlate with**:
- Winner consistency (same agent winning repeatedly)
- Niche affinity stability
- Switching rate between specialists

---

### Category 3: Market Microstructure

#### H7: SI = Market Liquidity State

**The Idea**: SI might correlate with market liquidity/microstructure.

```
High liquidity = Strategies work as expected → Specialists emerge → High SI
Low liquidity = Slippage, unpredictable execution → Strategies fail → Low SI
```

📈 **Market Microstructure Expert**: "Test correlation with bid-ask spread, volume, market depth. If SI correlates with liquidity, then SI is indirectly measuring execution environment."

**What to correlate with**:
- Trading volume
- Bid-ask spread (if available)
- Price impact
- Volatility of volatility

---

#### H8: SI = Information Flow Clarity

**The Idea**: High SI when information flows predictably into prices.

```
Clear information → Prices move smoothly → Strategies work → High SI
Noisy information → Prices jump randomly → Strategies fail → Low SI
```

📈 **Market Microstructure Expert**: "This is related to market efficiency. Test SI against measures of price discovery quality."

**What to correlate with**:
- Variance ratio tests
- Jump frequency
- News sentiment consistency

---

### Category 4: Meta / Abstract

#### H9: SI = Competitive Intensity

**The Idea**: SI might measure how HARD the agents are competing.

```
Easy market → Multiple strategies work → Low competition → Low SI
Hard market → Only specialists survive → High competition → High SI
```

🎲 **Behavioral Finance Researcher**: "This is about selection pressure. Track the gap between winner and loser returns. High SI should correlate with larger winner-loser gaps."

**What to correlate with**:
- Spread between best and worst agent returns
- Number of "viable" agents (positive return)
- Competition intensity metric

---

#### H10: SI = Complexity Matching

**The Idea**: High SI when agent complexity MATCHES market complexity.

```
Simple market + Specialized simple strategies → High SI
Complex market + Specialized complex strategies → High SI
Mismatch → Low SI
```

🌀 **Complexity Scientist**: "This is about the 'edge of chaos' idea. Systems perform best when their complexity matches the environment. SI might be measuring this fit."

**What to correlate with**:
- Market complexity (fractal dimension, entropy)
- Agent strategy complexity
- Ratio of the two

---

#### H11: SI = Alpha Persistence

**The Idea**: High SI when alpha (excess returns) persists over time.

```
Alpha decays quickly → Specialists can't maintain edge → Low SI
Alpha persists → Specialists can exploit consistently → High SI
```

🏦 **Quant Strategist**: "This is huge. If SI correlates with alpha persistence, then high SI periods are when you should trade more aggressively."

**What to correlate with**:
- Autocorrelation of excess returns
- Half-life of alpha
- Strategy decay rate

---

#### H12: SI = Fractal Self-Similarity

**The Idea**: SI at different time scales might be related (daily SI ↔ weekly SI ↔ monthly SI).

```
If SI is fractal, it reveals something fundamental about market structure
```

🌀 **Complexity Scientist**: "Compute SI at multiple time scales. If they correlate, there's a scale-invariant property. This would be a theoretical contribution."

**What to correlate with**:
- SI at different windows (1h, 4h, 1d, 1w)
- Cross-scale correlation

---

### Category 5: REALLY Wild Ideas

#### H13: SI = Market "Mood" Indicator

**The Idea**: SI might correlate with market sentiment/psychology.

```
Clear bullish/bearish sentiment → Trends persist → High SI
Mixed/confused sentiment → Choppy markets → Low SI
```

🎲 **Behavioral Finance Researcher**: "If you have access to sentiment data (fear/greed index, put/call ratio), test correlation with SI."

---

#### H14: SI as a Leading Indicator

**The Idea**: SI today might PREDICT something tomorrow.

```
Rising SI → Market becoming more structured → Tomorrow more predictable?
Falling SI → Market becoming chaotic → Tomorrow more volatile?
```

**What to test**:
- Does SI(t) predict volatility(t+1)?
- Does SI(t) predict regime change probability?
- Does SI(t) predict next-day returns?

---

#### H15: SI × Time = Different Meanings

**The Idea**: SI might mean different things at different times.

```
SI in trending market = Momentum specialist dominance
SI in ranging market = Mean-reversion specialist dominance
SI in crisis = Flight-to-safety specialist dominance
```

🔮 **Regime Detection Specialist**: "SI is regime-dependent. The SAME SI value means different things in different contexts. Analyze SI within regimes, not across."

---

#### H16: SI Velocity > SI Level

**The Idea**: CHANGE in SI is more informative than absolute SI.

```
Rising SI = Specialists emerging = System adapting = GOOD
Falling SI = Specialists converging = System confused = BAD
Stable SI = Equilibrium reached = Predictable
```

⚡ **Systems Dynamics Expert**: "This is about the derivative, not the value. Test dSI/dt against profit. Positive velocity might predict positive returns."

---

#### H17: SI as Canary in Coal Mine

**The Idea**: Sudden SI collapse might PREDICT crashes.

```
Specialists fail simultaneously = Correlation spike = All strategies broken = DANGER
```

📊 **Risk Manager**: "Track SI during known crisis periods (COVID crash, etc.). If SI drops BEFORE or AT the crash, it's an early warning signal."

---

#### H18: SI = Inverse of Crowding

**The Idea**: High SI means strategies are NOT crowded.

```
Many agents doing same thing = Crowded = Returns compete away = Low SI
Agents doing different things = Uncrowded = Returns preserved = High SI
```

🏦 **Quant Strategist**: "Crowding is a huge problem in real trading. If SI measures anti-crowding, that's directly valuable for position sizing."

---

### Category 6: Risk & Loss Avoidance (From Web Research)

#### H19: SI → Signal-to-Noise Ratio

**The Idea**: High SI corresponds to CLEANER signals with less noise.

```
High SI = Fewer false signals, more predictive setups
Low SI = Noisy, unreliable signals
```

📊 **Quant Researcher**: "Even if SI doesn't boost returns, it might reduce wasted capital and fewer losing trades. Track accuracy, precision/recall of signals tied to SI."

**What to correlate with**:
- Win rate of trades during high-SI vs low-SI periods
- Profit factor (gross profit / gross loss)
- Signal accuracy per SI quantile

**Research Evidence**: Signal-to-noise ratio is considered "the most misunderstood truth in trading" - clean signals matter more than signal frequency.

---

#### H20: SI → Loss Avoidance / Bad Trade Filter

**The Idea**: SI helps you AVOID bad trades, not find good ones.

```
High SI = Clear when to trade
Low SI = Warning to stay out
→ Value is in what you DON'T do
```

🛡️ **Risk Manager**: "Instead of using SI to enter, use low SI as a STOP signal. Filter out low-confidence periods."

**What to test**:
- Backtest "filtered strategy" discarding low-SI signals
- Compare false positive rates
- P&L from avoided incorrect trades

---

#### H21: SI → Tail Risk Protection / Crash Early Warning

**The Idea**: SI collapse might PRECEDE or COINCIDE with market crashes.

```
Normal: Specialists differentiated → High SI
Pre-crash: Correlations spike → Specialists fail together → SI drops
→ Falling SI = Early warning of systemic risk
```

📊 **Risk Manager**: "Research shows sentiment connectedness across firms forecasts crash risk. If SI drops before crashes, it's an early warning system."

**Research Evidence**: Studies show sentiment connectedness was incremental in forecasting stock price crash risk (PMC 2025).

**What to test**:
- SI behavior during COVID crash, 2022 crypto winter
- Lead time: Does SI drop BEFORE price drops?
- Compare to VIX, put/call ratio

---

#### H22: SI → Drawdown Recovery Time

**The Idea**: High SI systems recover from drawdowns FASTER.

```
High SI = Diverse specialists → Not all fail at once → Faster recovery
Low SI = Homogeneous → All fail together → Slow recovery
```

💼 **Portfolio Manager**: "Drawdown recovery time is often more important than drawdown depth. If SI correlates with faster recovery, that's huge."

**What to correlate with**:
- Time to recover from X% drawdown
- Recovery rate (% recovered per day)
- Maximum drawdown depth vs SI level

---

### Category 7: Timing & Regime (From Web Research)

#### H23: SI → Optimal Trading Windows

**The Idea**: SI identifies WHEN to trade, not just HOW.

```
High SI periods = Good time to be aggressive
Low SI periods = Good time to reduce exposure or sit out
```

🏦 **Factor Investor**: "Dynamic timing frameworks using AI show ~1.5%/year improvement. SI could be an input to such timing models."

**Research Evidence**: Northern Trust research shows AI-based factor timing achieves higher Sharpe by adjusting exposures dynamically.

**What to test**:
- Returns during high-SI vs low-SI periods
- Risk-adjusted returns when using SI for position sizing
- Sharpe improvement from SI-based exposure adjustment

---

#### H24: SI → Momentum vs Mean-Reversion Switch

**The Idea**: SI indicates which strategy TYPE will work.

```
High SI + Trending = Momentum wins
High SI + Ranging = Mean-reversion wins
Low SI = Neither works reliably
```

🔮 **Regime Specialist**: "SI combined with regime indicators could tell you not just WHEN to trade but WHAT strategy to use."

**What to test**:
- SI × ADX interaction (trend strength)
- SI × Bollinger Band width (ranging)
- Strategy returns conditional on SI and regime

---

#### H25: SI → Time-to-Profit (Latency)

**The Idea**: SI correlates with how QUICKLY trades resolve.

```
High SI = Fast winners/losers → Higher turnover, faster compounding
Low SI = Slow resolution → Capital tied up
```

⏱️ **Execution Specialist**: "Even if magnitude isn't bigger, speed matters for capital efficiency."

**What to correlate with**:
- Average holding period vs SI
- ROI per day stratified by SI
- Win rate vs time-to-resolution

---

### Category 8: Behavioral & Sentiment (From Web Research)

#### H26: SI → Contrarian Signal at Extremes

**The Idea**: EXTREME SI readings might predict reversals.

```
SI too high → Overconfidence → Overpriced → Reversal coming
SI too low → Panic → Oversold → Bounce coming
```

🎲 **Behavioral Finance**: "Extreme sentiment readings often precede reversals. If SI extremes predict mean-reversion, use it for contrarian entries."

**Research Evidence**: Wharton research shows long-short spreads following high sentiment are much more profitable due to subsequent reversals.

**What to test**:
- Returns after SI > 90th percentile
- Returns after SI < 10th percentile
- SI as contrarian indicator

---

#### H27: SI → Retail vs Institutional Behavior

**The Idea**: SI might differ based on market participant composition.

```
Retail-dominated periods: Noisier → Lower SI?
Institutional-dominated: More structured → Higher SI?
```

🎲 **Behavioral Finance**: "Retail flows create different market dynamics. SI might capture this shift in participant mix."

**What to correlate with**:
- Retail flow data (if available)
- Time of day patterns (retail more active at open/close)
- Small cap vs large cap SI differences

---

#### H28: SI → Sentiment Momentum / Feedback Loops

**The Idea**: SI might drive behavioral feedback loops.

```
Rising SI → Agents becoming more confident → More aggressive trading
→ Creates momentum in SI itself
→ Until SI reversal triggers strategy shift
```

**Research Evidence**: Studies show sentiment shocks have long half-life (~11 months) and sentiment-sorted portfolios deliver significant returns.

**What to test**:
- Autocorrelation of SI changes
- SI momentum as signal
- SI reversal patterns

---

### Category 9: Factor & Cross-Asset (From Web Research)

#### H29: SI → Factor Exposure Tilt

**The Idea**: SI correlates with which FACTORS outperform.

```
High SI → Growth/momentum factors work?
Low SI → Value/defensive factors work?
```

📈 **Factor Investor**: "SI might inform dynamic factor tilting. Even small tilt improvements compound significantly."

**What to correlate with**:
- Factor returns (value, momentum, size, quality) vs SI
- Factor timing using SI
- Cross-factor correlations

---

#### H30: SI → Cross-Asset Signals

**The Idea**: SI in one asset class might predict another.

```
Crypto SI high → Risk-on → Equities also rise?
Crypto SI low → Risk-off → Bonds outperform?
```

🌐 **Macro Strategist**: "Cross-asset signals are valuable. If SI in crypto predicts equity volatility, that's actionable."

**What to correlate with**:
- SI(BTC) vs SPY returns
- SI(crypto) vs VIX
- Cross-asset SI correlations

---

#### H31: SI → Sector Rotation Signal

**The Idea**: SI might indicate which sectors/assets to favor.

```
SI high in tech agents → Tech outperforming
SI shifts to energy agents → Rotate to energy
```

💼 **Portfolio Manager**: "SI variations across sectors could drive rotation strategies."

**What to test**:
- SI per asset (BTC, ETH, SOL) as relative signal
- Rotation strategy based on SI differences
- SI-weighted portfolio vs equal-weight

---

### Category 10: Operational & Meta (From Web Research)

#### H32: SI → Resource Allocation Efficiency

**The Idea**: Use SI to allocate computational/analytical resources.

```
High SI periods → Worth analyzing deeply → Deploy resources
Low SI periods → Noise, don't waste resources
```

⚙️ **Operations Expert**: "Better ROI on internal investment. Focus resources during high-opportunity windows."

**Value**: Even without direct profit, reduces operational costs.

---

#### H33: SI → Model Confidence Calibration

**The Idea**: SI as a meta-measure of how much to trust the system.

```
High SI = System confident, trust its signals
Low SI = System uncertain, reduce position sizes
```

🤖 **ML Researcher**: "Calibration is crucial. If SI correlates with prediction accuracy, use it to size bets."

**What to correlate with**:
- Prediction accuracy during high-SI vs low-SI
- Optimal position sizing as function of SI
- Kelly criterion adjustment based on SI

---

#### H34: SI Stability > SI Level

**The Idea**: How STABLE SI is matters more than absolute value.

```
Stable high SI = Reliable specialists = Consistent returns
Volatile SI = Unstable system = Unpredictable
```

📊 **Quant Researcher**: "Variance of SI over rolling window might be the key metric, not SI itself."

**What to correlate with**:
- Rolling std of SI vs returns
- SI stability vs Sharpe ratio
- Regime stability vs SI stability

---

#### H35: SI × Leverage = Profit Optimization

**The Idea**: Use SI to dynamically adjust leverage/exposure.

```
High SI = High confidence → Increase leverage
Low SI = Low confidence → Reduce leverage
```

🏦 **Risk Manager**: "Dynamic leverage based on signal quality is standard practice. If SI indicates signal quality, it's directly usable."

**What to test**:
- Backtested leveraged strategy with SI-based sizing
- Risk-adjusted returns with vs without SI-based leverage
- Optimal leverage curve as function of SI

---

### Category 11: Moonshot Ideas (From Web Research)

#### H36: SI → Macro/Policy Prediction

**The Idea**: SI might correlate with upcoming macro/policy changes.

```
Specialists picking up on subtle signals before announcements
SI changes → Macro shift incoming?
```

🌐 **Macro Strategist**: "If SI predicts Fed decisions, inflation surprises, etc., that's gold."

---

#### H37: SI → Black Swan Detection

**The Idea**: SI is especially valuable for asymmetric tail events.

```
Normal: SI not very predictive
Crisis: SI collapse → Signal to buy options, volatility
```

📊 **Options Trader**: "If SI only works for tail events, use it for asymmetric payoff structures (straddles, puts)."

---

#### H38: SI at Multiple Timescales

**The Idea**: Compute SI at 1h, 4h, 1d, 1w and look for patterns.

```
Short-term SI diverging from long-term SI → Regime change signal?
All timescales aligned → Strong trend continuation?
```

🌀 **Complexity Scientist**: "Multi-timescale analysis often reveals structure invisible at single scales."

---

#### H39: Ensemble/Crowdsourced SI

**The Idea**: Combine multiple SI-like metrics for robustness.

```
SI from Thompson Sampling
+ SI from strategy returns
+ SI from prediction accuracy
= Composite SI with less noise
```

🤖 **ML Researcher**: "Ensemble methods smooth noise. Meta-SI might be more predictive than raw SI."

---

#### H40: SI Network Effects

**The Idea**: SI behavior across NETWORK of agents might matter.

```
All agents' SI correlated → Systemic risk rising
Agents' SI diverging → Healthy diversification
```

🌐 **Network Scientist**: "Network-level metrics often predict system-wide events better than individual metrics."

---

## 📊 Comprehensive Correlation Matrix (40 Hypotheses)

Run ONE backtest, compute SI, then correlate with ALL of these:

```python
# =============================================================================
# CATEGORY 1-3: MARKET STATE & MICROSTRUCTURE
# =============================================================================

# Market State / Readability
correlate(SI, market_volatility)           # H1: Regime clarity
correlate(SI, market_trend_strength)       # H1: Regime clarity
correlate(SI, regime_confidence)           # H1: Regime clarity (HMM/GMM)
correlate(SI, return_autocorrelation)      # H2: Inverse entropy
correlate(SI, hurst_exponent)              # H2: Trendiness
correlate(SI, return_entropy)              # H2: Shannon entropy
correlate(SI, days_since_regime_change)    # H3: Regime persistence
correlate(SI, regime_duration)             # H3: Regime persistence

# Microstructure
correlate(SI, volume)                      # H7: Liquidity
correlate(SI, volume_volatility)           # H7: Liquidity stability
correlate(SI, bid_ask_spread)              # H7: Execution environment
correlate(SI, jump_frequency)              # H8: Information flow
correlate(SI, variance_ratio)              # H8: Price discovery

# =============================================================================
# CATEGORY 4-5: AGENT BEHAVIOR & META
# =============================================================================

# Agent Behavior
correlate(SI, agent_correlation)           # H4: Strategy orthogonality
correlate(SI, effective_n)                 # H4: Diversification ratio
correlate(SI, training_iteration)          # H5: Learning convergence
correlate(SI, agent_confidence)            # H5: Thompson posterior width
correlate(SI, winner_consistency)          # H6: Niche stability
correlate(SI, niche_switching_rate)        # H6: Niche stability
correlate(SI, winner_loser_spread)         # H9: Competitive intensity
correlate(SI, viable_agent_count)          # H9: How many agents profitable

# Meta / Abstract
correlate(SI, market_complexity)           # H10: Complexity matching
correlate(SI, alpha_autocorrelation)       # H11: Alpha persistence
correlate(SI, alpha_halflife)              # H11: Alpha decay
correlate(SI_1h, SI_1d)                    # H12: Fractal self-similarity
correlate(SI_1d, SI_1w)                    # H12: Cross-scale

# =============================================================================
# CATEGORY 6: RISK & LOSS AVOIDANCE (NEW FROM WEB RESEARCH)
# =============================================================================

# Risk Metrics
correlate(SI, max_drawdown)                # H21: Tail risk
correlate(SI, var_95)                      # H21: Value at Risk
correlate(SI, cvar_95)                     # H21: Expected Shortfall
correlate(SI, volatility_of_volatility)    # H21: Stability
correlate(SI, tail_ratio)                  # H21: Fat tails
correlate(SI, drawdown_recovery_time)      # H22: Recovery speed

# Signal Quality
correlate(SI, win_rate)                    # H19: Signal-to-noise
correlate(SI, profit_factor)               # H19: Gross profit / gross loss
correlate(SI, signal_accuracy)             # H20: Bad trade filter
correlate(SI, false_positive_rate)         # H20: Avoiding bad trades

# =============================================================================
# CATEGORY 7: TIMING & REGIME (NEW FROM WEB RESEARCH)
# =============================================================================

# Leading Indicators (SI as predictor)
correlate(SI_t, return_t1)                 # H14, H23: SI predicts next return
correlate(SI_t, volatility_t1)             # H14, H23: SI predicts next vol
correlate(SI_t, regime_change_t1)          # H14: Regime change probability
correlate(SI_t, momentum_return_t1)        # H24: Strategy type prediction
correlate(SI_t, meanrev_return_t1)         # H24: Strategy type prediction
correlate(SI, holding_period)              # H25: Time-to-profit

# Dynamics
correlate(dSI_dt, profit)                  # H16: SI velocity
correlate(SI_stability, profit)            # H34: SI consistency
correlate(SI_rolling_std, sharpe)          # H34: SI variance matters

# =============================================================================
# CATEGORY 8: BEHAVIORAL & SENTIMENT (NEW FROM WEB RESEARCH)
# =============================================================================

# Sentiment & Behavior
correlate(SI, fear_greed_index)            # H13, H26: Market mood
correlate(SI, put_call_ratio)              # H13: Sentiment proxy
correlate(SI_extreme_high, future_return)  # H26: Contrarian at extremes
correlate(SI_extreme_low, future_return)   # H26: Contrarian at extremes
correlate(SI_autocorr, profit)             # H28: SI momentum

# =============================================================================
# CATEGORY 9: FACTOR & CROSS-ASSET (NEW FROM WEB RESEARCH)
# =============================================================================

# Factor Exposure
correlate(SI, momentum_factor_return)      # H29: Factor tilt
correlate(SI, value_factor_return)         # H29: Factor tilt
correlate(SI, quality_factor_return)       # H29: Factor tilt

# Cross-Asset
correlate(SI_BTC, SI_ETH)                  # H30: Cross-asset SI
correlate(SI_crypto, equity_return)        # H30: Cross-asset signals
correlate(SI_crypto, VIX)                  # H30: Risk-off indicator

# Per-Asset SI
correlate(SI_per_asset_diff, rotation_return)  # H31: Sector rotation

# =============================================================================
# CATEGORY 10-11: OPERATIONAL & MOONSHOT (NEW FROM WEB RESEARCH)
# =============================================================================

# Operational
correlate(SI, prediction_accuracy)         # H33: Model calibration
correlate(SI, optimal_position_size)       # H35: Leverage optimization

# Moonshot
correlate(SI_pre_announcement, macro_surprise)  # H36: Macro prediction
correlate(SI_pre_crash, crash_magnitude)   # H37: Black swan detection
```

---

## 🎯 The Discovery Protocol

### Phase 0a: Data Collection

Run backtest and collect:
- SI time series
- Agent returns
- All market features
- All risk metrics

### Phase 0b: Correlation Discovery

```python
# Compute all correlations
correlations = {}
for feature in all_features:
    r, p = pearsonr(SI, feature)
    correlations[feature] = {'r': r, 'p': p}

# Sort by absolute correlation
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]['r']), reverse=True)

# Report top 10
print("SI is most strongly related to:")
for feature, stats in sorted_correlations[:10]:
    print(f"  {feature}: r={stats['r']:.3f}, p={stats['p']:.4f}")
```

### Phase 0c: Interpretation

Based on what SI correlates with, interpret:

| If SI correlates with... | Then SI measures... | Path to profit... |
|--------------------------|---------------------|-------------------|
| Market volatility (-) | Calm/clear markets | Trade more when SI high |
| Regime stability (+) | Regime persistence | Hold positions longer when SI high |
| Agent correlation (-) | Diversification | Better Sharpe through diversity |
| Return predictability (+) | Forecastability | SI as meta-signal for aggression |
| Next-day returns (+) | Leading indicator | SI as timing signal |
| Alpha persistence (+) | Strategy durability | Size up when SI high |

### Phase 0d: Trace to Profit

Once we know what SI measures, ask:
- "Does X relate to profit?"
- "Can we use SI as a SIGNAL for trading decisions?"
- "Is SI value in risk management, not return generation?"

---

## 🚀 Updated Experiment Plan

### Week 1: Discovery Phase

| Day | Task | Output |
|-----|------|--------|
| 1-2 | Build backtest infrastructure | Working system |
| 3-4 | Run backtest, compute SI + ALL features | Data dump |
| 5 | Correlation analysis | Top 10 SI correlates |
| 6 | Interpretation workshop | "SI measures X" |
| 7 | Path to profit analysis | "X leads to profit via Y" |

### Decision Points

```
After correlation analysis:

IF SI correlates with profit directly:
    → Great! Proceed with original thesis

ELIF SI correlates with risk metrics:
    → Pivot: "SI for risk management"

ELIF SI correlates with predictability:
    → Pivot: "SI as meta-signal"

ELIF SI correlates with diversification:
    → Pivot: "SI for portfolio construction"

ELIF SI correlates with nothing:
    → Deeper analysis or abandon
```

---

## 📋 Summary: All 40 Hypotheses

| # | Hypothesis | Category | Priority | Testability |
|---|------------|----------|----------|-------------|
| H1 | SI = Market Regime Clarity | Market State | ⭐⭐⭐ | Easy |
| H2 | SI = Inverse of Market Entropy | Market State | ⭐⭐⭐ | Easy |
| H3 | SI = Regime Persistence | Market State | ⭐⭐⭐ | Easy |
| H4 | SI = Strategy Orthogonality | Agent Behavior | ⭐⭐⭐ | Easy |
| H5 | SI = Learning Convergence | Agent Behavior | ⭐⭐ | Medium |
| H6 | SI = Niche Stability | Agent Behavior | ⭐⭐⭐ | Easy |
| H7 | SI = Market Liquidity State | Microstructure | ⭐⭐ | Medium |
| H8 | SI = Information Flow Clarity | Microstructure | ⭐⭐ | Medium |
| H9 | SI = Competitive Intensity | Meta | ⭐⭐ | Easy |
| H10 | SI = Complexity Matching | Meta | ⭐ | Hard |
| H11 | SI = Alpha Persistence | Meta | ⭐⭐⭐ | Medium |
| H12 | SI = Fractal Self-Similarity | Meta | ⭐ | Medium |
| H13 | SI = Market Mood Indicator | Wild | ⭐⭐ | Medium |
| H14 | SI as Leading Indicator | Wild | ⭐⭐⭐ | Easy |
| H15 | SI × Time = Different Meanings | Wild | ⭐⭐ | Medium |
| H16 | SI Velocity > SI Level | Wild | ⭐⭐⭐ | Easy |
| H17 | SI as Crash Early Warning | Wild | ⭐⭐⭐ | Medium |
| H18 | SI = Inverse of Crowding | Wild | ⭐⭐⭐ | Easy |
| H19 | SI → Signal-to-Noise Ratio | Risk | ⭐⭐⭐ | Easy |
| H20 | SI → Loss Avoidance Filter | Risk | ⭐⭐⭐ | Easy |
| H21 | SI → Tail Risk Protection | Risk | ⭐⭐⭐ | Medium |
| H22 | SI → Drawdown Recovery Time | Risk | ⭐⭐⭐ | Easy |
| H23 | SI → Optimal Trading Windows | Timing | ⭐⭐⭐ | Easy |
| H24 | SI → Momentum/MeanRev Switch | Timing | ⭐⭐⭐ | Medium |
| H25 | SI → Time-to-Profit | Timing | ⭐⭐ | Easy |
| H26 | SI → Contrarian at Extremes | Behavioral | ⭐⭐⭐ | Easy |
| H27 | SI → Retail vs Institutional | Behavioral | ⭐ | Hard |
| H28 | SI → Sentiment Momentum | Behavioral | ⭐⭐ | Medium |
| H29 | SI → Factor Exposure Tilt | Factor | ⭐⭐ | Medium |
| H30 | SI → Cross-Asset Signals | Factor | ⭐⭐ | Medium |
| H31 | SI → Sector Rotation | Factor | ⭐⭐ | Medium |
| H32 | SI → Resource Allocation | Operational | ⭐ | Easy |
| H33 | SI → Model Calibration | Operational | ⭐⭐ | Easy |
| H34 | SI Stability > SI Level | Operational | ⭐⭐⭐ | Easy |
| H35 | SI × Leverage Optimization | Operational | ⭐⭐⭐ | Medium |
| H36 | SI → Macro/Policy Prediction | Moonshot | ⭐ | Hard |
| H37 | SI → Black Swan Detection | Moonshot | ⭐⭐ | Hard |
| H38 | SI at Multiple Timescales | Moonshot | ⭐⭐ | Medium |
| H39 | Ensemble/Crowdsourced SI | Moonshot | ⭐ | Medium |
| H40 | SI Network Effects | Moonshot | ⭐ | Hard |

---

## 🎯 Top 10 Hypotheses to Test First

Based on priority (value if true) × testability (ease of testing):

| Rank | Hypothesis | Why Priority |
|------|------------|--------------|
| 1 | **H19: SI → Signal-to-Noise** | Direct value: cleaner signals = less wasted capital |
| 2 | **H16: SI Velocity > Level** | Novel insight: rate of change might matter more |
| 3 | **H22: SI → Drawdown Recovery** | Risk managers care deeply about this |
| 4 | **H23: SI → Trading Windows** | Practical: when to be aggressive |
| 5 | **H14: SI as Leading Indicator** | If true, SI becomes a timing signal |
| 6 | **H4: SI → Diversification** | Classic portfolio value |
| 7 | **H18: SI = Anti-Crowding** | Huge for real-world trading |
| 8 | **H21: SI → Tail Protection** | Crash early warning is gold |
| 9 | **H11: SI → Alpha Persistence** | When SI high, alpha lasts |
| 10 | **H26: SI → Contrarian Extremes** | SI extremes as reversal signal |

---

## 💡 Expert Final Thoughts

🧠 **Information Theorist**: "SI is fundamentally about INFORMATION. Either information about the market (regime clarity) or information about the agents (learning convergence). Frame your analysis around information flow."

🌀 **Complexity Scientist**: "Don't expect simple linear correlations. SI might have nonlinear relationships, threshold effects, or context-dependent meanings. Consider mutual information, not just Pearson correlation."

📈 **Market Microstructure Expert**: "The market features you have access to matter. With just price data, you're limited. With order book data, you could test much richer hypotheses."

🎲 **Behavioral Finance Researcher**: "Remember: correlation ≠ causation. Even if SI correlates with something, the causal arrow might point the other way. Always test temporal precedence (Granger causality)."

⚡ **Systems Dynamics Expert**: "Dynamic systems rarely have static relationships. Test how SI's correlations CHANGE over time. A correlation that holds only in certain periods is still valuable."

🔮 **Regime Detection Specialist**: "Stratify everything by regime. SI might mean completely different things in trending vs mean-reverting markets."

🏦 **Quant Strategist**: "The most valuable finding would be SI correlating with alpha persistence or anti-crowding. These translate directly to trading decisions."

📊 **Risk Manager**: "If SI predicts tail risk or crash timing even with lead time of hours, that's extremely valuable for hedging and position sizing."

🛡️ **Execution Specialist**: "Don't forget about signal quality. A signal that's right 60% of the time with low SI vs 70% with high SI is a huge difference."

🤖 **ML Researcher**: "Consider using mutual information instead of Pearson correlation for nonlinear relationships. Also test SI as a weighting factor in ensemble models."

🌐 **Macro Strategist**: "Cross-asset SI signals are underexplored. If crypto SI predicts equity volatility, you're onto something unique."

🔢 **Statistician**: "With 40 hypotheses, beware of multiple testing. Use Bonferroni correction or FDR control. Report effect sizes, not just p-values."

---

## 📚 Key Research References (From Web Search)

| Finding | Source | Relevance |
|---------|--------|-----------|
| Sentiment connectedness forecasts crash risk | PMC 2025 | Supports H17, H21 |
| Long-short spreads after high sentiment are profitable | Wharton | Supports H26 |
| LLM sentiment achieves Sharpe ~3.05 | arxiv 2412.19245 | Sentiment signals work |
| Sentiment half-life ~11 months | arxiv 2509.11970 | H28 sentiment momentum |
| AI factor timing +1.5%/year | Northern Trust | H23 timing value |
| Domain-specific sentiment beats general | ScienceDirect 2024 | Context matters |
| Signal-to-noise most misunderstood in trading | TradingView | H19 importance |

---

## 🚀 Next Steps

1. **Build backtest infrastructure** (Day 1-2)
2. **Compute SI + all 70+ features** (Day 3-4)
3. **Run correlation matrix** (Day 5)
4. **Identify top 10 correlations** (Day 6)
5. **Deep-dive on top findings** (Day 7)
6. **Expert review of interpretation** (Week 2)

---

*This is a DISCOVERY document. We're not assuming anything - we're finding out what SI actually measures.*

*40 hypotheses. 70+ features. 14 experts. Let's find out what SI really is.*

*Last Updated: January 17, 2026*
