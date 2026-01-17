# Future Enhancement: Markov Chain Regime Prediction

> ⚠️ **DO NOT START UNTIL**: SI correlation testing is complete and validated

---

## Prerequisites

- [ ] SI testing complete (Phase 1-8)
- [ ] SI correlates with regime-related features
- [ ] Cross-market validation passed

---

## The Idea

**Current limitation**: We detect regimes AFTER they happen, not predict them.

**Enhancement**: Use Hidden Markov Models (HMM) to predict regime TRANSITIONS.

```
CURRENT:
  Market data → Detect current regime → SI signals → Trade

ENHANCED:
  Market data → Detect current regime → Predict NEXT regime → SI signals → Trade AHEAD
```

---

## Why This Matters

| Without Prediction | With Prediction |
|--------------------|-----------------|
| React to regime change | Anticipate regime change |
| Enter after transition | Position before transition |
| Chase the move | Catch the move |

---

## Implementation (When Ready)

```python
from hmmlearn import hmm
import numpy as np
import pandas as pd

class RegimePredictor:
    """
    Predict next market regime using Hidden Markov Model.

    DO NOT USE UNTIL SI IS VALIDATED.
    """

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.regime_names = ['trending', 'mean_reverting', 'volatile']
        self.fitted = False

    def fit(self, features: np.ndarray):
        """
        Fit HMM to historical features.

        Args:
            features: Array of shape (n_samples, n_features)
                      e.g., [volatility, trend_strength, autocorr]
        """
        self.model.fit(features)
        self.fitted = True
        return self

    def predict_current_regime(self, features: np.ndarray) -> int:
        """Get current hidden state (regime)."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(features)[-1]

    def predict_next_regime(self, features: np.ndarray) -> dict:
        """
        Predict probability of each regime in the NEXT period.

        Returns:
            dict: {regime_name: probability}
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get current hidden state
        current_state = self.predict_current_regime(features)

        # Transition matrix gives P(next_state | current_state)
        transition_probs = self.model.transmat_[current_state]

        return {
            name: float(prob)
            for name, prob in zip(self.regime_names, transition_probs)
        }

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get regime transition probability matrix.

        Returns:
            DataFrame where cell [i,j] = P(regime_j | regime_i)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return pd.DataFrame(
            self.model.transmat_,
            index=self.regime_names,
            columns=self.regime_names
        )

    def get_regime_durations(self) -> dict:
        """
        Estimate expected duration of each regime.

        Duration = 1 / (1 - P(stay in same regime))
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        durations = {}
        for i, name in enumerate(self.regime_names):
            p_stay = self.model.transmat_[i, i]
            expected_duration = 1 / (1 - p_stay) if p_stay < 1 else float('inf')
            durations[name] = expected_duration

        return durations


def prepare_hmm_features(data: pd.DataFrame) -> np.ndarray:
    """
    Prepare features for HMM training.

    Uses:
    - Volatility (24h rolling)
    - Trend strength (7d)
    - Return autocorrelation (1d)
    """
    returns = data['close'].pct_change()

    features = pd.DataFrame({
        'volatility': returns.rolling(24).std(),
        'trend': abs(returns.rolling(168).mean()) / returns.rolling(168).std(),
        'autocorr': returns.rolling(24).apply(lambda x: x.autocorr(lag=1), raw=False),
    }).dropna()

    # Normalize
    features = (features - features.mean()) / features.std()

    return features.values
```

---

## How SI + Regime Prediction Work Together

```python
def enhanced_trading_signal(si: float, regime_probs: dict,
                            si_regime_map: dict) -> str:
    """
    Combine SI signal with regime prediction.

    Args:
        si: Current Specialization Index
        regime_probs: P(next_regime) from HMM
        si_regime_map: What SI means in each regime (from SI testing)

    Returns:
        Trading action
    """
    # Get most likely next regime
    next_regime = max(regime_probs, key=regime_probs.get)
    next_prob = regime_probs[next_regime]

    # If high confidence in regime transition
    if next_prob > 0.6:
        # Use SI interpretation for NEXT regime
        si_meaning = si_regime_map.get(next_regime, 'unknown')

        if si > 0.7 and si_meaning == 'high_si_is_bullish':
            return 'LONG_AHEAD_OF_TRANSITION'
        elif si < 0.3 and si_meaning == 'low_si_is_bearish':
            return 'SHORT_AHEAD_OF_TRANSITION'

    return 'HOLD'
```

---

## Validation Plan (When Ready)

1. **Fit HMM** on training data
2. **Evaluate** regime prediction accuracy on validation data
3. **Compare**:
   - SI-only signals vs
   - SI + regime prediction signals
4. **Measure** improvement in Sharpe ratio

---

## Dependencies

```
hmmlearn>=0.3.0
```

---

## Estimated Effort

| Task | Time |
|------|------|
| Implement RegimePredictor | 1 day |
| Integrate with SI | 1 day |
| Backtest comparison | 2 days |
| **Total** | **4 days** |

---

## References

- [hmmlearn documentation](https://hmmlearn.readthedocs.io/)
- Hamilton (1989) "A New Approach to the Economic Analysis of Nonstationary Time Series"
- Regime Switching Models in Finance

---

*This enhancement is PARKED. Start only after SI validation is complete.*
