#!/usr/bin/env python3
"""
Minimal test to verify SI computation works.
Run this BEFORE downloading real data.

Usage:
    python experiments/smoke_test.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd


def test_si_computation():
    """Test SI with synthetic data."""
    print("=" * 60)
    print("SMOKE TEST: SI Computation")
    print("=" * 60)

    # 1. Create synthetic price data
    np.random.seed(42)
    n = 1000
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n)),
        'low': prices - np.abs(np.random.randn(n)),
        'close': prices + np.random.randn(n) * 0.1,
        'volume': np.random.randint(100, 10000, n),
    }, index=pd.date_range('2024-01-01', periods=n, freq='1h'))

    print(f"✅ Created synthetic data: {len(data)} rows")

    # 2. Create simple strategies
    class SimpleStrategy:
        def __init__(self, bias: float, name: str = "simple"):
            self.bias = bias
            self.name = name

        def signal(self, data, idx):
            if idx < 10:
                return 0
            ret = data['close'].iloc[idx] / data['close'].iloc[idx-10] - 1
            return 1.0 if ret > self.bias else -1.0 if ret < -self.bias else 0.0

    strategies = [
        SimpleStrategy(0.01, "S1"),
        SimpleStrategy(0.02, "S2"),
        SimpleStrategy(0.03, "S3")
    ]
    print(f"✅ Created {len(strategies)} strategies")

    # 3. Create agents with niche affinities
    class SimpleAgent:
        def __init__(self, agent_id, strategy_idx):
            self.agent_id = agent_id
            self.strategy_idx = strategy_idx
            self.niche_affinity = np.ones(3) / 3

        def update(self, regime, won, alpha=0.1):
            if won:
                self.niche_affinity[regime] += alpha * (1 - self.niche_affinity[regime])
            else:
                self.niche_affinity[regime] *= (1 - alpha)
            total = self.niche_affinity.sum()
            if total > 0:
                self.niche_affinity /= total

    agents = [SimpleAgent(i, i % 3) for i in range(9)]
    print(f"✅ Created {len(agents)} agents")

    # 4. Run competition
    competition_rounds = 0
    for idx in range(100, 500):
        regime = idx % 3  # Simple regime rotation

        # Get returns
        returns = []
        for agent in agents:
            signal = strategies[agent.strategy_idx].signal(data, idx-1)
            ret = signal * (data['close'].iloc[idx] / data['close'].iloc[idx-1] - 1)
            returns.append((agent, ret))

        # Winner-take-all
        winner = max(returns, key=lambda x: x[1])[0]

        for agent, ret in returns:
            agent.update(regime, agent == winner)

        competition_rounds += 1

    print(f"✅ Ran {competition_rounds} competition rounds")

    # 5. Compute SI
    def compute_si(agents):
        entropies = []
        for agent in agents:
            p = agent.niche_affinity + 1e-10
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(len(p))
            entropies.append(entropy / max_entropy if max_entropy > 0 else 0)
        return 1 - np.mean(entropies)

    si = compute_si(agents)
    print(f"✅ Computed SI: {si:.3f}")

    # 6. Validate
    assert 0 <= si <= 1, f"SI out of range: {si}"
    print(f"✅ SI is in valid range [0, 1]")

    # 7. Check agents specialized
    print("\n   Agent specialization:")
    for i, agent in enumerate(agents):
        dominant = np.argmax(agent.niche_affinity)
        print(f"   Agent {i}: dominant regime {dominant}, affinity {agent.niche_affinity.round(2)}")

    # 8. Test imports work
    print("\n" + "=" * 60)
    print("Testing imports...")
    print("=" * 60)

    try:
        from src.agents.strategies import DEFAULT_STRATEGIES, MomentumStrategy
        print(f"✅ Imported strategies: {len(DEFAULT_STRATEGIES)} default strategies")
    except ImportError as e:
        print(f"❌ Failed to import strategies: {e}")
        return False

    try:
        from src.competition.niche_population import NichePopulation, Agent
        print(f"✅ Imported NichePopulation")
    except ImportError as e:
        print(f"❌ Failed to import NichePopulation: {e}")
        return False

    try:
        from src.analysis.features import FeatureCalculator
        print(f"✅ Imported FeatureCalculator")
    except ImportError as e:
        print(f"❌ Failed to import FeatureCalculator: {e}")
        return False

    try:
        from src.analysis.correlations import CorrelationAnalyzer
        print(f"✅ Imported CorrelationAnalyzer")
    except ImportError as e:
        print(f"❌ Failed to import CorrelationAnalyzer: {e}")
        return False

    try:
        from src.data.loader import MultiMarketLoader, MarketType
        print(f"✅ Imported MultiMarketLoader")
    except ImportError as e:
        print(f"❌ Failed to import MultiMarketLoader: {e}")
        return False

    try:
        from src.data.validation import DataValidator
        print(f"✅ Imported DataValidator")
    except ImportError as e:
        print(f"❌ Failed to import DataValidator: {e}")
        return False

    # 9. Test NichePopulation with synthetic data
    print("\n" + "=" * 60)
    print("Testing NichePopulation with synthetic data...")
    print("=" * 60)

    try:
        population = NichePopulation(DEFAULT_STRATEGIES, n_agents_per_strategy=2)
        print(f"✅ Created NichePopulation with {len(population.agents)} agents")

        # Run a few rounds
        for idx in range(200, 300):
            population.compete(data, idx)

        si_real = population.compute_si()
        print(f"✅ NichePopulation SI: {si_real:.3f}")

        assert 0 <= si_real <= 1, f"SI out of range: {si_real}"
        print(f"✅ NichePopulation SI is valid")
    except Exception as e:
        print(f"❌ NichePopulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 10. Test FeatureCalculator
    print("\n" + "=" * 60)
    print("Testing FeatureCalculator...")
    print("=" * 60)

    try:
        calc = FeatureCalculator()
        features = calc.compute_all(data)
        print(f"✅ Computed {len(features.columns)} features")
        print(f"   Discovery features: {len(calc.get_discovery_features())}")
        print(f"   Prediction features: {len(calc.get_prediction_features())}")

        # Check for NaN explosion
        nan_pct = features.isna().mean().mean() * 100
        print(f"   NaN percentage: {nan_pct:.1f}%")

        if nan_pct > 50:
            print(f"⚠️  High NaN percentage - check feature calculations")
    except Exception as e:
        print(f"❌ FeatureCalculator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("🎉 SMOKE TEST PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_si_computation()
    sys.exit(0 if success else 1)
