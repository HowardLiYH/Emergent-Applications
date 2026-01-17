"""
NichePopulation algorithm - FREQUENCY AWARE VERSION.

Fixes the issue where hourly windows were applied to daily data.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Literal
from dataclasses import dataclass, field


# Lookback mapping: logical period -> bars
LOOKBACK_MAP = {
    'hourly': 168,     # 7 days in hours
    'daily': 7,        # 7 days
}


@dataclass
class Agent:
    """Trading agent with niche affinity tracking."""

    agent_id: int
    strategy_idx: int

    niche_affinity: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))
    cumulative_return: float = 0.0
    win_count: int = 0
    total_trades: int = 0

    def update_affinity(self, regime_idx: int, won: bool, alpha: float = 0.1):
        """Update niche affinity based on competition outcome."""
        if won:
            self.niche_affinity[regime_idx] += alpha * (1 - self.niche_affinity[regime_idx])
        else:
            self.niche_affinity[regime_idx] *= (1 - alpha)

        total = self.niche_affinity.sum()
        if total > 0:
            self.niche_affinity /= total


class NichePopulationV2:
    """
    Frequency-aware population of competing agents.

    Usage:
        population = NichePopulationV2(strategies, frequency='daily')
        population.run(train, start_idx=10)
        si = population.compute_si_timeseries(train, window=7)  # 7 days for daily
    """

    def __init__(
        self,
        strategies: List,
        n_agents_per_strategy: int = 3,
        frequency: Literal['hourly', 'daily'] = 'daily'
    ):
        self.strategies = strategies
        self.frequency = frequency
        self.lookback_7d = LOOKBACK_MAP[frequency]
        self.agents: List[Agent] = []

        agent_id = 0
        for strategy_idx in range(len(strategies)):
            for _ in range(n_agents_per_strategy):
                self.agents.append(Agent(
                    agent_id=agent_id,
                    strategy_idx=strategy_idx,
                    niche_affinity=np.ones(3) / 3
                ))
                agent_id += 1

        self.history = []

    def classify_regime(self, data: pd.DataFrame, idx: int) -> int:
        """
        Classify current market regime with frequency-aware lookback.

        Returns:
            0 = Trending
            1 = Mean-reverting
            2 = High volatility
        """
        lookback = self.lookback_7d

        if idx < lookback:
            return 0

        returns = data['close'].pct_change().iloc[idx-lookback:idx]

        vol = returns.std()
        trend = abs(returns.mean()) / vol if vol > 0 else 0

        # Get historical vol for comparison
        lookback_2x = lookback * 2
        if idx >= lookback_2x:
            historical_vol = data['close'].pct_change().iloc[idx-lookback_2x:idx-lookback].std()
        else:
            historical_vol = vol

        # Simple classification
        if vol > historical_vol * 1.5:
            return 2  # High volatility
        elif trend > 0.1:
            return 0  # Trending
        else:
            return 1  # Mean-reverting

    def compete(self, data: pd.DataFrame, idx: int) -> Dict:
        """Run one competition round."""
        regime = self.classify_regime(data, idx)

        returns_list = []
        next_return = data['close'].pct_change().iloc[idx] if idx < len(data) - 1 else 0

        for agent in self.agents:
            strategy = self.strategies[agent.strategy_idx]
            signal = strategy.signal(data, idx - 1)
            agent_return = signal * next_return
            returns_list.append((agent, agent_return))

        returns_list.sort(key=lambda x: x[1], reverse=True)
        winner = returns_list[0][0]

        for agent, agent_return in returns_list:
            won = (agent == winner)
            agent.update_affinity(regime, won)
            agent.cumulative_return += agent_return
            agent.total_trades += 1
            if won:
                agent.win_count += 1

        result = {
            'idx': idx,
            'timestamp': data.index[idx],
            'regime': regime,
            'winner_id': winner.agent_id,
            'winner_strategy': winner.strategy_idx,
            'returns': [r[1] for r in returns_list],
        }
        self.history.append(result)

        return result

    def run(self, data: pd.DataFrame, start_idx: int = None) -> pd.DataFrame:
        """Run competition over entire dataset."""
        from tqdm import tqdm

        # Auto-compute start_idx based on frequency
        if start_idx is None:
            start_idx = self.lookback_7d * 2  # Need 2x lookback for regime comparison

        # Ensure start_idx is valid
        start_idx = max(start_idx, self.lookback_7d * 2)
        start_idx = min(start_idx, len(data) - 10)

        self.data_index = data.index

        for idx in tqdm(range(start_idx, len(data)), desc="Competition"):
            self.compete(data, idx)

        return pd.DataFrame(self.history)

    def compute_si(self) -> float:
        """Compute Specialization Index for the population."""
        entropies = []
        for agent in self.agents:
            p = agent.niche_affinity + 1e-10
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(len(p))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            entropies.append(normalized_entropy)

        mean_entropy = np.mean(entropies)
        si = 1 - mean_entropy

        return si

    def compute_si_timeseries(self, data: pd.DataFrame, window: int = None) -> pd.Series:
        """
        Compute SI over time using rolling window.

        Args:
            data: Price data
            window: Rolling window in BARS (auto-set to 7d equivalent if None)
        """
        if len(self.history) == 0:
            raise ValueError("Run competition first!")

        # Auto-set window based on frequency
        if window is None:
            window = self.lookback_7d

        si_values = []
        timestamps = []

        # Store original affinities
        original_affinities = [agent.niche_affinity.copy() for agent in self.agents]

        for i in range(window, len(self.history)):
            # Reset affinities
            for agent in self.agents:
                agent.niche_affinity = np.ones(3) / 3

            # Replay window
            for j in range(i - window, i):
                record = self.history[j]
                regime = record['regime']
                winner_id = record['winner_id']

                for agent in self.agents:
                    won = (agent.agent_id == winner_id)
                    agent.update_affinity(regime, won)

            si_values.append(self.compute_si())
            timestamps.append(self.history[i]['timestamp'])

        # Restore original affinities
        for agent, affinity in zip(self.agents, original_affinities):
            agent.niche_affinity = affinity

        # Handle timezone-aware timestamps
        if len(timestamps) > 0:
            if hasattr(timestamps[0], 'tz') and timestamps[0].tz is not None:
                timestamps = [t.tz_convert('UTC').tz_localize(None) for t in timestamps]
            elif hasattr(timestamps[0], 'tzinfo') and timestamps[0].tzinfo is not None:
                timestamps = [t.replace(tzinfo=None) for t in timestamps]

        return pd.Series(si_values, index=pd.DatetimeIndex(timestamps), name='si')

    def get_recommended_si_window(self) -> int:
        """Get recommended SI window for the current frequency."""
        return self.lookback_7d
