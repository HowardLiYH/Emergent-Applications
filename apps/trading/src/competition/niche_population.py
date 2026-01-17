"""
NichePopulation algorithm for emergent specialization.
Adapted from Paper 1.
"""
import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class Agent:
    """Trading agent with niche affinity tracking."""

    agent_id: int
    strategy_idx: int  # Which strategy this agent uses

    # Niche affinities: one per market regime
    niche_affinity: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))

    # Performance tracking
    cumulative_return: float = 0.0
    win_count: int = 0
    total_trades: int = 0

    def update_affinity(self, regime_idx: int, won: bool, alpha: float = 0.1):
        """Update niche affinity based on competition outcome."""
        if won:
            # Increase affinity for this regime
            self.niche_affinity[regime_idx] += alpha * (1 - self.niche_affinity[regime_idx])
        else:
            # Decrease affinity
            self.niche_affinity[regime_idx] *= (1 - alpha)

        # Normalize
        total = self.niche_affinity.sum()
        if total > 0:
            self.niche_affinity /= total


class NichePopulation:
    """
    Population of competing agents that develop specialization.
    """

    def __init__(self, strategies: List, n_agents_per_strategy: int = 3):
        self.strategies = strategies
        self.agents: List[Agent] = []

        # Create agents
        agent_id = 0
        for strategy_idx in range(len(strategies)):
            for _ in range(n_agents_per_strategy):
                self.agents.append(Agent(
                    agent_id=agent_id,
                    strategy_idx=strategy_idx,
                    niche_affinity=np.ones(3) / 3  # Start uniform
                ))
                agent_id += 1

        self.history = []

    def classify_regime(self, data: pd.DataFrame, idx: int, lookback: int = 168) -> int:
        """
        Classify current market regime.

        Returns:
            0 = Trending
            1 = Mean-reverting
            2 = High volatility
        """
        if idx < lookback:
            return 0

        returns = data['close'].pct_change().iloc[idx-lookback:idx]

        vol = returns.std()
        trend = abs(returns.mean()) / vol if vol > 0 else 0

        # Get historical vol for comparison
        if idx >= lookback * 2:
            historical_vol = data['close'].pct_change().iloc[idx-lookback*2:idx-lookback].std()
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
        """
        Run one competition round.

        Returns:
            Dict with competition results and metrics
        """
        regime = self.classify_regime(data, idx)

        # Get signals and returns for each agent
        returns_list = []
        next_return = data['close'].pct_change().iloc[idx] if idx < len(data) - 1 else 0

        for agent in self.agents:
            strategy = self.strategies[agent.strategy_idx]
            signal = strategy.signal(data, idx - 1)  # Signal from previous bar
            agent_return = signal * next_return
            returns_list.append((agent, agent_return))

        # Winner-take-all: best return wins
        returns_list.sort(key=lambda x: x[1], reverse=True)
        winner = returns_list[0][0]

        # Update all agents
        for agent, agent_return in returns_list:
            won = (agent == winner)
            agent.update_affinity(regime, won)
            agent.cumulative_return += agent_return
            agent.total_trades += 1
            if won:
                agent.win_count += 1

        # Record history with timestamp
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

    def run(self, data: pd.DataFrame, start_idx: int = 200) -> pd.DataFrame:
        """Run competition over entire dataset."""
        from tqdm import tqdm

        self.data_index = data.index  # Store data index for later use

        for idx in tqdm(range(start_idx, len(data)), desc="Competition"):
            self.compete(data, idx)

        return pd.DataFrame(self.history)

    def compute_si(self) -> float:
        """
        Compute Specialization Index for the population.

        SI = 1 - mean(entropy of niche affinities)

        High SI = agents are specialized (low entropy)
        Low SI = agents are generalists (high entropy)
        """
        entropies = []
        for agent in self.agents:
            # Entropy of niche affinity distribution
            p = agent.niche_affinity + 1e-10  # Avoid log(0)
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(len(p))  # Maximum possible entropy
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            entropies.append(normalized_entropy)

        mean_entropy = np.mean(entropies)
        si = 1 - mean_entropy

        return si

    def compute_si_timeseries(self, data: pd.DataFrame, window: int = 168) -> pd.Series:
        """Compute SI over time using rolling window."""
        if len(self.history) == 0:
            raise ValueError("Run competition first!")

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
            # Use timestamp from history instead of integer index
            timestamps.append(self.history[i]['timestamp'])

        # Restore original affinities
        for agent, affinity in zip(self.agents, original_affinities):
            agent.niche_affinity = affinity

        # Convert timestamps to UTC if timezone-aware, else use as-is
        if len(timestamps) > 0:
            # Handle timezone-aware timestamps
            if hasattr(timestamps[0], 'tz') and timestamps[0].tz is not None:
                timestamps = [t.tz_convert('UTC').tz_localize(None) for t in timestamps]
            elif hasattr(timestamps[0], 'tzinfo') and timestamps[0].tzinfo is not None:
                timestamps = [t.replace(tzinfo=None) for t in timestamps]

        return pd.Series(si_values, index=pd.DatetimeIndex(timestamps), name='si')
