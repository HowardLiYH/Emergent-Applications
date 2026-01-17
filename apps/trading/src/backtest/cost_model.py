#!/usr/bin/env python3
"""
Transaction Cost Model for SI Strategy Backtesting.

Implements realistic transaction costs for different market types.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from enum import Enum


class MarketType(Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCKS = "stocks"
    COMMODITIES = "commodities"


# Research-based transaction costs (round-trip)
# Sources: Binance fee tiers, forex broker spreads, equity commission studies
TRANSACTION_COSTS = {
    MarketType.CRYPTO: {
        'fee': 0.0004,        # 4 bps taker fee (Binance standard)
        'slippage': 0.0002,   # 2 bps estimated slippage
        'total': 0.0006,      # 6 bps one-way, 12 bps round-trip
    },
    MarketType.FOREX: {
        'fee': 0.0000,        # No commission for most retail
        'slippage': 0.0001,   # 1 bp spread (major pairs)
        'total': 0.0001,      # 1 bp one-way, 2 bps round-trip
    },
    MarketType.STOCKS: {
        'fee': 0.0001,        # 1 bp commission (institutional)
        'slippage': 0.0001,   # 1 bp estimated slippage (liquid ETFs)
        'total': 0.0002,      # 2 bps one-way, 4 bps round-trip
    },
    MarketType.COMMODITIES: {
        'fee': 0.0002,        # 2 bps futures commission
        'slippage': 0.0001,   # 1 bp estimated slippage
        'total': 0.0003,      # 3 bps one-way, 6 bps round-trip
    },
}


class CostModel:
    """
    Transaction cost model for backtesting.

    Applies realistic trading costs including:
    - Trading fees (exchange/broker fees)
    - Slippage (market impact)
    """

    def __init__(self, market_type: MarketType, cost_multiplier: float = 1.0):
        """
        Initialize cost model.

        Args:
            market_type: Type of market (crypto, forex, stocks, commodities)
            cost_multiplier: Multiplier for sensitivity analysis (default 1.0)
        """
        self.market_type = market_type
        self.costs = TRANSACTION_COSTS[market_type]
        self.cost_multiplier = cost_multiplier

    @property
    def one_way_cost(self) -> float:
        """One-way transaction cost."""
        return self.costs['total'] * self.cost_multiplier

    @property
    def round_trip_cost(self) -> float:
        """Round-trip transaction cost (entry + exit)."""
        return self.one_way_cost * 2

    def apply_costs(self,
                    gross_returns: pd.Series,
                    positions: pd.Series) -> pd.Series:
        """
        Apply transaction costs to gross returns.

        Args:
            gross_returns: Gross strategy returns (before costs)
            positions: Position series (-1, 0, 1 or continuous)

        Returns:
            Net returns after transaction costs
        """
        # Detect trades: when position changes
        position_changes = positions.diff().fillna(0)
        trade_sizes = position_changes.abs()

        # Cost per trade (one-way cost for each direction change)
        trade_costs = trade_sizes * self.one_way_cost

        # Net returns
        net_returns = gross_returns - trade_costs

        return net_returns

    def calculate_cost_drag(self,
                           gross_returns: pd.Series,
                           positions: pd.Series) -> Dict:
        """
        Calculate detailed cost analysis.

        Returns:
            Dictionary with cost metrics
        """
        net_returns = self.apply_costs(gross_returns, positions)

        position_changes = positions.diff().fillna(0)
        trade_sizes = position_changes.abs()

        n_trades = (trade_sizes > 0).sum()
        total_turnover = trade_sizes.sum()
        total_cost = (gross_returns.sum() - net_returns.sum())

        return {
            'gross_return': float(gross_returns.sum()),
            'net_return': float(net_returns.sum()),
            'total_cost': float(total_cost),
            'cost_as_pct_of_gross': float(total_cost / (gross_returns.sum() + 1e-10)),
            'n_trades': int(n_trades),
            'total_turnover': float(total_turnover),
            'avg_cost_per_trade': float(total_cost / max(n_trades, 1)),
            'cost_bps_per_trade': float(self.one_way_cost * 10000),
        }


def get_cost_model(market_type: str, cost_multiplier: float = 1.0) -> CostModel:
    """
    Factory function to get cost model by market name.

    Args:
        market_type: Market name string ('crypto', 'forex', 'stocks', 'commodities')
        cost_multiplier: Multiplier for sensitivity analysis

    Returns:
        CostModel instance
    """
    market_map = {
        'crypto': MarketType.CRYPTO,
        'forex': MarketType.FOREX,
        'stocks': MarketType.STOCKS,
        'commodities': MarketType.COMMODITIES,
    }

    if market_type.lower() not in market_map:
        raise ValueError(f"Unknown market type: {market_type}")

    return CostModel(market_map[market_type.lower()], cost_multiplier)
