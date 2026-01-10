"""
Reinforcement Learning for Algorithmic Trading.

This module provides RL agents and environments for learning optimal
trading strategies through interaction with market simulations.
"""

try:
    from .dqn_trader import DQNTrader
    from .policy_gradient import A2CTrader, PPOTrader
    from .multi_agent import MultiAgentMarket
    from .environment import TradingEnvironment
    
    __all__ = [
        'DQNTrader',
        'A2CTrader',
        'PPOTrader',
        'MultiAgentMarket',
        'TradingEnvironment',
    ]
except ImportError:
    # RL dependencies are optional
    __all__ = []
