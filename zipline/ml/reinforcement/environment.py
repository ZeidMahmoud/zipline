"""
OpenAI Gym-compatible Trading Environment.
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class TradingEnvironment:
    """
    Trading environment compatible with OpenAI Gym interface.
    
    Parameters
    ----------
    data : np.ndarray
        Historical market data
    initial_balance : float, optional
        Starting cash balance (default: 100000)
    transaction_cost : float, optional
        Transaction cost as fraction (default: 0.001)
    
    Attributes
    ----------
    state_dim : int
        Dimension of observation space
    action_dim : int
        Dimension of action space
    """
    
    def __init__(self, data, initial_balance=100000, transaction_cost=0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        self.state_dim = data.shape[1] if len(data.shape) > 1 else 1
        self.action_dim = 3  # buy, hold, sell
        
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        
        logger.info(f"TradingEnvironment initialized with {len(data)} timesteps")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if len(self.data.shape) > 1:
            obs = self.data[self.current_step]
        else:
            obs = np.array([self.data[self.current_step]])
        return np.append(obs, [self.balance, self.position])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action in environment.
        
        Returns
        -------
        observation : np.ndarray
            Next state
        reward : float
            Reward for the action
        done : bool
            Whether episode is complete
        info : dict
            Additional information
        """
        # Execute trade
        price = self.data[self.current_step] if len(self.data.shape) == 1 else self.data[self.current_step, 0]
        
        if action == 0:  # buy
            shares = (self.balance * 0.1) / price
            cost = shares * price * (1 + self.transaction_cost)
            if cost <= self.balance:
                self.position += shares
                self.balance -= cost
        elif action == 2:  # sell
            if self.position > 0:
                proceeds = self.position * price * (1 - self.transaction_cost)
                self.balance += proceeds
                self.position = 0
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Calculate reward
        portfolio_value = self.balance + self.position * price
        reward = (portfolio_value - self.initial_balance) / self.initial_balance
        
        obs = self._get_observation() if not done else np.zeros(self.state_dim + 2)
        
        return obs, reward, done, {}
