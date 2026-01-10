"""Market Impact Models."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class AlmgrenChrissModel:
    """Optimal execution with market impact."""
    
    def __init__(self, total_shares, T, volatility, permanent_impact, temporary_impact, risk_aversion):
        self.total_shares = total_shares
        self.T = T
        self.sigma = volatility
        self.eta = permanent_impact
        self.epsilon = temporary_impact
        self.lambda_risk = risk_aversion
    
    def optimal_trajectory(self, n_intervals):
        """Calculate optimal execution trajectory."""
        tau = self.T / n_intervals
        kappa = np.sqrt(self.lambda_risk * self.sigma**2 / self.epsilon)
        
        trajectory = []
        for k in range(n_intervals + 1):
            t = k * tau
            shares_remaining = self.total_shares * np.sinh(kappa * (self.T - t)) / np.sinh(kappa * self.T)
            trajectory.append(shares_remaining)
        
        return np.array(trajectory)

class KyleModel:
    """Kyle's lambda market impact model."""
    
    def __init__(self, market_depth):
        self.market_depth = market_depth
    
    def calculate_impact(self, order_size):
        """Calculate price impact."""
        return order_size / self.market_depth
