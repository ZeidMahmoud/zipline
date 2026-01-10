"""Geometric Brownian Motion Simulator."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class GBMSimulator:
    """Simulate price paths using Geometric Brownian Motion."""
    
    def __init__(self, mu=0.1, sigma=0.2):
        self.mu = mu
        self.sigma = sigma
    
    def simulate(self, S0, T, dt, n_paths=1):
        """Simulate GBM paths."""
        n_steps = int(T / dt)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for t in range(1, n_steps + 1):
            z = np.random.standard_normal(n_paths)
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
            )
        
        return paths
    
    def estimate_parameters(self, prices):
        """Estimate mu and sigma from historical prices."""
        returns = np.diff(np.log(prices))
        self.mu = np.mean(returns) / (1/252)  # Annualized
        self.sigma = np.std(returns) * np.sqrt(252)
        return self.mu, self.sigma
