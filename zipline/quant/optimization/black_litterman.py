"""Black-Litterman Model."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class BlackLittermanModel:
    """Black-Litterman views-based allocation."""
    
    def __init__(self, market_caps, cov_matrix, risk_aversion=2.5):
        self.market_caps = np.array(market_caps)
        self.cov_matrix = np.array(cov_matrix)
        self.risk_aversion = risk_aversion
    
    def implied_returns(self):
        """Calculate market equilibrium returns."""
        market_weights = self.market_caps / np.sum(self.market_caps)
        return self.risk_aversion * self.cov_matrix @ market_weights
    
    def add_views(self, P, Q, omega):
        """Add views and compute posterior returns."""
        pi = self.implied_returns()
        tau = 0.05  # Scaling factor
        
        posterior_cov = np.linalg.inv(
            np.linalg.inv(tau * self.cov_matrix) + P.T @ np.linalg.inv(omega) @ P
        )
        posterior_returns = posterior_cov @ (
            np.linalg.inv(tau * self.cov_matrix) @ pi + P.T @ np.linalg.inv(omega) @ Q
        )
        
        return posterior_returns, posterior_cov
