"""Kelly Criterion for Position Sizing."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class KellyCriterion:
    """Optimal bet sizing using Kelly criterion."""
    
    def __init__(self, win_prob, win_loss_ratio):
        self.win_prob = win_prob
        self.win_loss_ratio = win_loss_ratio
    
    def calculate_fraction(self, fractional=0.5):
        """Calculate Kelly fraction."""
        kelly = (self.win_prob * self.win_loss_ratio - (1 - self.win_prob)) / self.win_loss_ratio
        return kelly * fractional  # Fractional Kelly for risk management
    
    def multi_asset_kelly(self, expected_returns, cov_matrix):
        """Multi-asset Kelly criterion."""
        inv_cov = np.linalg.inv(cov_matrix)
        kelly_weights = inv_cov @ expected_returns
        return kelly_weights
