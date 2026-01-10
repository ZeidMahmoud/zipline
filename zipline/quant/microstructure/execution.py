"""Execution Algorithms."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class TWAPExecutor:
    """Time-Weighted Average Price execution."""
    
    def __init__(self, total_shares, duration):
        self.total_shares = total_shares
        self.duration = duration
    
    def schedule(self, n_intervals):
        """Generate TWAP execution schedule."""
        shares_per_interval = self.total_shares / n_intervals
        return np.full(n_intervals, shares_per_interval)

class VWAPExecutor:
    """Volume-Weighted Average Price execution."""
    
    def __init__(self, total_shares, duration):
        self.total_shares = total_shares
        self.duration = duration
    
    def schedule(self, volume_profile):
        """Generate VWAP execution schedule based on volume profile."""
        total_volume = np.sum(volume_profile)
        return self.total_shares * (volume_profile / total_volume)
