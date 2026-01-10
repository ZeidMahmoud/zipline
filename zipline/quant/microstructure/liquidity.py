"""Liquidity Analysis Tools."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class LiquidityAnalyzer:
    """Analyze market liquidity."""
    
    def __init__(self):
        pass
    
    def calculate_spread(self, bid, ask):
        """Calculate bid-ask spread."""
        return ask - bid
    
    def calculate_percentage_spread(self, bid, ask):
        """Calculate percentage spread."""
        mid = (bid + ask) / 2
        return (ask - bid) / mid

class AmihudIlliquidity:
    """Amihud illiquidity measure."""
    
    def __init__(self):
        pass
    
    def calculate(self, returns, volumes):
        """Calculate Amihud illiquidity."""
        return np.mean(np.abs(returns) / volumes)
