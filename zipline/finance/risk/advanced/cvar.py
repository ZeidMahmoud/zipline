"""Expected Shortfall (CVaR)."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class ExpectedShortfall:
    """CVaR calculation and optimization."""
    
    def __init__(self):
        pass
    
    def historical_cvar(self, returns, confidence=0.95):
        """Calculate historical CVaR."""
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = np.mean(returns[returns <= var])
        return cvar
    
    def parametric_cvar(self, mu, sigma, confidence=0.95):
        """Calculate parametric CVaR assuming normal distribution."""
        from scipy.stats import norm
        var = mu + sigma * norm.ppf(1 - confidence)
        cvar = mu - sigma * norm.pdf(norm.ppf(1 - confidence)) / (1 - confidence)
        return cvar
