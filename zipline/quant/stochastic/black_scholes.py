"""Black-Scholes Options Pricing."""
import numpy as np
from scipy.stats import norm
import logging
logger = logging.getLogger(__name__)

class BlackScholesModel:
    """Black-Scholes option pricing and Greeks calculation."""
    
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
    
    def price(self):
        """Calculate option price."""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.option_type == 'call':
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
    
    def delta(self):
        """Calculate delta."""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        return norm.cdf(d1) if self.option_type == 'call' else -norm.cdf(-d1)
    
    def gamma(self):
        """Calculate gamma."""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        """Calculate vega."""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        return self.S * norm.pdf(d1) * np.sqrt(self.T)
    
    def theta(self):
        """Calculate theta."""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.option_type == 'call':
            return (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) - 
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            return (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) + 
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))

class BlackScholesMerton(BlackScholesModel):
    """Black-Scholes-Merton with dividends."""
    
    def __init__(self, S, K, T, r, sigma, q=0, option_type='call'):
        super().__init__(S, K, T, r, sigma, option_type)
        self.q = q  # Dividend yield
