"""Bayesian Portfolio Optimization."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class BayesianPortfolio:
    """Bayesian approach to portfolio construction."""
    def __init__(self, prior_mean=None, prior_cov=None):
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.posterior_mean = None
        self.posterior_cov = None
    
    def update(self, returns):
        """Update posterior with new data."""
        n, p = returns.shape
        sample_mean = np.mean(returns, axis=0)
        sample_cov = np.cov(returns.T)
        
        if self.prior_mean is None:
            self.posterior_mean = sample_mean
            self.posterior_cov = sample_cov
        else:
            # Bayesian update
            prior_precision = np.linalg.inv(self.prior_cov)
            sample_precision = n * np.linalg.inv(sample_cov)
            
            self.posterior_cov = np.linalg.inv(prior_precision + sample_precision)
            self.posterior_mean = self.posterior_cov @ (
                prior_precision @ self.prior_mean + sample_precision @ sample_mean
            )
        
        return self.posterior_mean, self.posterior_cov
    
    def optimize(self, risk_aversion=1.0):
        """Compute optimal portfolio weights."""
        if self.posterior_mean is None:
            raise ValueError("Must update posterior first")
        
        weights = np.linalg.solve(
            risk_aversion * self.posterior_cov,
            self.posterior_mean
        )
        weights /= np.sum(weights)
        return weights
