"""Factor Risk Models."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class FactorRiskModel:
    """Multi-factor risk decomposition."""
    
    def __init__(self, factor_returns, asset_returns):
        self.factor_returns = factor_returns
        self.asset_returns = asset_returns
        self.factor_loadings = None
    
    def fit(self):
        """Estimate factor loadings."""
        from sklearn.linear_model import LinearRegression
        
        n_assets = self.asset_returns.shape[1]
        self.factor_loadings = np.zeros((n_assets, self.factor_returns.shape[1]))
        
        for i in range(n_assets):
            lr = LinearRegression()
            lr.fit(self.factor_returns, self.asset_returns[:, i])
            self.factor_loadings[i] = lr.coef_
        
        return self.factor_loadings
    
    def decompose_risk(self, portfolio_weights):
        """Decompose portfolio risk into factor contributions."""
        if self.factor_loadings is None:
            self.fit()
        
        factor_cov = np.cov(self.factor_returns.T)
        portfolio_factor_exposure = self.factor_loadings.T @ portfolio_weights
        factor_risk = portfolio_factor_exposure @ factor_cov @ portfolio_factor_exposure
        
        return factor_risk
