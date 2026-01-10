"""Extreme Value Theory for Tail Risk."""
import numpy as np
from scipy import stats
import logging
logger = logging.getLogger(__name__)

class GEVDistribution:
    """Generalized Extreme Value distribution."""
    
    def __init__(self):
        self.params = None
    
    def fit(self, data):
        """Fit GEV to data."""
        self.params = stats.genextreme.fit(data)
        return self.params
    
    def var(self, confidence=0.95):
        """Calculate VaR using GEV."""
        if self.params is None:
            raise ValueError("Model must be fitted first")
        return stats.genextreme.ppf(confidence, *self.params)

class GPDModel:
    """Generalized Pareto Distribution for tail modeling."""
    
    def __init__(self, threshold=None):
        self.threshold = threshold
        self.params = None
    
    def fit(self, data, threshold_percentile=0.95):
        """Fit GPD to tail data."""
        if self.threshold is None:
            self.threshold = np.percentile(data, threshold_percentile * 100)
        
        exceedances = data[data > self.threshold] - self.threshold
        self.params = stats.genpareto.fit(exceedances)
        return self.params
    
    def var(self, confidence=0.99):
        """Calculate tail VaR."""
        if self.params is None:
            raise ValueError("Model must be fitted first")
        # Simplified calculation
        return self.threshold + stats.genpareto.ppf((confidence - 0.95) / 0.05, *self.params)
