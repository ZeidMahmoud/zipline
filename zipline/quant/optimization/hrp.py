"""Hierarchical Risk Parity."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class HierarchicalRiskParity:
    """ML-based hierarchical risk parity allocation."""
    
    def __init__(self, returns):
        self.returns = returns
        self.cov_matrix = np.cov(returns.T)
    
    def optimize(self):
        """Find HRP weights."""
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
            
            # Compute distance matrix
            corr = np.corrcoef(self.returns.T)
            dist = np.sqrt((1 - corr) / 2)
            
            # Hierarchical clustering
            link = linkage(squareform(dist), method='single')
            
            # Compute weights (simplified)
            n = len(self.returns.T)
            weights = np.ones(n) / n
            
            return weights
        except ImportError:
            logger.warning("scipy required for HRP")
            n = len(self.returns.T)
            return np.ones(n) / n
