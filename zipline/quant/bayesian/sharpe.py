"""Probabilistic Sharpe Ratio."""
import numpy as np
from scipy import stats
import logging
logger = logging.getLogger(__name__)

class ProbabilisticSharpe:
    """Probabilistic Sharpe ratio estimation."""
    
    def __init__(self, benchmark_sr=0.0):
        self.benchmark_sr = benchmark_sr
    
    def calculate(self, returns):
        """Calculate probabilistic Sharpe ratio."""
        n = len(returns)
        sr = np.mean(returns) / np.std(returns)
        
        # Assuming returns are i.i.d.
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        # Standard error of Sharpe ratio
        se_sr = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt - 3) * sr**2 / 4) / n)
        
        # Probability that SR > benchmark_sr
        z_score = (sr - self.benchmark_sr) / se_sr
        prob = stats.norm.cdf(z_score)
        
        return {
            'sharpe_ratio': sr,
            'prob_sr_positive': prob,
            'standard_error': se_sr
        }
