"""Monte Carlo Simulation Engine."""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import logging
logger = logging.getLogger(__name__)

class MonteCarloEngine:
    """General Monte Carlo simulation framework."""
    
    def __init__(self, n_simulations=10000, n_jobs=1, random_seed=None):
        self.n_simulations = n_simulations
        self.n_jobs = n_jobs
        self.random_seed = random_seed
    
    def simulate(self, simulation_func, *args, **kwargs):
        """Run Monte Carlo simulations."""
        if self.random_seed:
            np.random.seed(self.random_seed)
        
        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(
                    simulation_func,
                    [args] * self.n_simulations
                ))
        else:
            results = [simulation_func(*args, **kwargs) for _ in range(self.n_simulations)]
        
        return np.array(results)
    
    def estimate_var(self, returns, confidence=0.95):
        """Estimate Value at Risk."""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def estimate_cvar(self, returns, confidence=0.95):
        """Estimate Conditional Value at Risk (Expected Shortfall)."""
        var = self.estimate_var(returns, confidence)
        return np.mean(returns[returns <= var])
