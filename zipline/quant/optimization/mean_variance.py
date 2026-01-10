"""Markowitz Mean-Variance Optimization."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class MeanVarianceOptimizer:
    """Classic mean-variance portfolio optimization."""
    
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.0):
        self.expected_returns = np.array(expected_returns)
        self.cov_matrix = np.array(cov_matrix)
        self.risk_free_rate = risk_free_rate
    
    def optimize(self, target_return=None, constraints=None):
        """Find optimal portfolio weights."""
        try:
            import cvxpy as cp
            
            n = len(self.expected_returns)
            w = cp.Variable(n)
            
            # Objective: minimize variance
            objective = cp.Minimize(cp.quad_form(w, self.cov_matrix))
            
            # Constraints
            constraints_list = [cp.sum(w) == 1]
            
            if target_return is not None:
                constraints_list.append(w @ self.expected_returns >= target_return)
            
            if constraints and 'long_only' in constraints:
                constraints_list.append(w >= 0)
            
            problem = cp.Problem(objective, constraints_list)
            problem.solve()
            
            return w.value
            
        except ImportError:
            logger.error("cvxpy required for optimization")
            # Fallback: equal weights
            return np.ones(len(self.expected_returns)) / len(self.expected_returns)
    
    def efficient_frontier(self, n_points=50):
        """Calculate efficient frontier."""
        returns = []
        risks = []
        
        min_ret = np.min(self.expected_returns)
        max_ret = np.max(self.expected_returns)
        
        for target_ret in np.linspace(min_ret, max_ret, n_points):
            try:
                weights = self.optimize(target_return=target_ret)
                port_ret = weights @ self.expected_returns
                port_risk = np.sqrt(weights @ self.cov_matrix @ weights)
                returns.append(port_ret)
                risks.append(port_risk)
            except:
                continue
        
        return np.array(risks), np.array(returns)
