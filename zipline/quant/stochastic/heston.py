"""Stochastic Volatility Models."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class HestonModel:
    """Heston stochastic volatility model."""
    def __init__(self, S0=100, v0=0.04, r=0.05, kappa=2, theta=0.04, sigma=0.3, rho=-0.7):
        self.S0 = S0
        self.v0 = v0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
    
    def simulate(self, T, dt, n_paths=1):
        """Simulate Heston paths using Euler discretization."""
        n_steps = int(T / dt)
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        for t in range(1, n_steps + 1):
            z1 = np.random.standard_normal(n_paths)
            z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.standard_normal(n_paths)
            
            v[:, t] = np.maximum(v[:, t-1] + self.kappa * (self.theta - v[:, t-1]) * dt + 
                                 self.sigma * np.sqrt(v[:, t-1] * dt) * z2, 0)
            S[:, t] = S[:, t-1] * np.exp((self.r - 0.5 * v[:, t-1]) * dt + 
                                          np.sqrt(v[:, t-1] * dt) * z1)
        
        return S, v

class SABRModel:
    """SABR (Stochastic Alpha Beta Rho) model."""
    def __init__(self, alpha=0.3, beta=0.5, rho=-0.3, nu=0.4):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        logger.info("SABRModel initialized")
