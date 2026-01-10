"""Jump Diffusion Models."""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class MertonJumpDiffusion:
    """Merton's jump-diffusion model."""
    def __init__(self, mu=0.1, sigma=0.2, lambda_jump=0.1, mu_jump=0, sigma_jump=0.05):
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
    
    def simulate(self, S0, T, dt, n_paths=1):
        """Simulate jump diffusion paths."""
        n_steps = int(T / dt)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for t in range(1, n_steps + 1):
            z = np.random.standard_normal(n_paths)
            n_jumps = np.random.poisson(self.lambda_jump * dt, n_paths)
            jump_sizes = np.random.normal(self.mu_jump, self.sigma_jump, n_paths) * n_jumps
            
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * dt + 
                self.sigma * np.sqrt(dt) * z + jump_sizes
            )
        
        return paths

class KouJumpDiffusion:
    """Kou's double exponential jump-diffusion model."""
    def __init__(self, mu=0.1, sigma=0.2, lambda_jump=0.1):
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        logger.info("KouJumpDiffusion initialized")
