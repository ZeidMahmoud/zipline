"""
Value at Risk (VaR) calculations for portfolio risk management.

This module provides various methods for calculating VaR including
historical, parametric, and Monte Carlo approaches.
"""
from typing import Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
import logging

log = logging.getLogger(__name__)


class HistoricalVaR:
    """
    Historical Value at Risk calculation.
    
    Calculates VaR using historical returns distribution without
    assuming any particular distribution shape.
    
    Parameters
    ----------
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.95 for 95% VaR).
        Default is 0.95.
    
    Examples
    --------
    >>> var_calc = HistoricalVaR(confidence_level=0.95)
    >>> returns = np.random.randn(1000) * 0.02
    >>> var = var_calc.calculate(returns)
    >>> print(f"95% VaR: {var:.2%}")
    """
    
    def __init__(self, confidence_level: float = 0.95):
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        self.confidence_level = confidence_level
    
    def calculate(self, returns: Union[np.ndarray, pd.Series],
                 portfolio_value: float = 1.0) -> float:
        """
        Calculate historical VaR.
        
        Parameters
        ----------
        returns : array-like
            Historical returns.
        portfolio_value : float, optional
            Current portfolio value. Default is 1.0.
            
        Returns
        -------
        float
            VaR value (positive number representing potential loss).
        """
        if len(returns) == 0:
            return np.nan
        
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return np.nan
        
        # Calculate the percentile corresponding to our confidence level
        percentile = (1 - self.confidence_level) * 100
        var_return = np.percentile(returns, percentile)
        
        # Convert to dollar VaR
        var = -var_return * portfolio_value
        
        return var
    
    def calculate_cvar(self, returns: Union[np.ndarray, pd.Series],
                      portfolio_value: float = 1.0) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        CVaR is the expected loss given that the loss exceeds VaR.
        
        Parameters
        ----------
        returns : array-like
            Historical returns.
        portfolio_value : float, optional
            Current portfolio value. Default is 1.0.
            
        Returns
        -------
        float
            CVaR value.
        """
        if len(returns) == 0:
            return np.nan
        
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return np.nan
        
        # Find VaR threshold
        percentile = (1 - self.confidence_level) * 100
        var_return = np.percentile(returns, percentile)
        
        # Calculate mean of returns below VaR
        tail_returns = returns[returns <= var_return]
        
        if len(tail_returns) == 0:
            return np.nan
        
        cvar_return = np.mean(tail_returns)
        cvar = -cvar_return * portfolio_value
        
        return cvar


class ParametricVaR:
    """
    Parametric (Variance-Covariance) Value at Risk.
    
    Assumes returns follow a normal distribution and calculates VaR
    based on mean and standard deviation.
    
    Parameters
    ----------
    confidence_level : float, optional
        Confidence level for VaR. Default is 0.95.
    
    Examples
    --------
    >>> var_calc = ParametricVaR(confidence_level=0.99)
    >>> returns = np.random.randn(1000) * 0.02
    >>> var = var_calc.calculate(returns)
    """
    
    def __init__(self, confidence_level: float = 0.95):
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        self.confidence_level = confidence_level
    
    def calculate(self, returns: Union[np.ndarray, pd.Series],
                 portfolio_value: float = 1.0) -> float:
        """
        Calculate parametric VaR.
        
        Parameters
        ----------
        returns : array-like
            Historical returns.
        portfolio_value : float, optional
            Current portfolio value. Default is 1.0.
            
        Returns
        -------
        float
            VaR value.
        """
        if len(returns) == 0:
            return np.nan
        
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return np.nan
        
        # Calculate mean and std of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Get z-score for confidence level
        z_score = stats.norm.ppf(1 - self.confidence_level)
        
        # Calculate VaR
        var_return = mean_return + z_score * std_return
        var = -var_return * portfolio_value
        
        return var


class MonteCarloVaR:
    """
    Monte Carlo Value at Risk simulation.
    
    Generates multiple simulated return paths and calculates VaR
    from the distribution of simulated outcomes.
    
    Parameters
    ----------
    confidence_level : float, optional
        Confidence level for VaR. Default is 0.95.
    n_simulations : int, optional
        Number of Monte Carlo simulations. Default is 10000.
    time_horizon : int, optional
        Time horizon in days. Default is 1.
    
    Examples
    --------
    >>> var_calc = MonteCarloVaR(n_simulations=50000)
    >>> returns = np.random.randn(1000) * 0.02
    >>> var = var_calc.calculate(returns)
    """
    
    def __init__(self, confidence_level: float = 0.95,
                 n_simulations: int = 10000,
                 time_horizon: int = 1):
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        self.confidence_level = confidence_level
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
    
    def calculate(self, returns: Union[np.ndarray, pd.Series],
                 portfolio_value: float = 1.0,
                 random_seed: Optional[int] = None) -> float:
        """
        Calculate Monte Carlo VaR.
        
        Parameters
        ----------
        returns : array-like
            Historical returns for parameter estimation.
        portfolio_value : float, optional
            Current portfolio value. Default is 1.0.
        random_seed : int, optional
            Random seed for reproducibility.
            
        Returns
        -------
        float
            VaR value.
        """
        if len(returns) == 0:
            return np.nan
        
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return np.nan
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Estimate parameters from historical returns
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Simulate returns for the time horizon
        simulated_returns = np.random.normal(
            mean_return * self.time_horizon,
            std_return * np.sqrt(self.time_horizon),
            self.n_simulations
        )
        
        # Calculate portfolio values
        simulated_values = portfolio_value * (1 + simulated_returns)
        simulated_losses = portfolio_value - simulated_values
        
        # Calculate VaR as percentile of losses
        percentile = self.confidence_level * 100
        var = np.percentile(simulated_losses, percentile)
        
        return var
    
    def calculate_cvar(self, returns: Union[np.ndarray, pd.Series],
                      portfolio_value: float = 1.0,
                      random_seed: Optional[int] = None) -> float:
        """
        Calculate Monte Carlo CVaR (Expected Shortfall).
        
        Parameters
        ----------
        returns : array-like
            Historical returns for parameter estimation.
        portfolio_value : float, optional
            Current portfolio value. Default is 1.0.
        random_seed : int, optional
            Random seed for reproducibility.
            
        Returns
        -------
        float
            CVaR value.
        """
        if len(returns) == 0:
            return np.nan
        
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return np.nan
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Estimate parameters from historical returns
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Simulate returns for the time horizon
        simulated_returns = np.random.normal(
            mean_return * self.time_horizon,
            std_return * np.sqrt(self.time_horizon),
            self.n_simulations
        )
        
        # Calculate portfolio values
        simulated_values = portfolio_value * (1 + simulated_returns)
        simulated_losses = portfolio_value - simulated_values
        
        # Calculate VaR threshold
        percentile = self.confidence_level * 100
        var_threshold = np.percentile(simulated_losses, percentile)
        
        # Calculate CVaR as mean of losses exceeding VaR
        tail_losses = simulated_losses[simulated_losses >= var_threshold]
        cvar = np.mean(tail_losses) if len(tail_losses) > 0 else np.nan
        
        return cvar
