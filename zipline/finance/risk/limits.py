"""
Risk limits and position sizing for portfolio risk management.

This module provides risk controls including drawdown limits,
volatility limits, correlation limits, and position sizing.
"""
from typing import Dict, Optional, Callable
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


class MaxDrawdownLimit:
    """
    Maximum drawdown risk limit.
    
    Stops trading or reduces positions when portfolio drawdown
    exceeds a specified threshold.
    
    Parameters
    ----------
    max_drawdown : float
        Maximum allowed drawdown as a decimal (e.g., 0.20 for 20%).
    action : str, optional
        Action to take when limit is breached: 'stop' or 'reduce'.
        Default is 'stop'.
    
    Examples
    --------
    >>> limit = MaxDrawdownLimit(max_drawdown=0.15)
    >>> portfolio_values = [100000, 95000, 90000, 85000]
    >>> is_breached = limit.check(portfolio_values)
    """
    
    def __init__(self, max_drawdown: float, action: str = 'stop'):
        if not 0 < max_drawdown < 1:
            raise ValueError("max_drawdown must be between 0 and 1")
        if action not in ['stop', 'reduce']:
            raise ValueError("action must be 'stop' or 'reduce'")
        
        self.max_drawdown = max_drawdown
        self.action = action
        self._peak = None
    
    def check(self, portfolio_values: np.ndarray) -> bool:
        """
        Check if drawdown limit is breached.
        
        Parameters
        ----------
        portfolio_values : array-like
            Historical portfolio values.
            
        Returns
        -------
        bool
            True if limit is breached, False otherwise.
        """
        if len(portfolio_values) == 0:
            return False
        
        values = np.asarray(portfolio_values)
        
        # Calculate running peak
        if self._peak is None:
            self._peak = values[0]
        
        current_peak = np.maximum.accumulate(values)
        self._peak = current_peak[-1]
        
        # Calculate current drawdown
        current_value = values[-1]
        current_drawdown = (self._peak - current_value) / self._peak
        
        is_breached = current_drawdown > self.max_drawdown
        
        if is_breached:
            log.warning(
                f"Drawdown limit breached: {current_drawdown:.2%} > {self.max_drawdown:.2%}"
            )
        
        return is_breached
    
    def reset(self):
        """Reset the peak value."""
        self._peak = None


class VolatilityLimit:
    """
    Volatility-based risk limit.
    
    Reduces exposure when volatility exceeds a threshold,
    implements volatility-based position sizing.
    
    Parameters
    ----------
    max_volatility : float
        Maximum allowed volatility (annualized standard deviation).
    window : int, optional
        Lookback window for volatility calculation. Default is 20.
    target_volatility : float, optional
        Target volatility for position sizing. If None, only checks limits.
    
    Examples
    --------
    >>> limit = VolatilityLimit(max_volatility=0.25, target_volatility=0.15)
    >>> returns = np.random.randn(100) * 0.02
    >>> is_breached = limit.check(returns)
    >>> scale = limit.get_position_scale(returns)
    """
    
    def __init__(self, max_volatility: float, window: int = 20,
                 target_volatility: Optional[float] = None):
        self.max_volatility = max_volatility
        self.window = window
        self.target_volatility = target_volatility
    
    def check(self, returns: np.ndarray) -> bool:
        """
        Check if volatility limit is breached.
        
        Parameters
        ----------
        returns : array-like
            Historical returns.
            
        Returns
        -------
        bool
            True if limit is breached, False otherwise.
        """
        current_vol = self.calculate_volatility(returns)
        
        is_breached = current_vol > self.max_volatility
        
        if is_breached:
            log.warning(
                f"Volatility limit breached: {current_vol:.2%} > {self.max_volatility:.2%}"
            )
        
        return is_breached
    
    def calculate_volatility(self, returns: np.ndarray) -> float:
        """
        Calculate annualized volatility.
        
        Parameters
        ----------
        returns : array-like
            Historical returns.
            
        Returns
        -------
        float
            Annualized volatility.
        """
        if len(returns) == 0:
            return 0.0
        
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 2:
            return 0.0
        
        # Use only the most recent window
        recent_returns = returns[-self.window:]
        
        # Calculate daily volatility
        daily_vol = np.std(recent_returns, ddof=1)
        
        # Annualize (assuming 252 trading days)
        annualized_vol = daily_vol * np.sqrt(252)
        
        return annualized_vol
    
    def get_position_scale(self, returns: np.ndarray) -> float:
        """
        Get position scaling factor based on volatility.
        
        Parameters
        ----------
        returns : array-like
            Historical returns.
            
        Returns
        -------
        float
            Position scaling factor (1.0 = full size, 0.5 = half size, etc.).
        """
        if self.target_volatility is None:
            return 1.0
        
        current_vol = self.calculate_volatility(returns)
        
        if current_vol == 0:
            return 1.0
        
        # Scale positions to target volatility
        scale = self.target_volatility / current_vol
        
        # Cap at maximum scaling factor of 2.0
        scale = min(scale, 2.0)
        
        return scale


class CorrelationLimit:
    """
    Correlation-based risk limit.
    
    Limits positions in highly correlated assets to reduce
    concentration risk.
    
    Parameters
    ----------
    max_correlation : float
        Maximum allowed correlation between positions.
    window : int, optional
        Lookback window for correlation calculation. Default is 60.
    
    Examples
    --------
    >>> limit = CorrelationLimit(max_correlation=0.7)
    >>> returns_dict = {
    ...     'AAPL': np.random.randn(100) * 0.02,
    ...     'MSFT': np.random.randn(100) * 0.02,
    ... }
    >>> violations = limit.check_portfolio(returns_dict)
    """
    
    def __init__(self, max_correlation: float, window: int = 60):
        if not -1 <= max_correlation <= 1:
            raise ValueError("max_correlation must be between -1 and 1")
        self.max_correlation = max_correlation
        self.window = window
    
    def check_pair(self, returns1: np.ndarray, returns2: np.ndarray) -> bool:
        """
        Check correlation between two assets.
        
        Parameters
        ----------
        returns1 : array-like
            Returns for first asset.
        returns2 : array-like
            Returns for second asset.
            
        Returns
        -------
        bool
            True if correlation exceeds limit, False otherwise.
        """
        corr = self.calculate_correlation(returns1, returns2)
        
        is_breached = abs(corr) > abs(self.max_correlation)
        
        if is_breached:
            log.warning(
                f"Correlation limit breached: {corr:.2f} > {self.max_correlation:.2f}"
            )
        
        return is_breached
    
    def calculate_correlation(self, returns1: np.ndarray, 
                             returns2: np.ndarray) -> float:
        """
        Calculate correlation between two return series.
        
        Parameters
        ----------
        returns1 : array-like
            Returns for first asset.
        returns2 : array-like
            Returns for second asset.
            
        Returns
        -------
        float
            Correlation coefficient.
        """
        r1 = np.asarray(returns1)[-self.window:]
        r2 = np.asarray(returns2)[-self.window:]
        
        # Align lengths
        min_len = min(len(r1), len(r2))
        r1 = r1[-min_len:]
        r2 = r2[-min_len:]
        
        if len(r1) < 2:
            return 0.0
        
        # Remove NaN values
        mask = ~(np.isnan(r1) | np.isnan(r2))
        r1 = r1[mask]
        r2 = r2[mask]
        
        if len(r1) < 2:
            return 0.0
        
        corr = np.corrcoef(r1, r2)[0, 1]
        
        return corr if not np.isnan(corr) else 0.0
    
    def check_portfolio(self, returns_dict: Dict[str, np.ndarray]) -> list:
        """
        Check correlations across entire portfolio.
        
        Parameters
        ----------
        returns_dict : dict
            Dictionary mapping asset names to return arrays.
            
        Returns
        -------
        list
            List of tuples (asset1, asset2, correlation) for violations.
        """
        violations = []
        assets = list(returns_dict.keys())
        
        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                corr = self.calculate_correlation(
                    returns_dict[asset1],
                    returns_dict[asset2]
                )
                
                if abs(corr) > abs(self.max_correlation):
                    violations.append((asset1, asset2, corr))
        
        return violations


class PositionSizer:
    """
    Volatility-adjusted position sizing.
    
    Sizes positions inversely to volatility to maintain consistent
    risk across different assets.
    
    Parameters
    ----------
    target_volatility : float
        Target portfolio volatility (annualized).
    window : int, optional
        Lookback window for volatility calculation. Default is 20.
    max_position_size : float, optional
        Maximum position size as fraction of portfolio. Default is 0.2 (20%).
    
    Examples
    --------
    >>> sizer = PositionSizer(target_volatility=0.15)
    >>> returns = np.random.randn(100) * 0.03
    >>> size = sizer.calculate_size(returns, portfolio_value=100000)
    """
    
    def __init__(self, target_volatility: float, window: int = 20,
                 max_position_size: float = 0.2):
        self.target_volatility = target_volatility
        self.window = window
        self.max_position_size = max_position_size
    
    def calculate_size(self, returns: np.ndarray, 
                      portfolio_value: float) -> float:
        """
        Calculate position size based on volatility.
        
        Parameters
        ----------
        returns : array-like
            Historical returns for the asset.
        portfolio_value : float
            Current portfolio value.
            
        Returns
        -------
        float
            Dollar value to allocate to this position.
        """
        if len(returns) == 0:
            return 0.0
        
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 2:
            return 0.0
        
        # Use recent returns
        recent_returns = returns[-self.window:]
        
        # Calculate daily volatility
        daily_vol = np.std(recent_returns, ddof=1)
        
        if daily_vol == 0:
            return 0.0
        
        # Annualize
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Calculate position size to achieve target volatility
        # position_size * asset_vol = target_vol
        fraction = self.target_volatility / annualized_vol
        
        # Apply max position limit
        fraction = min(fraction, self.max_position_size)
        
        position_value = portfolio_value * fraction
        
        return position_value
